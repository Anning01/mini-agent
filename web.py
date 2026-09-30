"""本地网页入口：接收用户消息，通过 SSE 持续发送 Agent 状态和最终回答。"""

import json
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from agent import Agent


class WebHandler(BaseHTTPRequestHandler):
    """只提供一个页面和三个接口，不开放项目目录中的其他文件。"""

    def send_json(self, status: int, data: dict):
        # ASCII 转义也能安全传输模型可能返回的孤立代理字符，浏览器会还原中文。
        body = json.dumps(data, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_event(self, name: str, data: dict):
        """写入一条 SSE 事件；空行是事件边界，flush 让浏览器立即收到数据。"""
        # JSON 内的换行会转义，不会与 SSE 的空行分隔符混淆。
        payload = json.dumps(data, ensure_ascii=True)
        self.wfile.write(f"event: {name}\ndata: {payload}\n\n".encode("utf-8"))
        self.wfile.flush()

    def do_GET(self):
        if self.path == "/":
            body = Path(__file__).with_name("index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/skills":
            self.send_json(200, {"skills": [
                {"name": name, "description": info["description"]}
                for name, info in self.server.agent.skills.index.items()
            ]})
        else:
            self.send_json(404, {"error": "没有这个页面或接口"})

    def do_POST(self):
        # 只接受从本地页面发来的 JSON 请求，避免其他网站借浏览器调用本地 Agent。
        host = self.headers.get("Host", "")
        port = self.server.server_port
        origin = self.headers.get("Origin")
        if host not in {f"127.0.0.1:{port}", f"localhost:{port}"} or (
            origin is not None and origin != f"http://{host}"
        ):
            self.send_json(403, {"error": "请从本地网页访问"})
            return
        if self.path not in {"/api/chat", "/api/reset"}:
            self.send_json(404, {"error": "没有这个接口"})
            return
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            self.send_json(415, {"error": "请求必须使用 application/json"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                raise ValueError("请求内容不能为空，且不能超过 64 KB")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("请求内容必须是 JSON 对象")
            if self.path == "/api/chat":
                message = data.get("message")
                skill_name = data.get("skill_name")
                if not isinstance(message, str) or not message.strip():
                    raise ValueError("请输入消息")
                if skill_name is not None and (
                    not isinstance(skill_name, str)
                    or skill_name not in self.server.agent.skills.index
                ):
                    raise ValueError("请选择已登记的 Skill")
        except (ValueError, UnicodeDecodeError) as error:
            self.send_json(400, {"error": str(error)})
            return

        agent = self.server.agent
        if self.path == "/api/reset":
            # 仅清空当前会话；SQLite 中的长期记忆继续保留。
            agent.history.clear()
            agent.state = None
            self.send_json(200, {"ok": True})
            return

        # 输入校验失败仍返回普通 JSON；校验通过后只发送一次 SSE 响应头。
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        # 长度事先未知，用关闭连接标记响应结束，保持现有 HTTP/1.0 服务即可。
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        # 先清除旧状态；run 会立即挂载本轮状态。
        agent.state = None
        try:
            # 传入绑定方法；Agent 每次调用 on_event，服务就发出一条 SSE 事件。
            answer = agent.run(
                message.strip(), skill_name=skill_name, on_event=self.send_event,
            )
            self.send_event("done", {"answer": answer, "state": agent.state.model_dump()})
        except Exception as error:
            if agent.state is not None and agent.state.status == "running":
                agent.state.status = "failed"
                for step in agent.state.plan.steps if agent.state.plan else []:
                    if step.status == "running":
                        step.status = "failed"
            if isinstance(error, (BrokenPipeError, ConnectionResetError)):
                print("浏览器连接已断开，停止本次事件发送。")
                return
            traceback.print_exc()
            # 响应头已发出，不能再发送 HTTP 500；通过 error 事件报告执行错误。
            try:
                self.send_event("error", {
                    "error": str(error),
                    "state": agent.state.model_dump() if agent.state else None,
                })
            except (BrokenPipeError, ConnectionResetError):
                pass  # 客户端已经断开，不能再向同一连接报告错误。


if __name__ == "__main__":
    # ponytail: 一个本地会话、串行处理；需要多人使用时再增加会话隔离和任务队列。
    # HTTPServer 在同一线程执行 Agent，沿用现有 SQLite 连接的线程限制。
    server = HTTPServer(("127.0.0.1", 8000), WebHandler)
    server.agent = Agent()
    print("MiniAgent 网页已启动：http://127.0.0.1:8000（Ctrl+C 停止）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.agent.memory_store.conn.close()

"""LLM 服务层：封装与 OpenAI 兼容接口的单次对话请求。"""

import os

from dotenv import load_dotenv
from openai import OpenAI

# 从 .env 读取 API_KEY、BASE_URL 和 MODEL_NAME 等配置。
# 凭证不应写进代码，也不应提交到版本控制。
load_dotenv()


class LLM:
    """向模型服务发送 Chat Completions 请求的薄封装。"""

    def __init__(self):
        # 该客户端既可连接官方 API，也可连接兼容 Chat Completions 协议的服务。
        self.openai = OpenAI(
            api_key=os.environ.get("API_KEY"),
            base_url=os.environ.get("BASE_URL"),
        )

    def chat(self, messages: list, tools=None):
        """发送一轮模型请求，并返回 SDK 的 assistant message 对象。

        Args:
            messages: 按 role 组织好的本轮上下文。
            tools: 可选的函数工具 schema；传入后模型可以返回 tool_calls。
        """

        response = self.openai.chat.completions.create(
            # 模型和连接配置来自环境变量，避免把提供商或模型名称写死在业务代码中。
            model=os.environ.get("MODEL_NAME"),
            messages=messages,
            tools=tools,
            # 非流式请求：等待完整响应后，才交给 Agent 处理。
            stream=False,
            # 这两项需要实际服务支持，不是所有兼容接口都接受；接入说明见 README。
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}},
        )

        # Agent 会自行判断该 message 是最终文本回答，还是包含 tool_calls。
        return response.choices[0].message

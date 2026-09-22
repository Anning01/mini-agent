"""长期记忆提取器。

本模块只负责调用 LLM，从对话中筛选值得长期保存的键值对；SQLite 的读写职责
由 memory_store.py 单独承担。
"""

import json

from llm import LLM


class MemoryManager:
    """从用户对话中提取长期记忆的业务逻辑。"""

    def __init__(self):
        # 这是早期的内存示例；当前 Agent 的长期记忆实际由 MemoryStore 保存到 SQLite。
        self.memories = []

        # 此处创建了独立客户端；独立请求也可以复用客户端，这不是必须分开的要求。
        self.llm = LLM()

    def add(self, content: str):
        """向旧的内存列表示例追加一条内容。"""

        self.memories.append(content)

    def get_all(self):
        """返回旧的内存列表示例中的全部内容。"""

        return self.memories

    def extract(self, messages: list):
        """从消息列表提取值得长期保存的用户信息。

        Args:
            messages: 需要分析的对话消息；MVP 通常只传当前用户消息。

        Returns:
            模型返回并解析后的字典，例如 ``{"职业": "Python 开发者"}``。
        """

        # 提示词要求返回 JSON，但属于文字约定，不能保证模型一定遵守。
        prompt = """
        你是一个 Memory Extractor。

        请从下面的对话中提取值得长期记忆的信息。

        只提取：
        - 用户身份信息
        - 用户长期偏好
        - 用户长期目标
        - 用户正在进行的长期项目
        - 用户明确要求记住的信息

        不要提取：
        - 临时问题
        - 一次性信息
        - 当前天气
        - 普通闲聊
        - 没有长期价值的信息

        如果没有值得记忆的信息，返回 {}。

        只返回 JSON，不要返回其他文字。

        格式：
        {
            "key": "value"
        }
        """
        # 将原始消息序列化成 JSON 文本，作为“待提取材料”发送给模型。
        # ensure_ascii=False 让中文以可读形式传递，而不是变成 \uXXXX 转义。
        memory_messages = [
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": json.dumps(
                    messages,
                    ensure_ascii=False,
                ),
            },
        ]

        # 这里调用的是“记忆提取”请求，不是正常回答用户问题的请求。
        response = self.llm.chat(memory_messages)

        # json.loads 只负责解析 JSON，不保证结果一定是 dict[str, str]。
        # 非法 JSON 会在这里抛异常；列表或嵌套值可能在后续保存时出错。
        return json.loads(response.content)


if __name__ == "__main__":
    # 可独立运行本文件，验证样例对话的记忆提取结果。
    memory = MemoryManager()

    messages = [
        {
            "role": "user",
            "content": "我叫张三，我是一名 Python 开发者。",
        },
        {
            "role": "assistant",
            "content": "你好张三。",
        },
        {
            "role": "user",
            "content": "我最近正在学习 Agent，希望以后自己从底层实现，不想直接使用 LangChain。",
        },
    ]

    result = memory.extract(messages)

    # 预期输出类似：{'姓名': '张三', '职业': 'Python 开发者', ...}
    print(result)

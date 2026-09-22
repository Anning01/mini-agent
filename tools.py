"""Agent 可调用工具的定义，以及提供给 LLM 的工具 schema。"""


class Tools:
    """维护工具函数注册表，并输出 OpenAI 兼容的 function-calling 描述。"""

    def __init__(self):
        # registry 是运行时路由表：模型给出工具名后，Agent 用它找到真实 Python 函数。
        self.registry = {
            "get_weather": self.get_weather,
        }

    def get_weather(self, city: str):
        """返回指定城市的天气文本。

        当前是测试桩（stub）：固定文案只用于验证工具调用链，不代表实时天气。
        后续可以保持函数签名不变，只替换内部实现为真实天气 API。
        """

        return f"""
            **{city}今天（2026年9月21日，星期一）天气晴朗，气温适宜，总体不错。**

            - **当前实况**（上午时段）：气温约25–26℃，东北风或南风3级左右，相对湿度约59%，气压约1016 hPa，无降水。
            - **全天预报**：白天晴，最高气温约29℃；夜间晴，最低气温约18℃。风力微风（南风或西南风，小于3级）。

            早晚温差较大（约10℃以上），建议适时增减衣物，注意保暖防感冒。白天阳光充足，紫外线适中，外出可适当防晒。空气质量整体较好。

            数据来源于中国气象局、中央气象台及相关权威预报，实际天气可能有细微变化，建议出行前再确认最新实况。
        """

    def get_tools_schema(self):
        """返回发送给 LLM 的 function-calling schema。

        schema 不执行工具；它只是告诉模型可用工具、每个工具的用途及参数格式。
        真实执行发生在 Agent 通过 ``registry`` 查找 Python 函数时。
        """

        return [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "根据城市获取天气",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "城市名称",
                            },
                        },
                        "required": ["city"],
                    },
                },
            }
        ]

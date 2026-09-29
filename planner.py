"""任务规划层：调用模型拆解用户目标，将 JSON 文本解析为 Plan 对象。"""

from typing import Any

from llm import LLM
from state import Plan


class Planner:
    """负责生成计划，但不执行工具、不直接回答用户"""

    def __init__(self, llm: LLM):
        # 复用 Agent 传入的 LLM 实例；客户端共用，但消息列表单独组装。
        self.llm = llm

    def create_plan(self, user_input: str, skill_text: str | None = None) -> Plan:
        """根据本轮输入生成计划，不在此处执行任何计划步骤。

        返回 Plan 实例。API 请求失败、输出不是合法 JSON 或字段校验不通过时，
        异常直接向上传播；当前没有增加重试或默认计划。
        """

        # planner 与正常 Agent 对话使用不同的提示词
        # 它只负责拆分步骤，不能直接回答问题，也不能调用工具
        messages = [
            {
                "role": "system",
                "content": """
                你是一个任务规划器。

请把用户目标拆成完成任务所需的少量步骤。

规则：
- 每步必须具体、可执行。
- 步骤按合理顺序排列。
- 简单任务可以只有一步。
- 不要调用工具。
- 不要回答用户问题。
- 只返回 JSON，不要 Markdown，不要解释。
- 不要返回 status 或 result 字段。

返回格式：
{
  "steps": [
    {
      "id": 1,
      "description": "步骤说明"
    }
  ]
}
                """,
            },
            {
                "role": "user",
                "content": user_input,
            },
        ]

        if skill_text:
            messages.insert(1, {
                "role": "system",
                "content": f"制定计划时请参考以下技能说明：\n\n{skill_text}",
            })

        # 不传 tools，因此此请求只用于生成计划；当前也未传入工具清单和历史消息。
        response = self.llm.chat(messages=messages)

        # 外部 JSON 文本 -> Plan 实例；model_dump 则是反方向的“实例 -> 字典”。
        # steps 的拼写必须匹配模型定义。结构校验不代表计划可执行或一定能完成目标。
        return Plan.model_validate_json(response.content)

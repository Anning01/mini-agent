"""单次 Agent 运行过程的结构化状态定义。"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """一条已经执行完成的工具调用记录。"""

    # 工具注册表中的名称，例如 "get_weather"。
    name: str

    # 模型请求该工具时提供的参数，例如 {"city": "北京"}。
    # Any 是因为不同工具的参数类型和嵌套结构可能不同。
    arguments: dict[str, Any]

    # 工具返回内容统一转成字符串，便于展示、记录和再次放入模型上下文。
    result: str


class PlanState(BaseModel):
    """一个计划步骤的描述与执行状态；不是整个 Agent 的状态。"""

    # 步骤标识，例如 1、2、3。当前仅校验类型，没有校验编号是否重复。
    id: int

    # 这一步要完成的子目标；由 Planner 生成，执行器将其放入步骤指令。
    description: str

    # 初始化/解析时限定为四种状态；默认 pending，之后由 Python 执行器更新。
    # 当前没有启用 validate_assignment，后续属性赋值不会自动重新校验。
    status: Literal["pending", "running", "completed", "failed"] = "pending"

    # 模型给出的步骤结果，与 ToolResult 中原始工具输出不同。尚未执行时为 None。
    result: str | None = None


class Plan(BaseModel):
    """按顺序排列的任务步骤，与 Planner 输出 JSON 的 steps 字段对应。"""

    # default_factory 每次创建独立列表；当前允许空计划，尚无最少步骤数校验。
    steps: list[PlanState] = Field(default_factory=list)


class AgentState(BaseModel):
    """描述一次 ``Agent.run`` 从开始到结束的全部可观察状态。

    这是运行状态，不是跨会话的长期记忆。每调用一次 ``run``，都会创建一个新实例；
    Agent 仅通过 ``agent.state`` 保留最近一次实例以便调试。
    """

    # 本次任务的原始用户输入，是整个运行的起点。
    user_input: str

    # 本次实际发送给 LLM 的上下文，沿用 Chat Completions 的 dict 消息格式。
    # default_factory=list 可确保每个 State 实例拥有独立列表，不会意外共享消息。
    messages: list[dict[str, Any]] = Field(default_factory=list)

    # 所有计划步骤的执行循环请求次数之和，不含规划、记忆提取或 SDK 内部重试。
    current_step: int = 0

    # 生命周期状态：创建时 running；正常回答后 completed；超出最大步骤后 failed。
    status: str = "running"

    # 调试用的工具轨迹；模型读取的是 messages 中的 tool 消息，不会自动读取此字段。
    tool_results: list[ToolResult] = Field(default_factory=list)

    # 当前取最后一个计划步骤的结果；失败时保存失败提示，尚未结束时为 None。
    final_answer: str | None = None

    # 本次任务的实时计划对象，包含每一步不断更新的状态和结果。
    plan: Plan | None = None

    # 正常开始前/全部执行完后为 None；步骤返回失败时保留其 id，便于定位。
    current_plan_step_id: int | None = None

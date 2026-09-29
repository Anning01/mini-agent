"""Agent 的主执行循环。

这里负责协调 LLM、工具、短期历史、长期记忆与单次运行状态，并驱动
“模型决定 -> 调用工具（可选）-> 模型继续决定”的 Agent Loop。
"""

import json
from pathlib import Path

from openai.types.responses import response

from llm import LLM
from memory import MemoryManager
from memory_store import MemoryStore
from planner import Planner
from skills import SkillManager
from state import AgentState, ToolResult, Plan
from tools import Tools


class Agent:
    """最小 Agent 的协调者。

    ``history`` 保存跨轮有效的短期历史；每次 ``run`` 则创建一个全新的
    ``AgentState``，记录当前任务实际发送的上下文和执行过程。
    """

    def __init__(self):
        # LLM 负责与模型 API 通信。
        self.llm = LLM()

        # Tools 保存真实 Python 工具函数及其给模型使用的 schema。
        self.tools = Tools()

        # MemoryManager 负责“从对话中提取什么值得长期记住”。
        self.memories = MemoryManager()

        # MemoryStore 负责把长期记忆保存到 SQLite，并在下次运行时读取。
        self.memory_store = MemoryStore()

        # 指向最近一次（或正在进行的）任务状态；尚未运行时为 None。
        self.state: AgentState | None = None

        # Planner 复用模型客户端，通过独立提示词拆分任务，不执行工具。
        self.planner = Planner(self.llm)

        # 获取skills
        self.skills = SkillManager(Path(__file__).resolve().parent / "skills")

        # 系统消息定义 Agent 的基础角色；每次请求模型时都会放在上下文最前面。
        self.system_message = {
            "role": "system",
            "content": "You are a helpful assistant",
        }

        # 历史按“完整对话回合”保存；长期记忆提示不放进这里，避免重复注入上下文。
        self.history = []

        # 每次 execute_agent_loop 最多请求模型 10 次，即每个计划步骤各有此上限。
        # 整个任务还包括其他步骤、规划和记忆提取，因此总请求数可能超过 10 次。
        self.max_steps = 10

    def get_memory_message(self):
        """把数据库中的长期记忆转换成一条临时系统消息。

        这条消息只属于本次请求，不会写进 ``self.history``；下次运行会重新读取
        数据库中的最新记忆，因此不会随着对话轮数重复累积。
        """

        memories = self.memory_store.get_all()

        # 没有长期记忆时，无需向模型额外发送一条 system message。
        if not memories:
            return None

        return {
            "role": "system",
            "content": f"""以下是关于用户的长期记忆：
{json.dumps(memories, ensure_ascii=False)}

请在回答时合理参考这些信息。""",
        }

    def get_plan_message(self, plan: Plan) -> dict[str, str]:
        """生成计划的文本快照，供执行模型参考。

        后续修改 Plan 对象不会自动更新这段 JSON 字符串。实际进度保存在 State，
        当前执行哪一步则由 execute_plan 追加的步骤指令告知模型。
        """

        # model_dump() 把 Pydantic Plan 对象转成普通字典；
        # json.dumps() 再把字典变成模型可阅读的 JSON 文本。
        plan_text = json.dumps(plan.model_dump(), ensure_ascii=False, indent=2)

        return {
            "role": "system",
            "content": f"""以下是本次任务的执行计划：
{plan_text}
请参考这个计划完成用户任务。
按步骤推进；如果某一步需要工具，请调用合适的工具。""",
        }

    def execute_agent_loop(
            self,
            messages: list[dict],
            current_turn: list[dict],
            tools_schema: list,
            state: AgentState,
    ) -> str | None:
        """完成一个计划步骤所需的模型与工具循环。

        messages 和 current_turn 都会原地追加消息：前者供模型继续执行，后者供
        run 保存本轮历史。state 记录所有步骤累计的执行循环调用次数与工具轨迹。

        Returns:
            本步骤的文本结果；循环耗尽或模型 content 为 None 时返回 None。
            模型返回文本仅表示结束当前循环，不保证步骤的业务目标已经达成。
        """

        for _ in range(1, self.max_steps + 1):

            # 在所有计划步骤之间累计；不包含 Planner 和记忆提取的独立请求。
            state.current_step += 1

            response = self.llm.chat(
                messages=messages,
                tools=tools_schema,
            )

            # SDK 返回的是对象；上下文统一存 dict，便于再次发送、保存和序列化。
            # tool_calls 也会保留在这条 assistant 消息中，供下一次模型请求关联结果。
            assistant_message = response.model_dump(exclude_none=True)
            messages.append(assistant_message)
            current_turn.append(assistant_message)

            # 无工具请求时结束本步骤；是否继续下一步由 execute_plan 控制。
            if not response.tool_calls:
                return response.content

            # 模型请求工具时，逐个执行所有 tool call，再把结果放回消息上下文。
            for tool_call in response.tool_calls:
                # arguments 是 JSON 字符串；解析后才能用 **arguments 调用 Python 函数。
                name = tool_call.function.name
                # 解析发生在 try 之前，非法 JSON 当前会直接抛出异常。
                arguments = json.loads(tool_call.function.arguments)

                try:
                    # registry 将模型返回的工具名映射到真实 Python 函数。
                    result = self.tools.registry[name](**arguments)
                except Exception as e:
                    # 工具错误同样返回给模型，使它有机会解释、重试或改用其他方案。
                    result = f"工具调用失败，错误信息：{str(e)}"

                # tool_call_id 用于将此结果关联回模型提出的那次工具调用。
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                }

                # 工具结果既进入当前请求上下文，也进入本轮历史。
                messages.append(tool_message)
                current_turn.append(tool_message)

                # 额外以结构化形式记录工具轨迹，方便通过 agent.state 调试。
                state.tool_results.append(
                    ToolResult(
                        name=name,
                        arguments=arguments,
                        result=str(result),
                    )
                )
        return None

    def execute_plan(
            self,
            state: AgentState,
            messages: list[dict],
            current_turn: list[dict],
            tools_schema: list,
    ) -> str | None:
        """按计划列表的顺序执行步骤，并更新步骤状态。

        Args:
            state: 本次任务状态，包含 Planner 生成的计划。
            messages: 各步骤共用的上下文，后续步骤可以读取前面的工具与模型结果。
            current_turn: 本轮对话记录，不包含内部 system 指令。
            tools_schema: 执行模型可使用的工具说明。

        Returns:
            最后一步结果。缺少计划、空计划或某一步返回 None 时也返回 None。
            未捕获的异常会向调用方传播，当前尚未统一转换为 failed 状态。
        """

        # Planner 已经为当前任务生成计划，正常情况下不会是 None。
        if state.plan is None:
            return None

        # 用来保存最后一步的结果。
        # 当前直接把最后一步结果交给用户，并没有单独的最终汇总请求。
        final_answer = None

        # 按列表顺序执行，不按 id 排序；id 只是用于定位步骤的标识。
        for step in state.plan.steps:

            # step 与 state.plan.steps 中的元素是同一对象，修改会反映到 State。
            state.current_plan_step_id = step.id

            # 从待执行进入执行中；模型本身不负责修改这些 Python 状态。
            step.status = "running"

            step_message = {
                "role": "system",
                "content": f"""
                    你是计划步骤执行器。

                    现在只执行第 {step.id} 步：

                    {step.description}

                    要求：
                    - 只完成当前步骤。
                    - 可以使用已有上下文和可用工具。
                    - 完成后直接返回本步骤的结果。
                    - 不要询问用户是否继续。
                    - 不要执行或宣布后续步骤。
                    - 不要判断整个计划是否完成。
                """,
            }
            # 将步骤指令加入实际发送给 LLM 的上下文。
            # 不加入 current_turn，因为它不是用户与助手之间的真实对话。
            messages.append(step_message)

            # 复用同一执行循环；本步骤追加的结果留在 messages 中供后续步骤读取。
            step_result = self.execute_agent_loop(
                messages, current_turn, tools_schema, state
            )

            # None 通常表示循环耗尽，也可能是模型没有返回 content；暂用相同提示。
            if step_result is None:
                step.status = "failed"
                step.result = "执行次数超过限制，步骤未完成。"

                # current_plan_step_id 保留失败步骤的 id，
                # 方便之后判断任务失败在哪里。
                return None
            # 当前将“拿到文本”视为完成，尚未识别文本中的工具失败或无法完成信息。
            step.status = "completed"
            step.result = step_result

            # 不断覆盖，循环结束后保留的就是最后一步结果
            final_answer = step.result

        # 所有计划步骤执行结束，当前没有正在运行的步骤。
        state.current_plan_step_id = None

        return final_answer

    def choose_skills(self, text: str) -> str | None:
        """根据用户输入，从已登记的技能中选择一个或者不选"""
        catalog = {
            name: info["description"]
            for name, info in self.skills.index.items()
        }
        if not catalog:
            return None
        response = self.llm.chat([
            {
                "role": "system",
                "content": (
                    "根据用户任务和技能目录，选择最相关的一个技能。"
                    "不需要技能时返回 null。"
                    '只返回 JSON，格式为 {"name": "技能名或 null"}。'
                    f"\n技能目录：{json.dumps(catalog, ensure_ascii=False)}"
                ),
            },
            {
                "role": "user",
                "content": text
            }
        ])

        choice = json.loads(response.content)["name"]

        # 模型只能提议名称，程序复制它确实存在
        if choice is not None and choice not in self.skills.index:
            raise ValueError(f"模型选择了未知技能{choice}")
        return choice

    def run(self, text: str, skill_name: str | None = None) -> str:
        """执行一次用户任务，并返回最终文本回答。

        Args:
            text: 用户本轮输入。
            skill_name: 技能名称。

        Returns:
            模型最终回答；若循环超限，则返回失败提示。
        """

        # State 只属于一次 run，不能复用上一轮的 messages、步骤数或工具结果。
        state = AgentState(user_input=text)

        # 把skills技能信息传入agent
        if skill_name is not None:
            skill_name = self.choose_skills(text)
        skill_text = self.skills.load(skill_name) if skill_name else None

        # Planner 当前只读取本轮输入；生成计划时还没有收到历史、记忆或工具 schema。
        state.plan = self.planner.create_plan(user_input=text, skill_text=skill_text)

        # 规划成功后才挂到实例上；若上面的规划抛异常，此引用仍是上一轮状态。
        # run 正常返回后，调用方可通过 agent.state 查看本轮执行轨迹。
        self.state = state

        # 本次 run 从空列表组装上下文，后续模型请求复用并追加该列表。
        # 赋值不会复制列表：messages 与 state.messages 指向同一对象。
        messages = state.messages
        messages.append(self.system_message)

        # 传入skills
        if skill_text:
            messages.append({
                "role": "system",
                "content": f"本次任务请参考以下技能说明：\n\n{skill_text}",
            })

        # 注入计划的初始快照；这条内部指令不写入 history。
        plan_message = self.get_plan_message(state.plan)
        messages.append(plan_message)

        # 长期记忆只在此处注入一次，不会被写入 self.history。
        memory_message = self.get_memory_message()
        if memory_message:
            messages.append(memory_message)

        # 仅注入最近五个完整回合；并没有删除 history 中更早的回合或限制 token 数。
        for history_turn in self.history[-5:]:
            messages.extend(history_turn)

        # 当前用户输入放在上下文最后，作为本轮任务的起点。
        user_message = {"role": "user", "content": text}
        messages.append(user_message)

        # 只记录本轮真实对话，不含系统提示和长期记忆提示；成功后会保存到 history。
        current_turn = [user_message]

        # schema 是给模型阅读的“工具说明书”，并不直接执行工具。
        tools_schema = self.tools.get_tools_schema()

        # 外层执行计划，内层循环调用模型和工具。保存历史和记忆只在整体结束后做一次。
        final_answer = self.execute_plan(state, messages, current_turn, tools_schema)

        if final_answer is None:
            # 缺失/空计划、执行超限或模型返回 None 都会进入这里，当前提示尚未细分。
            state.status = "failed"
            state.final_answer = "Agent 执行次数超过限制，任务未完成。"
            return state.final_answer

        # 以整轮为单位保存短期历史，后续不会从工具调用链中间截断消息。
        self.history.append(current_turn)

        # MVP 阶段只从用户输入提取记忆，避免把模型生成内容误认为用户事实。
        new_memories = self.memories.extract([user_message])

        # MemoryStore 以 key 为唯一键：同名记忆会更新而不是无限重复新增。
        self.memory_store.save(new_memories)

        # State 必须在返回前更新，才能准确描述这次任务已经成功完成。
        state.status = "completed"
        state.final_answer = final_answer
        return final_answer

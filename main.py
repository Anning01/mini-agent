"""命令行入口：持续读取用户输入并交给 Agent 执行。"""

from agent import Agent


if __name__ == "__main__":
    # 同一个 Agent 实例会在整个终端会话中复用，因此可以保留短期对话历史。
    agent = Agent()

    while True:
        # input() 的返回值本来就是 str；显式转换强调 Agent.run 的输入是文本。
        message = input("请输入你想要问的问题：")
        message = str(message)

        # 输入 exit 时退出循环，不再请求模型。
        if message == "exit":
            break

        # run() 内部完成上下文组装、工具循环、记忆提取和 State 更新。
        response = agent.run(message)

        # 打印给用户看的最终自然语言回答。
        print(response)

        # 打印最近一次运行的结构化状态，便于学习和调试执行过程。
        # model_dump() 返回字典，打印时字符串用 repr 表示，可显示孤立代理字符的转义。
        # 这只是方便定位异常文本，并不修复数据；输出还包含记忆和对话，分享前需检查。
        print(agent.state.model_dump())

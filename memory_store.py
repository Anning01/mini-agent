"""长期记忆的 SQLite 持久化层。"""

import sqlite3


class MemoryStore:
    """负责初始化、保存和读取键值对形式的长期记忆。"""

    def __init__(self, db_path="memory.db"):
        # 相对路径以进程启动目录为基准；文件不存在时 SQLite 自动创建它。
        self.conn = sqlite3.connect(db_path)

        # key 具有 UNIQUE 约束：完全相同的键只保留一行，不会自动合并语义相近的键。
        # 这是 MVP 的最小 schema，后续可扩展来源、时间和置信度等字段。
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                value TEXT NOT NULL
            )
        """)

        # 提交可能存在的事务；若连接当前没有活动事务，此调用不做任何事。
        self.conn.commit()

    def save(self, memories: dict):
        """保存一组长期记忆；相同 key 会更新而不是重复新增。"""

        for key, value in memories.items():
            # 参数化 SQL 将数据与 SQL 语句分开，避免字符串拼接造成注入或转义问题。
            # excluded.value 是这次原本准备插入的新值，发生键冲突时用它覆盖旧值。
            self.conn.execute(
                """
                INSERT INTO memories (key, value)
                VALUES (?, ?)
                ON CONFLICT(key)
                DO UPDATE SET value = excluded.value
            """,
                (key, value),
            )

        # 本次循环内的所有 INSERT/UPDATE 统一在一次 commit 后生效。
        self.conn.commit()

    def get_all(self):
        """读取所有长期记忆，并转换为 Agent 便于使用的字典。"""

        cursor = self.conn.execute("""
            SELECT key, value
            FROM memories
        """)

        # fetchall() 得到 [(key, value), ...]；dict() 将其转换为 {key: value}。
        return dict(cursor.fetchall())


if __name__ == "__main__":
    # 可独立运行本文件，查看当前数据库中的全部长期记忆。
    store = MemoryStore()

    # 取消注释后可手动写入示例记忆，用于观察 ON CONFLICT 的更新行为。
    # store.save({
    #     "姓名": "张三",
    #     "职业": "Python 开发者",
    #     "当前学习方向": "Agent",
    #     "长期目标": "从底层自行实现 Agent",
    #     "偏好": "不希望直接使用 LangChain"
    # })

    print(store.get_all())

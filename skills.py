from pathlib import Path


class SkillManager:
    """扫描本地skills目录，按名称读取skills说明"""

    def __init__(self, root: Path):
        # 索引保存技能简介和路径；不在这里把全部正文放进模型上下文。
        self.index: dict[str, dict] = {}

        for path in root.glob("*/SKILL.md"):
            text = path.read_text(encoding="utf-8")

            # 第一版只处理你当前使用的简单 frontmatter：
            # ---
            # name: weather-compare
            # description: ...
            # ---
            if not text.startswith("---\n"):
                raise ValueError(f"技能缺少 frontmatter：{path}")

            header, separator, _body = text[4:].partition("\n---\n")
            if not separator:
                raise ValueError(f"技能 frontmatter 未结束：{path}")

            metadata = {}
            for line in header.splitlines():
                key, separator, value = line.partition(":")
                if separator and key in {"name", "description"}:
                    metadata[key] = value.strip()

            name = metadata.get("name")
            description = metadata.get("description")
            if not name or not description:
                raise ValueError(f"技能缺少 name 或 description：{path}")
            if name in self.index:
                raise ValueError(f"发现重复技能名：{name}")

            self.index[name] = {
                "description": description,
                "path": path,
            }

    def load(self, name: str) -> str:
        """只读取已登记的技能；未知名称不能变成任意文件路径。"""
        if name not in self.index:
            raise ValueError(f"未知技能：{name}")

        return self.index[name]["path"].read_text(encoding="utf-8")

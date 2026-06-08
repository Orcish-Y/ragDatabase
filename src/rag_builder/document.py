"""Markdown 文档解析 — 按标题层级切分为 Section 对象。"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Section:
    """文档中的一个节（#### 标题块）。"""

    section_id: str
    section_title: str
    content: str
    chapter: Optional[str] = None
    category: str = ""
    source_file: str = ""
    content_hash: str = ""
    start_line: int = 0

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.sha256(
                self.content.encode("utf-8")
            ).hexdigest()[:16]


# 匹配 #### 标题行，提取编号和标题
HEADING_H4_PATTERN = re.compile(r"^####\s+(\S+)\s*(.*)$")
# 匹配 ### 标题行
HEADING_H3_PATTERN = re.compile(r"^###\s+(.*)$")


def load_chapter_map(path: str | Path) -> dict[str, str]:
    """加载章映射 JSON 文件。

    格式: {"编号前缀": "章名"}
    示例: {"01": "制剂通则", "04": "光谱法"}
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_chapter(
    section_id: str,
    chapter_map: dict[str, str] | None,
    h3_chapter: str | None,
) -> str | None:
    """按优先级解析章名: JSON 映射 > ### 推导 > None。"""
    if chapter_map:
        # 前缀最长匹配
        best_prefix = ""
        best_name = None
        for prefix, name in chapter_map.items():
            if section_id.startswith(prefix) and len(prefix) > len(best_prefix):
                best_prefix = prefix
                best_name = name
        if best_name:
            return best_name
    if h3_chapter:
        return h3_chapter
    return None


def parse_markdown(
    file_path: str | Path,
    chapter_map: dict[str, str] | None = None,
) -> list[Section]:
    """解析 Markdown 文件，提取 #### 节。

    Args:
        file_path: Markdown 文件路径。
        chapter_map: 可选的编号前缀 → 章名映射。

    Returns:
        Section 对象列表，按文档出现顺序排列。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 文件不是 .md 后缀或未找到任何 #### 节。
    """
    file_path = Path(file_path)

    if file_path.suffix.lower() != ".md":
        raise ValueError(f"仅支持 .md 文件，收到: {file_path.suffix}")

    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    source_name = file_path.stem
    lines = file_path.read_text(encoding="utf-8").splitlines()

    sections: list[Section] = []
    current_h3: str | None = None
    current_id: str | None = None
    current_title: str | None = None
    current_start: int = 0
    current_section_h3: str | None = None  # 节开始时的 h3 快照
    content_lines: list[str] = []

    def flush_section(end_line: int) -> None:
        nonlocal current_id, current_title, current_start, current_section_h3, content_lines
        if current_id is not None:
            content = "\n".join(content_lines).strip()
            chapter = resolve_chapter(current_id, chapter_map, current_section_h3)
            sections.append(
                Section(
                    section_id=current_id,
                    section_title=current_title or "",
                    content=content,
                    chapter=chapter,
                    source_file=source_name,
                    start_line=current_start,
                )
            )
        current_id = None
        current_title = None
        content_lines = []

    for i, line in enumerate(lines):
        # 检查 ### 章标题
        h3_match = HEADING_H3_PATTERN.match(line)
        if h3_match:
            current_h3 = h3_match.group(1).strip()
            continue

        # 检查 #### 节标题
        h4_match = HEADING_H4_PATTERN.match(line)
        if h4_match:
            flush_section(i)
            current_id = h4_match.group(1)
            current_title = h4_match.group(2).strip()
            current_start = i + 1  # 1-based
            current_section_h3 = current_h3  # 快照当前 h3
            content_lines = []
            continue

        # 如果还在当前节内，收集内容
        if current_id is not None:
            content_lines.append(line)

    # 最后一个节
    flush_section(len(lines))

    if not sections:
        raise ValueError(f"文档中未找到任何 #### 节: {file_path}")

    return sections

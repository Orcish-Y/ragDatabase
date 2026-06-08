"""默认参数常量 — 从 rag_core 继承，扩展建库专用。"""
from rag_core.config import *  # noqa: F401, F403

# 句子边界正则（建库专用）
SENTENCE_BOUNDARY_PATTERN_ZH = r"(?<=[。！？；])\s*"
SENTENCE_BOUNDARY_PATTERN_EN = r"(?<=[.!?;])\s+"

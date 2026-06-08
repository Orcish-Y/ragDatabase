"""嵌入模型 — 从 rag_core 导入，向后兼容。"""
from rag_core.embedder import (
    DashScopeEmbedder,
    Embedder,
    _build_jsonl,
    _require_api_key,
    _truncate_text,
    MAX_CHARS_PER_TEXT,
)

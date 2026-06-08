"""向量库 — 从 rag_core 导入 ReadVectorStore，向后兼容。"""
from rag_core.vector_store import (
    VectorStore,
    _EmbedderAdapter,
)

# 别名，保持向后兼容
ReadVectorStore = VectorStore

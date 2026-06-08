"""rag_server — RAG 知识库轻量检索服务。

仅含检索所需：Embedding + Chroma 读取 + MCP 服务。
不含建库逻辑（文档解析、语义切分、Batch API）。
"""

__version__ = "0.1.0"

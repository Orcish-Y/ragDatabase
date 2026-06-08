"""Chroma 向量库操作 — 仅检索（只读）。"""
from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings

from .config import DEFAULT_COLLECTION_NAME, PARENT_COLLECTION_SUFFIX
from .embedder import DashScopeEmbedder


class VectorStore:
    """Chroma 向量库管理器（只读检索）。

    管理两个 Collection：
    - chunk 索引 (rag_documents)：语义 chunk 向量，用于相似度检索
    - 父文档存储 (rag_documents_parents)：整节全文，用于按 section_id 反查
    """

    def __init__(
        self,
        persist_dir: str | Path,
        embedder: DashScopeEmbedder,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ):
        self.persist_dir = str(persist_dir)
        self.embedder = embedder
        self.collection_name = collection_name
        self.parent_collection_name = collection_name + PARENT_COLLECTION_SUFFIX
        self._vectorstore: Chroma | None = None
        self._parent_store: Chroma | None = None
        self._adapter: _EmbedderAdapter | None = None

    @property
    def _get_adapter(self) -> _EmbedderAdapter:
        if self._adapter is None:
            self._adapter = _EmbedderAdapter(self.embedder)
        return self._adapter

    @property
    def vectorstore(self) -> Chroma:
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self._get_adapter,
                persist_directory=self.persist_dir,
            )
        return self._vectorstore

    @property
    def parent_store(self) -> Chroma:
        """父文档 Collection —— 存储整节全文，按 section_id 查询。"""
        if self._parent_store is None:
            self._parent_store = Chroma(
                collection_name=self.parent_collection_name,
                embedding_function=self._get_adapter,
                persist_directory=self.persist_dir,
            )
        return self._parent_store

    def close(self) -> None:
        """关闭 Chroma 客户端，释放文件句柄。"""
        for attr in ("_vectorstore", "_parent_store"):
            obj = getattr(self, attr, None)
            if obj is not None:
                del obj
        self._vectorstore = None
        self._parent_store = None

    def get_parent_by_id(self, section_id: str) -> LCDocument | None:
        """按 section_id 查询父文档全文。"""
        try:
            results = self.parent_store.get(
                where={"section_id": section_id},
                include=["metadatas", "documents"],
            )
            if results["ids"] and results["documents"]:
                return LCDocument(
                    page_content=results["documents"][0],
                    metadata=results["metadatas"][0],
                )
        except Exception:
            pass
        return None

    def has_parent_collection(self) -> bool:
        """检查父文档 Collection 是否存在且非空。"""
        try:
            results = self.parent_store.get(include=["metadatas"])
            return bool(results["ids"])
        except Exception:
            return False


class _EmbedderAdapter(Embeddings):
    """将 Embedder 适配到 LangChain Embeddings 接口。"""

    def __init__(self, embedder: DashScopeEmbedder):
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._embedder.embed(texts)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vectors = self._embedder.embed([text])
        return vectors[0].tolist()

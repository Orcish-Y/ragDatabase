"""Chroma 向量库操作 — 语义 chunk 索引 + 元数据存储。"""

from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings

from .config import DEFAULT_COLLECTION_NAME
from .document import Section
from .embedder import Embedder
from .chunker import chunk_section


def make_doc_id(section: Section) -> str:
    """生成节唯一 ID：{源文件名}#{节编号}。"""
    return f"{section.source_file}#{section.section_id}"


class VectorStore:
    """Chroma 向量库管理器。

    将语义 chunk 写入 Chroma，附带完整元数据。
    检索时消费者可通过元数据中的 section_id 反查整节全文。
    """

    def __init__(
        self,
        persist_dir: str | Path,
        embedder: Embedder,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ):
        self.persist_dir = str(persist_dir)
        self.embedder = embedder
        self.collection_name = collection_name
        self._vectorstore: Chroma | None = None

    @property
    def vectorstore(self) -> Chroma:
        if self._vectorstore is None:
            adapter = _EmbedderAdapter(self.embedder)
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=adapter,
                persist_directory=self.persist_dir,
            )
        return self._vectorstore

    def close(self) -> None:
        """关闭 Chroma 客户端，释放文件句柄。"""
        if self._vectorstore is not None:
            # Chroma 没有显式 close，但可以删除引用让 GC 回收
            del self._vectorstore
            self._vectorstore = None

    def _get_existing_hashes(self) -> dict[str, str]:
        """获取数据库中已有的 (section_id → content_hash) 映射。"""
        try:
            results = self.vectorstore.get(include=["metadatas"])
            existing: dict[str, str] = {}
            if results["metadatas"]:
                seen: set[str] = set()
                for meta in results["metadatas"]:
                    sid = meta.get("section_id", "")
                    if sid and sid not in seen:
                        seen.add(sid)
                        existing[sid] = meta.get("content_hash", "")
            return existing
        except Exception:
            return {}

    def add_sections(
        self,
        sections: list[Section],
        threshold: float | None = None,
        overlap: int = 2,
        min_chunk: int = 3,
        max_chunk: int = 20,
    ) -> dict[str, int]:
        """将 Section 列表写入 Chroma，支持增量更新。

        Returns:
            {"inserted": N, "skipped": N, "updated": N}
        """
        existing_hashes = self._get_existing_hashes()

        inserted = 0
        skipped = 0
        updated = 0

        for section in sections:
            existing_hash = existing_hashes.get(section.section_id)

            # 内容未变 → 跳过
            if existing_hash and existing_hash == section.content_hash:
                skipped += 1
                continue

            # 语义切分
            chunks = chunk_section(
                section,
                self.embedder,
                threshold=threshold,
                overlap=overlap,
                min_chunk=min_chunk,
                max_chunk=max_chunk,
            )

            if not chunks:
                skipped += 1
                continue

            # 如果是更新，先删旧
            if existing_hash and existing_hash != section.content_hash:
                self._delete_by_section_id(section.section_id)
                updated += 1
            else:
                inserted += 1

            # 写入子文档
            docs = [
                LCDocument(
                    page_content=c.text,
                    metadata={
                        "section_id": section.section_id,
                        "section_title": section.section_title,
                        "chapter": section.chapter or "",
                        "source_file": section.source_file,
                        "content_hash": section.content_hash,
                        "chunk_index": c.chunk_index,
                    },
                )
                for c in chunks
            ]
            self.vectorstore.add_documents(docs)

        return {"inserted": inserted, "skipped": skipped, "updated": updated}

    def _delete_by_section_id(self, section_id: str) -> None:
        """删除指定 section_id 的所有文档。"""
        try:
            results = self.vectorstore.get(
                where={"section_id": section_id}, include=["metadatas"]
            )
            if results["ids"]:
                self.vectorstore.delete(ids=results["ids"])
        except Exception:
            pass


class _EmbedderAdapter(Embeddings):
    """将我们的 Embedder 适配到 LangChain Embeddings 接口。"""

    def __init__(self, embedder: Embedder):
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._embedder.embed(texts)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vectors = self._embedder.embed([text])
        return vectors[0].tolist()

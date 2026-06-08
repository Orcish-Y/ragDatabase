"""Chroma 向量库操作 — 语义 chunk 索引 + 父文档存储。"""

from __future__ import annotations

import uuid
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings

from .config import DEFAULT_COLLECTION_NAME
from .document import Section
from .embedder import DashScopeEmbedder, Embedder
from .chunker import chunk_section

PARENT_COLLECTION_SUFFIX = "_parents"


class _ChunkData:
    """内部辅助：暂存 chunk 切分结果，供批量 embedding 使用。"""
    __slots__ = ("chunk_id", "text", "section", "chunk_index")

    def __init__(self, chunk_id: str, text: str, section: Section, chunk_index: int):
        self.chunk_id = chunk_id
        self.text = text
        self.section = section
        self.chunk_index = chunk_index


def make_doc_id(section: Section) -> str:
    """生成节唯一 ID：{源文件名}#{节编号}。"""
    return f"{section.source_file}#{section.section_id}"


def _make_parent_metadata(section: Section) -> dict:
    return {
        "section_id": section.section_id,
        "section_title": section.section_title,
        "chapter": section.chapter or "",
        "category": section.category or "",
        "source_file": section.source_file,
        "content_hash": section.content_hash,
    }


def _make_chunk_metadata(section: Section, chunk_index: int) -> dict:
    meta = _make_parent_metadata(section)
    meta["chunk_index"] = chunk_index
    return meta


class VectorStore:
    """Chroma 向量库管理器。

    管理两个 Collection：
    - chunk 索引 (rag_documents)：语义 chunk 向量，用于相似度检索
    - 父文档存储 (rag_documents_parents)：整节全文，用于按 section_id 反查
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

    def _get_existing_hashes(self) -> dict[str, str]:
        """从父文档 Collection 获取已有的 (section_id → content_hash) 映射。"""
        try:
            results = self.parent_store.get(include=["metadatas"])
            existing: dict[str, str] = {}
            if results["metadatas"]:
                for meta in results["metadatas"]:
                    sid = meta.get("section_id", "")
                    if sid:
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
        """将 Section 列表写入 Chroma，同时写 chunk 和父文档两个 Collection。

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

            # 写入子文档（向量索引）
            child_docs = [
                LCDocument(
                    page_content=c.text,
                    metadata=_make_chunk_metadata(section, c.chunk_index),
                )
                for c in chunks
            ]
            self.vectorstore.add_documents(child_docs)

            # 写入父文档（整节全文）
            parent_doc = LCDocument(
                page_content=section.content,
                metadata=_make_parent_metadata(section),
            )
            self.parent_store.add_documents([parent_doc])

        return {"inserted": inserted, "skipped": skipped, "updated": updated}

    def add_sections_batch(
        self,
        sections: list[Section],
        threshold: float | None = None,
        overlap: int = 2,
        min_chunk: int = 3,
        max_chunk: int = 20,
    ) -> dict[str, int]:
        """Batch 建库：chunk 全部 Section → 一次性批量 embedding → 写入 Chroma。

        语义切分阶段仍使用实时 embed() 计算 sentence similarity，
        但 chunk 文本的向量化通过 embed_batch() 走 OpenAI Batch API。

        Returns:
            {"inserted": N, "skipped": N, "updated": N}
        """
        existing_hashes = self._get_existing_hashes()

        # 1. 收集 chunk 数据
        all_chunk_data, skipped, metadata_updates = self._collect_chunks(
            sections, existing_hashes, threshold, overlap, min_chunk, max_chunk
        )

        if metadata_updates > 0:
            print(f"  仅更新 metadata: {metadata_updates} 节 (无需 embedding)", flush=True)

        if not all_chunk_data:
            return {"inserted": 0, "skipped": skipped, "updated": 0}

        # 2. 批量 embedding
        print(f"\n  采集到 {len(all_chunk_data)} 个 chunk 文本，开始批量 embedding...")
        chunk_texts = [d.text for d in all_chunk_data]
        embeddings = self.embedder.embed_batch(chunk_texts)

        if embeddings.shape[0] != len(chunk_texts):
            raise RuntimeError(
                f"embed_batch 返回数量不匹配: "
                f"期望 {len(chunk_texts)}, 实际 {embeddings.shape[0]}"
            )

        # 3. 写入 Chroma
        self._write_chunks_to_chroma(all_chunk_data, embeddings)

        # 4. 统计
        processed_sids = set(d.section.section_id for d in all_chunk_data)
        inserted = 0
        updated = 0
        for sid in processed_sids:
            if sid in existing_hashes:
                updated += 1
            else:
                inserted += 1

        return {"inserted": inserted, "skipped": skipped, "updated": updated}

    def _collect_chunks(
        self,
        sections: list[Section],
        existing_hashes: dict[str, str],
        threshold: float | None,
        overlap: int,
        min_chunk: int,
        max_chunk: int,
    ) -> tuple[list[_ChunkData], int, int]:
        """对所有 Section 语义切分，返回 (chunk_data, skipped, metadata_updates)。"""
        all_chunk_data: list[_ChunkData] = []
        skipped = 0
        metadata_updates = 0

        for i, section in enumerate(sections):
            if i % 20 == 0 and i > 0:
                print(f"  切分进度: {i}/{len(sections)} 节...", flush=True)
            existing_hash = existing_hashes.get(section.section_id)

            if existing_hash and existing_hash == section.content_hash:
                if self._update_section_metadata(section):
                    metadata_updates += 1
                skipped += 1
                continue

            chunks = chunk_section(
                section, self.embedder,
                threshold=threshold, overlap=overlap,
                min_chunk=min_chunk, max_chunk=max_chunk,
            )

            if not chunks:
                skipped += 1
                continue

            if existing_hash and existing_hash != section.content_hash:
                self._delete_by_section_id(section.section_id)

            for c in chunks:
                all_chunk_data.append(
                    _ChunkData(
                        chunk_id=str(uuid.uuid4()),
                        text=c.text,
                        section=section,
                        chunk_index=c.chunk_index,
                    )
                )

        return all_chunk_data, skipped, metadata_updates

    def _write_chunks_to_chroma(
        self,
        all_chunk_data: list[_ChunkData],
        embeddings: "np.ndarray",
    ) -> None:
        """写入 chunk 和 parent 两个 Collection（分批以适配 Chroma 限制）。"""
        CHROMA_BATCH = 5000
        total_chunks = len(all_chunk_data)
        dim = embeddings.shape[1]

        # — chunk collection
        for batch_start in range(0, total_chunks, CHROMA_BATCH):
            batch_end = min(batch_start + CHROMA_BATCH, total_chunks)
            self.vectorstore._collection.add(
                ids=[d.chunk_id for d in all_chunk_data[batch_start:batch_end]],
                embeddings=embeddings[batch_start:batch_end].tolist(),
                documents=[d.text for d in all_chunk_data[batch_start:batch_end]],
                metadatas=[
                    _make_chunk_metadata(d.section, d.chunk_index)
                    for d in all_chunk_data[batch_start:batch_end]
                ],
            )
            if total_chunks > CHROMA_BATCH:
                print(f"  写入 chunk: {batch_end}/{total_chunks}", flush=True)

        # — parent collection（去重）
        written_parents: set[str] = set()
        parent_ids: list[str] = []
        parent_docs: list[str] = []
        parent_metas: list[dict] = []
        parent_embeddings: list[list[float]] = []

        for d in all_chunk_data:
            sid = d.section.section_id
            if sid in written_parents:
                continue
            written_parents.add(sid)
            parent_ids.append(sid)
            parent_docs.append(d.section.content)
            parent_metas.append(_make_parent_metadata(d.section))
            parent_embeddings.append([0.0] * dim)

        self.parent_store._collection.add(
            ids=parent_ids,
            embeddings=parent_embeddings,
            documents=parent_docs,
            metadatas=parent_metas,
        )

    def _delete_by_section_id(self, section_id: str) -> None:
        """删除指定 section_id 的所有文档（子 + 父）。"""
        for store in (self.vectorstore, self.parent_store):
            try:
                results = store.get(
                    where={"section_id": section_id}, include=["metadatas"]
                )
                if results["ids"]:
                    store.delete(ids=results["ids"])
            except Exception:
                pass

    def _update_section_metadata(self, section: Section) -> bool:
        """内容未变时仅更新 metadata（不重新 embedding）。"""
        new_meta = _make_parent_metadata(section)
        changed = False

        # 更新父文档 collection
        try:
            results = self.parent_store.get(
                where={"section_id": section.section_id}, include=["metadatas"]
            )
            if results["ids"]:
                existing_meta = results["metadatas"][0] if results["metadatas"] else {}
                # 检查 category 等字段是否变化
                for key in new_meta:
                    if new_meta.get(key) != existing_meta.get(key):
                        changed = True
                        break
                if changed:
                    self.parent_store._collection.update(
                        ids=results["ids"],
                        metadatas=[new_meta] * len(results["ids"]),
                    )
        except Exception:
            pass

        # 更新 chunk collection 中的 metadata
        new_chunk_base = {
            k: new_meta[k]
            for k in ("section_id", "section_title", "chapter", "category",
                       "source_file", "content_hash")
        }
        try:
            results = self.vectorstore.get(
                where={"section_id": section.section_id}, include=["metadatas"]
            )
            if results["ids"]:
                updated_metas = []
                for meta in (results["metadatas"] or []):
                    m = {**new_chunk_base, "chunk_index": meta.get("chunk_index", 0)}
                    updated_metas.append(m)
                self.vectorstore._collection.update(
                    ids=results["ids"],
                    metadatas=updated_metas,
                )
        except Exception:
            pass

        return changed

    def get_parent_by_id(self, section_id: str) -> LCDocument | None:
        """按 section_id 查询父文档全文。

        Returns:
            匹配的 LangChain Document，未找到返回 None。
        """
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
    """将我们的 Embedder 适配到 LangChain Embeddings 接口。"""

    def __init__(self, embedder: Embedder):
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._embedder.embed(texts)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vectors = self._embedder.embed([text])
        return vectors[0].tolist()

"""测试 Chroma 向量库操作。"""

import gc
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rag_builder.document import Section
from rag_builder.vector_store import VectorStore, make_doc_id


class MockEmbedder:
    """轻量 mock，返回固定维度随机向量。"""

    def __init__(self, model_name: str = "mock"):
        self.model_name = model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(42)
        vecs = rng.random((len(texts), 384)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    def similarity(self, a, b):
        return float(np.dot(a, b))


def make_section(sid="0101", title="测试", content="测试内容。第二句。") -> Section:
    return Section(
        section_id=sid,
        section_title=title,
        content=content,
        source_file="test_doc",
    )


@pytest.fixture
def store_and_tmpdir():
    """创建临时目录和 VectorStore，测试结束后清理。"""
    tmpdir = tempfile.mkdtemp()
    embedder = MockEmbedder()
    store = VectorStore(persist_dir=tmpdir, embedder=embedder)
    yield store, tmpdir
    # 清理
    store.close()
    gc.collect()
    # 尝试删除临时目录（忽略 Windows 文件锁错误）
    import shutil
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass


# ── make_doc_id ────────────────────────────────────────────────


class TestMakeDocId:
    def test_format(self):
        s = make_section()
        assert make_doc_id(s) == "test_doc#0101"


# ── VectorStore ────────────────────────────────────────────────


class TestVectorStore:
    def test_build_and_retrieve(self, store_and_tmpdir):
        """端到端：建库 → 检索。"""
        store, _ = store_and_tmpdir
        sections = [
            make_section("0101", "片剂", "片剂系指原料药物制成的固体制剂。应符合崩解时限要求。"),
            make_section("0102", "注射剂", "注射剂系指原料药物制成的供注入体内的制剂。应无菌。"),
        ]
        stats = store.add_sections(sections)
        assert stats["inserted"] == 2
        assert stats["skipped"] == 0
        assert stats["updated"] == 0

        # 验证可检索
        results = store.vectorstore.similarity_search("崩解时限", k=2)
        assert len(results) > 0

    def test_incremental_skip(self, store_and_tmpdir):
        """重复入库同一内容 → 跳过。"""
        store, _ = store_and_tmpdir
        sections = [make_section()]

        stats1 = store.add_sections(sections)
        assert stats1["inserted"] == 1

        stats2 = store.add_sections(sections)
        assert stats2["skipped"] == 1
        assert stats2["inserted"] == 0

    def test_incremental_update(self, store_and_tmpdir):
        """修改内容后重入 → 更新。"""
        store, _ = store_and_tmpdir
        s1 = make_section("0101", "片剂", "旧内容。")
        store.add_sections([s1])

        s2 = make_section("0101", "片剂", "新内容。完全不同。第二句。")
        stats = store.add_sections([s2])
        assert stats["updated"] == 1
        assert stats["inserted"] == 0

    def test_incremental_new_section(self, store_and_tmpdir):
        """新增节 → 插入。"""
        store, _ = store_and_tmpdir
        store.add_sections([make_section("0101", "片剂", "内容。")])

        stats = store.add_sections([
            make_section("0101", "片剂", "内容。"),  # skip
            make_section("0102", "注射剂", "内容。"),  # insert
        ])
        assert stats["skipped"] == 1
        assert stats["inserted"] == 1


# ── Metadata ───────────────────────────────────────────────────


class TestMetadata:
    def test_all_fields_present(self, store_and_tmpdir):
        store, _ = store_and_tmpdir
        s = Section(
            section_id="0401",
            section_title="紫外-可见分光光度法",
            content="测试内容。",
            chapter="光谱法",
            source_file="药典四部",
        )
        store.add_sections([s])

        results = store.vectorstore.get(include=["metadatas"])
        assert len(results["ids"]) > 0
        meta = results["metadatas"][0]
        assert meta["section_id"] == "0401"
        assert meta["section_title"] == "紫外-可见分光光度法"
        assert meta["chapter"] == "光谱法"
        assert meta["source_file"] == "药典四部"
        assert "content_hash" in meta
        assert meta["chunk_index"] is not None

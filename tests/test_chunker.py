"""测试语义切分模块。"""

import numpy as np
import pytest

from rag_builder.chunker import (
    Chunk,
    chunk_section,
    compute_breakpoints,
    split_sentences,
)
from rag_builder.document import Section


from .conftest import MockEmbedder


class FixedEmbedder(MockEmbedder):
    """返回链式相邻相似度的向量。

    target_sims[i] 控制 sentences[i] 和 sentences[i+1] 的余弦相似度。
    通过 2D 旋转累积构造，确保相邻对精确匹配目标相似度。
    """

    def __init__(self, target_similarities: list[float]):
        super().__init__()
        self.target_sims = target_similarities

    def embed(self, texts: list[str]) -> np.ndarray:
        n = len(texts)
        vecs = np.zeros((n, 384), dtype=np.float32)
        # 从 (1, 0) 开始
        vecs[0, 0] = 1.0
        angle = 0.0  # 累积角度
        for i in range(n - 1):
            sim = self.target_sims[i] if i < len(self.target_sims) else 0.8
            sim = max(-1.0, min(1.0, sim))
            delta = np.arccos(sim)
            angle += delta
            vecs[i + 1, 0] = np.cos(angle)
            vecs[i + 1, 1] = np.sin(angle)
        return vecs


def make_section(content: str) -> Section:
    return Section(
        section_id="0001",
        section_title="测试节",
        content=content,
        source_file="test",
    )


# ── split_sentences ────────────────────────────────────────────


class TestSplitSentences:
    def test_chinese_period(self):
        sents = split_sentences("第一句话。第二句话。第三句话。")
        assert len(sents) == 3
        assert "第一句话" in sents[0]

    def test_mixed_punctuation(self):
        sents = split_sentences("你好！这是测试；真的吗？是的。")
        assert len(sents) >= 2

    def test_no_punctuation(self):
        sents = split_sentences("没有标点的一句话")
        assert len(sents) == 1
        assert sents[0] == "没有标点的一句话"

    def test_empty_string(self):
        sents = split_sentences("")
        assert sents == []

    def test_whitespace_only(self):
        sents = split_sentences("   \n  ")
        assert sents == []

    def test_english(self):
        sents = split_sentences("Hello world. How are you? I am fine.")
        assert len(sents) == 3


# ── compute_breakpoints ────────────────────────────────────────


class TestComputeBreakpoints:
    def test_no_break_for_short_text(self, tmp_path):
        sents = ["句1", "句2"]
        embedder = MockEmbedder()
        bps = compute_breakpoints(sents, embedder, min_chunk=3)
        assert bps == []

    def test_min_chunk_constraint(self):
        sents = [f"句子{i}" for i in range(10)]
        embedder = MockEmbedder()
        bps = compute_breakpoints(sents, embedder, min_chunk=4, threshold=0.0)
        # 设置 threshold=0 会让非常多的转折点，但 min_chunk=4 应过滤
        for bp in bps:
            assert bp >= 4
            assert 10 - bp >= 4 or bp == bps[-1]

    def test_max_chunk_constraint(self):
        sents = [f"句子{i}" for i in range(50)]
        embedder = FixedEmbedder([0.9] * 49)  # 全部高相似度 → 无自然转折
        bps = compute_breakpoints(sents, embedder, max_chunk=10, threshold=0.5)
        # 无自然转折但有 max chunk 约束 → 每 10 句强制切
        assert len(bps) > 0
        for bp in bps:
            assert bp % 10 == 0 or bp == 50

    def test_explicit_threshold(self):
        # 构造明显低相似度的句子对
        sents = ["苹果很好吃。", "香蕉也不错。", "火箭发射到太空。", "爱因斯坦相对论。"]
        embedder = FixedEmbedder([0.95, 0.02, 0.90])
        bps = compute_breakpoints(sents, embedder, threshold=0.3, min_chunk=1)
        # 第2对(索引1)相似度 0.02 < 0.3，应在索引2处切分
        assert 2 in bps


# ── chunk_section ──────────────────────────────────────────────


class TestChunkSection:
    def test_single_chunk_for_short_section(self):
        section = make_section("短内容。就两句。")
        embedder = MockEmbedder()
        chunks = chunk_section(section, embedder, min_chunk=3)
        assert len(chunks) == 1
        assert chunks[0].section_id == "0001"
        assert chunks[0].chunk_index == 0

    def test_overlap_between_chunks(self):
        section = make_section(
            "句1。句2。句3。句4。句5。句6。句7。句8。"
        )
        embedder = FixedEmbedder([0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1])
        chunks = chunk_section(
            section, embedder, threshold=0.5, min_chunk=1, overlap=2, max_chunk=10
        )
        if len(chunks) > 1:
            # chunk 0 的尾部应该与 chunk 1 的头部有重叠
            c0_sents = set(split_sentences(chunks[0].text))
            c1_sents = set(split_sentences(chunks[1].text))
            assert len(c0_sents & c1_sents) > 0

    def test_empty_content(self):
        section = make_section("")
        embedder = MockEmbedder()
        chunks = chunk_section(section, embedder)
        assert chunks == []

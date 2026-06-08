"""语义切分引擎 — 基于相邻句子 Embedding 相似度找转折点。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .config import (
    DEFAULT_MIN_CHUNK_SENTENCES,
    DEFAULT_MAX_CHUNK_SENTENCES,
    DEFAULT_OVERLAP_SENTENCES,
    DEFAULT_THRESHOLD_STD_DEV,
)
from .document import Section
from .embedder import Embedder

# 中文句子边界
_RE_SENTENCE_ZH = re.compile(r"(?<=[。！？；])\s*")
# 英文句子边界
_RE_SENTENCE_EN = re.compile(r"(?<=[.!?;])\s+")


@dataclass
class Chunk:
    """语义切分产生的一个文本块。"""

    text: str
    section_id: str
    chunk_index: int = 0
    sentence_range: tuple[int, int] = (0, 0)


def split_sentences(text: str) -> list[str]:
    """将文本按句末标点拆分为句子列表。

    支持中文（。！？；）和英文（.!?;）。
    空句子被过滤掉。
    如果没有任何标点，整段作为一个句子返回。
    """
    # 先尝试中文标点拆分
    parts = _RE_SENTENCE_ZH.split(text)
    result: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # 对英文标点进一步拆分
        sub_parts = _RE_SENTENCE_EN.split(part)
        for sp in sub_parts:
            sp = sp.strip()
            if sp:
                result.append(sp)

    if not result and text.strip():
        result = [text.strip()]

    return result


def compute_breakpoints(
    sentences: list[str],
    embedder: Embedder,
    threshold: float | None = None,
    min_chunk: int = DEFAULT_MIN_CHUNK_SENTENCES,
    max_chunk: int = DEFAULT_MAX_CHUNK_SENTENCES,
) -> list[int]:
    """计算语义转折点索引。

    对相邻句子对计算余弦相似度，低于阈值则标记为转折点。
    转折点索引 i 表示在第 i 句和第 i+1 句之间切分。

    Args:
        sentences: 句子列表。
        embedder: 嵌入模型。
        threshold: 绝对相似度阈值。None 时使用自适应阈值 (μ - 1.5σ)。
        min_chunk: 最小 chunk 句子数，转折点过于密集时合并。
        max_chunk: 最大 chunk 句子数，强制切分以避免过长 chunk。

    Returns:
        切分点索引列表（在哪个句子之前切分，不含 0）。
    """
    n = len(sentences)
    if n <= min_chunk:
        return []

    # 计算相邻句子相似度
    embeddings = embedder.embed(sentences)
    similarities = np.array(
        [embedder.similarity(embeddings[i], embeddings[i + 1]) for i in range(n - 1)]
    )

    # 确定阈值
    if threshold is None:
        mu = float(np.mean(similarities))
        sigma = float(np.std(similarities))
        threshold = mu - DEFAULT_THRESHOLD_STD_DEV * sigma

    # 标记低于阈值的转折点
    raw_breaks: set[int] = set()
    for i, sim in enumerate(similarities):
        if sim < threshold:
            raw_breaks.add(i + 1)  # 在第 i+1 句之前切

    # 应用最小 chunk 约束：去除太密集的切分点
    breaks = sorted(raw_breaks)
    filtered: list[int] = []
    last_break = 0
    for b in breaks:
        if b - last_break >= min_chunk and n - b >= min_chunk:
            filtered.append(b)
            last_break = b

    # 应用最大 chunk 约束：强制在 max_chunk 的倍数处切分
    result: list[int] = []
    last = 0
    for b in filtered:
        while b - last > max_chunk:
            forced = last + max_chunk
            result.append(forced)
            last = forced
        result.append(b)
        last = b
    # 尾部检查
    while n - last > max_chunk:
        last = last + max_chunk
        result.append(last)

    return sorted(set(result))


def chunk_section(
    section: Section,
    embedder: Embedder,
    threshold: float | None = None,
    overlap: int = DEFAULT_OVERLAP_SENTENCES,
    min_chunk: int = DEFAULT_MIN_CHUNK_SENTENCES,
    max_chunk: int = DEFAULT_MAX_CHUNK_SENTENCES,
) -> list[Chunk]:
    """对单个 Section 做语义切分，返回 Chunk 列表。

    Args:
        section: 文档节。
        embedder: 嵌入模型。
        threshold: 绝对阈值，None 时自适应。
        overlap: 相邻 chunk 重叠句子数。
        min_chunk: 最小句子数。
        max_chunk: 最大句子数。

    Returns:
        Chunk 列表，chunk.text 包含整节标题行 + chunk 文本 + 重叠。
    """
    sentences = split_sentences(section.content)
    if not sentences:
        return []

    breakpoints = compute_breakpoints(
        sentences, embedder, threshold, min_chunk, max_chunk
    )

    # 按切分点分段
    chunks: list[Chunk] = []
    start = 0
    chunk_idx = 0

    for bp in breakpoints:
        end = bp
        # 向后取 overlap 句
        overlap_end = min(end + overlap, len(sentences))
        chunk_sents = sentences[start:overlap_end]
        chunks.append(
            Chunk(
                text="\n".join(chunk_sents),
                section_id=section.section_id,
                chunk_index=chunk_idx,
                sentence_range=(start, overlap_end - 1),
            )
        )
        start = bp
        chunk_idx += 1

    # 最后一个 chunk
    if start < len(sentences):
        chunks.append(
            Chunk(
                text="\n".join(sentences[start:]),
                section_id=section.section_id,
                chunk_index=chunk_idx,
                sentence_range=(start, len(sentences) - 1),
            )
        )

    return chunks

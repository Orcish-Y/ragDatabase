"""共享测试装备 — MockEmbedder 供所有测试文件复用。"""
from __future__ import annotations

import numpy as np
import pytest

from rag_builder.embedder import Embedder


class MockEmbedder(Embedder):
    """共享 mock embedder，返回固定维度归一化随机向量。

    实现完整的 Embedder 协议：embed() + embed_batch() + similarity()。
    """

    def __init__(self, model_name: str = "mock", dim: int = 384):
        self._model_name = model_name
        self._dim = dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        rng = np.random.default_rng(42)
        vecs = rng.random((len(texts), self._dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return self.embed(texts)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    return MockEmbedder()

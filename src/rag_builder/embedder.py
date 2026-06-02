"""嵌入模型封装 — SentenceTransformer 包装器。"""

import os

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """SentenceTransformer 嵌入模型封装。

    延迟加载：模型在首次调用 embed() 时才加载，避免 CLI -h 时不必要的下载。
    """

    def __init__(self, model_name: str, mirror: str | None = None):
        self._model_name = model_name
        self._mirror = mirror
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            if self._mirror:
                os.environ.setdefault("HF_ENDPOINT", self._mirror)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        """将文本列表转为嵌入向量。

        Returns:
            shape (n_texts, embedding_dim) 的 numpy 数组。
        """
        if not texts:
            return np.array([])
        return self.model.encode(texts, normalize_embeddings=True)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算两个已归一化向量的余弦相似度。"""
        return float(np.dot(a, b))

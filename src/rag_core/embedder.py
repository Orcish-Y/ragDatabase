"""嵌入模型封装 — 阿里云 DashScope 在线 Embedding。

实时 API：语义切分 + MCP 检索 query embedding（≤10 条/请求）。
Batch API：建库大批量 embedding（OpenAI 兼容 Batch，50% 价格）。
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from openai import OpenAI

from .config import (
    BATCH_RETRY_BASE_DELAY,
    BATCH_RETRY_MAX,
    DASHSCOPE_BASE_URL,
    DEFAULT_MODEL_NAME,
    EMBED_BATCH_SIZE,
)


def _require_api_key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not key:
        raise RuntimeError(
            "未设置 DASHSCOPE_API_KEY。请在项目根目录 .env 文件中添加：\n"
            "  DASHSCOPE_API_KEY=sk-xxx"
        )
    return key


# 阿里云 DashScope text-embedding-v4 单条文本 token 上限 8192
# 中文字符 ≈ 1-2 tokens，保守取 4000 字符
MAX_CHARS_PER_TEXT = 4000


def _truncate_text(text: str) -> str:
    """截断过长文本至安全长度。"""
    if len(text) <= MAX_CHARS_PER_TEXT:
        return text
    return text[:MAX_CHARS_PER_TEXT]


class Embedder(ABC):
    """嵌入模型协议 — 所有 embedder 的抽象接口。

    两个 adapter：DashScopeEmbedder（在线）+ MockEmbedder（测试）。
    """

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """返回 shape (n_texts, embedding_dim) 的归一化向量。"""
        ...

    @abstractmethod
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算两个已归一化向量的余弦相似度。"""
        ...


class DashScopeEmbedder(Embedder):
    """阿里云 DashScope 在线 Embedding。

    延迟初始化 OpenAI client，首次调用 embed() 或 embed_batch() 时才检查 API key。
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        base_url: str | None = None,
    ):
        self._model_name = model_name
        self._base_url = base_url or os.environ.get(
            "DASHSCOPE_BASE_URL", DASHSCOPE_BASE_URL
        )
        self._client: OpenAI | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                api_key=_require_api_key(),
                base_url=self._base_url,
            )
        return self._client

    # ── 实时 API ──────────────────────────────────────────────

    def embed(self, texts: list[str]) -> np.ndarray:
        """实时 embedding，自动按 EMBED_BATCH_SIZE 分批。

        Returns:
            shape (n_texts, 1024) 的归一化 numpy 数组。
        """
        if not texts:
            return np.array([])

        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = [_truncate_text(t) for t in texts[i : i + EMBED_BATCH_SIZE]]
            resp = self.client.embeddings.create(
                model=self._model_name, input=batch
            )
            # resp.data 按 input 顺序排列
            all_vectors.extend(d.embedding for d in resp.data)

        arr = np.array(all_vectors, dtype=np.float32)
        # DashScope text-embedding-v4 返回的已是归一化向量，但为安全再归一化
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    # ── Batch API ─────────────────────────────────────────────

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """OpenAI Batch API，全自动：JSONL → 上传 → 轮询 → 下载 → 解析。

        内置 3 次指数退避重试，仅重试失败条目。

        Returns:
            shape (n_texts, 1024) 的归一化 numpy 数组。
        """
        if not texts:
            return np.array([])

        pending_texts = texts
        pending_indices = list(range(len(texts)))
        all_vectors: dict[int, list[float]] = {}

        for retry in range(BATCH_RETRY_MAX + 1):
            if not pending_texts:
                break

            if retry > 0:
                delay = BATCH_RETRY_BASE_DELAY * (2 ** (retry - 1))
                print(f"  重试第 {retry} 次（{len(pending_texts)} 条），等待 {delay}s...")
                time.sleep(delay)

            jsonl_content = _build_jsonl(
                pending_texts, pending_indices, self._model_name
            )
            batch_id = self._upload_and_run_batch(jsonl_content, retry)
            batch_vectors, failed = self._download_and_parse(batch_id, pending_indices)

            for idx, vec in batch_vectors.items():
                all_vectors[idx] = vec
            pending_texts = [texts[i] for i in failed]
            pending_indices = failed

        if pending_indices:
            raise RuntimeError(
                f"Batch embedding 失败：{len(pending_indices)} 条在 "
                f"{BATCH_RETRY_MAX} 次重试后仍未成功。"
                f"失败 custom_id: {pending_indices}"
            )

        # 按原始顺序排列
        result = np.array(
            [all_vectors[i] for i in range(len(texts))], dtype=np.float32
        )
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return result / norms

    def _upload_and_run_batch(self, jsonl_content: str, retry: int) -> str:
        """上传 JSONL，创建并等待 batch 任务完成。"""
        label = "重新提交" if retry > 0 else "提交"
        print(f"  {label} Batch 任务（{jsonl_content.count(chr(10))} 条）...")

        # 写入临时文件
        fd, tmp_path = tempfile.mkstemp(suffix=".jsonl", prefix="rag_batch_")
        try:
            with open(fd, "w", encoding="utf-8") as f:
                f.write(jsonl_content)

            # 上传
            with open(tmp_path, "rb") as f:
                file_obj = self.client.files.create(file=f, purpose="batch")
            print(f"  文件已上传: {file_obj.id}")
        finally:
            os.unlink(tmp_path)

        # 创建 batch 任务
        batch = self.client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/embeddings",
            completion_window="24h",
        )
        batch_id = batch.id
        print(f"  Batch 任务已创建: {batch_id}，等待完成...")

        # 轮询
        dots = 0
        while True:
            batch = self.client.batches.retrieve(batch_id)
            status = batch.status
            if status in ("completed", "failed", "expired", "cancelled"):
                print()
                break
            if dots % 20 == 0 and dots > 0:
                print()
            print(".", end="", flush=True)
            time.sleep(10)
            dots += 1

        if status != "completed":
            errors = getattr(batch, "errors", None)
            raise RuntimeError(
                f"Batch 任务未成功完成（状态: {status}）。"
                f"{' 错误: ' + str(errors.data) if errors else ''}"
            )

        print(f"  Batch 任务完成")
        return batch_id

    def _download_and_parse(
        self, batch_id: str, expected_indices: list[int]
    ) -> tuple[dict[int, list[float]], list[int]]:
        """下载 batch 结果，返回 (成功映射, 失败索引列表)。"""
        print("  下载结果...")
        batch = self.client.batches.retrieve(batch_id)

        if not batch.output_file_id:
            raise RuntimeError(f"Batch {batch_id} 无输出文件")

        content = self.client.files.content(batch.output_file_id)

        vectors: dict[int, list[float]] = {}
        failed: list[int] = []

        for line in content.text.strip().split("\n"):
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = int(obj.get("custom_id", -1))

            # 检查响应状态
            status_code = obj.get("response", {}).get("status_code", 0)
            if status_code != 200:
                failed.append(cid)
                continue

            body = obj.get("response", {}).get("body", {})
            data = body.get("data", [])
            if data:
                vectors[cid] = data[0].get("embedding", [])
            else:
                failed.append(cid)

        succeeded = len(vectors)
        failed_count = len(failed)
        print(f"  成功: {succeeded}, 失败: {failed_count}")
        return vectors, failed

    # ── 相似度 ────────────────────────────────────────────────

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算两个已归一化向量的余弦相似度。"""
        return float(np.dot(a, b))


def _build_jsonl(
    texts: list[str], indices: list[int], model: str
) -> str:
    """构建 Batch API 输入 JSONL 字符串。"""
    lines = []
    for idx, text in zip(indices, texts):
        line = {
            "custom_id": str(idx),
            "method": "POST",
            "url": "/v1/embeddings",
            "body": {
                "model": model,
                "input": _truncate_text(text),
            },
        }
        lines.append(json.dumps(line, ensure_ascii=False))
    return "\n".join(lines) + "\n"

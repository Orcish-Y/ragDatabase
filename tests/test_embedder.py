"""测试 DashScopeEmbedder — 实时 API 分批 + Batch API + 重试。"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_builder.embedder import DashScopeEmbedder, _build_jsonl


# ── _build_jsonl ───────────────────────────────────────────────

class TestBuildJSONL:
    def test_basic(self):
        texts = ["文本一", "文本二"]
        result = _build_jsonl(texts, [0, 1], "text-embedding-v4")
        lines = result.strip().split("\n")
        assert len(lines) == 2
        # 每行都是合法 JSON
        import json
        for line in lines:
            obj = json.loads(line)
            assert obj["method"] == "POST"
            assert obj["url"] == "/v1/embeddings"
            assert obj["body"]["model"] == "text-embedding-v4"
            assert "input" in obj["body"]

    def test_custom_id_matches_index(self):
        texts = ["a", "b", "c"]
        result = _build_jsonl(texts, [5, 10, 15], "m")
        lines = result.strip().split("\n")
        import json
        ids = [json.loads(l)["custom_id"] for l in lines]
        assert ids == ["5", "10", "15"]


# ── DashScopeEmbedder ──────────────────────────────────────────

def _make_mock_embedding_response(texts: list[str]):
    """构造 OpenAI embeddings.create 的 mock 返回值。"""
    rng = np.random.default_rng(42)
    data_items = []
    for i, _ in enumerate(texts):
        vec = rng.random(1024).astype(np.float32).tolist()
        data_items.append(
            MagicMock(embedding=vec, index=i)
        )
    return MagicMock(data=data_items)


class TestDashScopeEmbedderRealTime:
    """实时 embed() 测试。"""

    def test_empty_input(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
        e = DashScopeEmbedder()
        result = e.embed([])
        assert isinstance(result, np.ndarray)
        assert result.size == 0

    def test_single_batch(self, monkeypatch):
        """单批 ≤10 条，一次请求完成。"""
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
        e = DashScopeEmbedder()
        texts = ["句1", "句2", "句3"]

        mock_resp = _make_mock_embedding_response(texts)
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_resp

        e._client = mock_client
        result = e.embed(texts)

        assert result.shape == (3, 1024)
        assert result.dtype == np.float32
        # 验证归一化
        norms = np.linalg.norm(result, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)
        # 验证只调用了一次
        mock_client.embeddings.create.assert_called_once()

    def test_multi_batch_split(self, monkeypatch):
        """25 条 → 3 批（10+10+5）。"""
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
        e = DashScopeEmbedder()
        texts = [f"t{i}" for i in range(25)]

        call_count = [0]

        def side_effect(model, input):
            call_count[0] += 1
            return _make_mock_embedding_response(input)

        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = side_effect

        e._client = mock_client
        result = e.embed(texts)

        assert result.shape == (25, 1024)
        assert call_count[0] == 3

    def test_missing_api_key(self, monkeypatch):
        """未设置 key 时报错。"""
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        e = DashScopeEmbedder()
        with patch.object(e, "_client", None):
            with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
                _ = e.client


class TestDashScopeEmbedderBatch:
    """Batch embed_batch() 测试。"""

    def test_empty_input(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
        e = DashScopeEmbedder()
        result = e.embed_batch([])
        assert result.size == 0

    def test_happy_path(self, monkeypatch):
        """正常流程：上传 → 创建 → 轮询(completed) → 下载。"""
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
        e = DashScopeEmbedder()
        texts = ["t1", "t2", "t3"]

        rng = np.random.default_rng(42)
        vectors = {
            0: rng.random(1024).astype(np.float32).tolist(),
            1: rng.random(1024).astype(np.float32).tolist(),
            2: rng.random(1024).astype(np.float32).tolist(),
        }

        # 构造 batch 结果 JSONL
        import json
        result_lines = []
        for i in range(3):
            result_lines.append(json.dumps({
                "custom_id": str(i),
                "response": {
                    "status_code": 200,
                    "body": {"data": [{"embedding": vectors[i]}]},
                },
            }))

        mock_file = MagicMock()
        mock_file.id = "file-123"

        mock_batch_initial = MagicMock()
        mock_batch_initial.id = "batch-456"
        mock_batch_initial.status = "validating"

        mock_batch_completed = MagicMock()
        mock_batch_completed.id = "batch-456"
        mock_batch_completed.status = "completed"
        mock_batch_completed.output_file_id = "output-789"

        # _download_and_parse 也会调用一次 retrieve
        mock_batch_for_download = MagicMock()
        mock_batch_for_download.id = "batch-456"
        mock_batch_for_download.status = "completed"
        mock_batch_for_download.output_file_id = "output-789"

        mock_output = MagicMock()
        mock_output.text = "\n".join(result_lines)

        mock_client = MagicMock()
        mock_client.files.create.return_value = mock_file
        mock_client.batches.create.return_value = mock_batch_initial
        mock_client.batches.retrieve.side_effect = [
            MagicMock(status="validating"),
            MagicMock(status="in_progress"),
            MagicMock(status="finalizing"),
            mock_batch_completed,
            mock_batch_for_download,  # _download_and_parse 调用
        ]
        mock_client.files.content.return_value = mock_output

        e._client = mock_client
        result = e.embed_batch(texts)

        assert result.shape == (3, 1024)
        norms = np.linalg.norm(result, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_partial_failure_retry(self, monkeypatch):
        """模拟部分条目失败 → 自动重试。"""
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
        e = DashScopeEmbedder()
        texts = ["a", "b", "c"]

        import json
        rng = np.random.default_rng(99)
        vec_a = rng.random(1024).astype(np.float32).tolist()
        vec_b = rng.random(1024).astype(np.float32).tolist()
        vec_c = rng.random(1024).astype(np.float32).tolist()

        # 第一次：custom_id=1 失败
        first_output_lines = [
            json.dumps({"custom_id": "0", "response": {"status_code": 200, "body": {"data": [{"embedding": vec_a}]}}}),
            json.dumps({"custom_id": "1", "response": {"status_code": 500, "body": {}}}),  # 失败
            json.dumps({"custom_id": "2", "response": {"status_code": 200, "body": {"data": [{"embedding": vec_c}]}}}),
        ]
        # 重试：custom_id=1 成功
        retry_output_lines = [
            json.dumps({"custom_id": "1", "response": {"status_code": 200, "body": {"data": [{"embedding": vec_b}]}}}),
        ]

        mock_file_1 = MagicMock(id="file-1")
        mock_file_2 = MagicMock(id="file-2")

        mock_client = MagicMock()
        mock_client.files.create.side_effect = [mock_file_1, mock_file_2]
        mock_client.batches.create.side_effect = [
            MagicMock(id="batch-1", status="validating"),
            MagicMock(id="batch-2", status="validating"),
        ]

        output1 = MagicMock()
        output1.text = "\n".join(first_output_lines)
        output2 = MagicMock()
        output2.text = "\n".join(retry_output_lines)

        mock_client.files.content.side_effect = [output1, output2]

        # 每次 retrieve 都是 completed（简化）
        mock_client.batches.retrieve.side_effect = [
            # batch-1: completed
            MagicMock(id="batch-1", status="completed", output_file_id="out-1"),
            MagicMock(id="batch-1", status="completed", output_file_id="out-1"),
            # batch-2: completed
            MagicMock(id="batch-2", status="completed", output_file_id="out-2"),
            MagicMock(id="batch-2", status="completed", output_file_id="out-2"),
        ]

        e._client = mock_client
        with patch("rag_core.embedder.time.sleep", return_value=None):
            result = e.embed_batch(texts)

        assert result.shape == (3, 1024)
        # 应该调用了两次 upload（第一次 + 重试）
        assert mock_client.files.create.call_count == 2

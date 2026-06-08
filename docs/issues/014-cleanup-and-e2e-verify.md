## Parent

[PRD: 阿里云 DashScope 在线 Embedding 替换本地模型](../PRD-dashscope-embedding.md)

## What to build

最终清理和集成验证：

1. `pyproject.toml` — 移除 `sentence-transformers>=3.0.0`（已在方案讨论中改好，确认）
2. 删除旧 `Embedder` 类（如果还有残留引用）
3. `config.py` — `DEFAULT_MODEL_NAME` 改为 `"text-embedding-v4"`
4. 新增 `DASHSCOPE_BASE_URL` 常量
5. 全量测试：`pytest tests/ -v` 全部通过
6. 确认 `.env` 已加入 `.gitignore`

## Acceptance criteria

- [ ] `pyproject.toml` 中无 `sentence-transformers` 依赖
- [ ] `config.py` 中 `DEFAULT_MODEL_NAME = "text-embedding-v4"`
- [ ] `config.py` 中无 `DEFAULT_MODEL_NAME` 旧值残留
- [ ] 代码仓库中 `sentence_transformers` / `SentenceTransformer` / `hf_mirror` 无引用
- [ ] `pytest tests/ -v` 全量通过
- [ ] `.gitignore` 包含 `.env`

## Blocked by

- #012 CLI — rag-build 接入 DashScope
- #013 MCP — rag-mcp 接入 DashScope

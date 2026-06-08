## Parent

[PRD: 阿里云 DashScope 在线 Embedding 替换本地模型](../PRD-dashscope-embedding.md)

## What to build

更新 `mcp_server.py`（`rag-mcp` entry point）：

- 用 `DashScopeEmbedder` 替换 `Embedder`
- 移除 `--hf-mirror` 参数
- 启动时 `load_dotenv()` 加载 `.env`
- `--model` 改为可选，默认 `text-embedding-v4`
- MCP `search_rag` tool 行为保持不变 — Chroma 内部通过 `_EmbedderAdapter` 调用 `embedder.embed()` 做 query embedding

由于 `DashScopeEmbedder` 与旧 `Embedder` 接口兼容（`embed()` + `similarity()`），VectorStore 初始化无需改动。

## Acceptance criteria

- [ ] `rag-mcp --db <db_dir>` 启动，加载 Chroma 数据库
- [ ] `search_rag(query, k)` 可正常返回搜索结果（含 section_id、content、score）
- [ ] 不支持 `--hf-mirror` 参数
- [ ] 缺少 `DASHSCOPE_API_KEY` 时，启动不报错（仅在首次 `search_rag` 调用时报错）
- [ ] 已有 `test_mcp_server.py` 测试用 MockEmbedder 通过
- [ ] SSE 模式（`--transport sse --port 8080`）可用

## Blocked by

- #009 DashScopeEmbedder — 实时 embedding（需要 `DashScopeEmbedder` 类）

# PRD: 阿里云 DashScope 在线 Embedding 替换本地模型

## Problem Statement

当前 RAG 建库工具使用本地 SentenceTransformer 模型（`paraphrase-multilingual-MiniLM-L12-v2`）做嵌入。这带来两个问题：

1. **部署负担**：本地模型需要下载（~500MB）、加载到内存，启动慢，且模型文件不能随 Chroma 数据库一起迁移到另一台机器
2. **无法拆分为轻量检索服务**：用户希望建库后用 Batch API 省钱，然后把建好的 Chroma 数据库复制到另一台机器，只跑一个极简的 MCP 检索服务（不含建库逻辑、不含本地模型）

## Solution

用阿里云 DashScope `text-embedding-v4` 在线 API 完全替换本地 SentenceTransformer。建库时默认走 OpenAI 兼容 Batch API（实时价格的 50%），检索时走实时 API。整个工具保持单一 Python 包、两个 CLI entry point 的架构。

## User Stories

1. 作为一个 RAG 建库使用者，我希望用阿里云在线 embedding 建库，不需要在本地下载和管理任何模型文件
2. 作为一个 RAG 建库使用者，我希望建库默认使用 Batch API，以节省一半的 embedding 费用
3. 作为一个 RAG 建库使用者，我希望语义切分仍然能正常工作 —— 即使建库走 Batch，切分阶段的 sentence similarity 也能实时完成
4. 作为一个 RAG 建库使用者，我希望建库时不需要传 API key 到命令行，用 `.env` 文件管理密钥更安全
5. 作为一个 RAG 建库使用者，我希望 Batch API 的进度在终端有直观的输出（已提交、等待中、下载结果）
6. 作为一个 RAG 建库使用者，我希望如果 Batch 中有部分向量化失败，系统能自动重试（3 次，指数退避），而不是静默丢数据
7. 作为一个 RAG 检索服务使用者，我希望能把建好的 Chroma 数据库目录复制到另一台机器，安装同一个包后直接启动 `rag-mcp`，不需要模型文件
8. 作为一个 RAG 检索服务使用者，我希望检索时的 query embedding 通过实时 API 完成，延迟在毫秒级
9. 作为一个 RAG 检索服务使用者，我希望实时 API 的认证同样走 `.env` 文件，不需要额外的命令行参数
10. 作为一个 RAG 开发者，我希望换模型后向量维度自动切换（从 384 到 1024），不需要手动改配置
11. 作为一个 RAG 开发者，我希望新增的 `embed_batch()` 方法和现有的 `embed()` 方法在同一个 Embedder 接口上，方便 MockEmbedder 在测试中复用
12. 作为一个 RAG 开发者，我希望去掉 `sentence-transformers` 和 `--hf-mirror` 等旧依赖和参数，保持代码库清爽

## Implementation Decisions

### 架构：单包双入口

保持 `rag-builder` 单一 Python 包，两个 CLI entry point：
- `rag-build`：建库入口（含 markdown 解析、语义切分、Batch embedding、Chroma 写入）
- `rag-mcp`：检索服务入口（加载 Chroma + 实时 embedding + MCP stdio/SSE）

检索服务只依赖 `chromadb`、`langchain-chroma`、`openai`、`mcp`，不引入 markdown 解析或 chunker 的运行时开销（import 层面的依赖不可避免，但运行时不会触发）。

### SDK 选择

使用 `openai` Python 包，通过阿里云 OpenAI 兼容端点 `https://dashscope.aliyuncs.com/compatible-mode/v1` 访问。

### 嵌入模型

`text-embedding-v4`（1024 维），中英文支持，向量归一化输出（余弦相似度兼容）。

### 两种 API 模式

| 方法 | 用途 | API | 限制 |
|------|------|-----|------|
| `embed(texts)` | 语义切分、MCP query | 实时 `/v1/embeddings` | 10 条/请求，8192 tokens/请求 |
| `embed_batch(texts)` | 建库大批量 | OpenAI Batch API | 50% 价格，分钟级延迟 |

`embed()` 内部自动按 10 条分批发送，对调用方透明。
`embed_batch()` 全自动封装 JSONL 构建、上传、轮询、结果下载。内置 3 次指数退避重试（1s→2s→4s），仅重试失败条目。

### 认证

通过项目根目录 `.env` 文件加载，使用 `python-dotenv`：
```
DASHSCOPE_API_KEY=sk-xxx
```
可选覆盖：
```
DASHSCOPE_BASE_URL=https://custom-endpoint/v1
```

CLI 不暴露 `--api-key` 参数。启动时 `load_dotenv()` 加载，未找到 `DASHSCOPE_API_KEY` 则报错退出。

### Embedder 接口

```python
class DashScopeEmbedder:
    model_name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """实时 embedding，shape (n_texts, 1024)，归一化。"""
        ...

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """OpenAI Batch API，全自动 + 重试。"""
        ...

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """余弦相似度（已归一化向量直接点积）。"""
        ...
```

此接口与原 `Embedder` 完全兼容 —— `embed()` 和 `similarity()` 签名不变，新增 `embed_batch()`。

### VectorStore 新增 Batch 建库路径

新增方法负责建库时的大批量 embedding，分三步：
1. 对所有 Section 做语义切分（仍调用 `embedder.embed()` 实时完成 sentence similarity 判断）
2. 收集所有 chunk 文本，一次调用 `embedder.embed_batch()`
3. 拿到向量后手动写入 Chroma（bypass LangChain 的 `add_documents`，因为它会再次触发实时 embedding）

增量更新逻辑（哈希比对、skip/update/insert）全部保留。

### 移除的依赖和参数

- 移除 `sentence-transformers` 依赖
- 移除 `Embedder` 类（旧本地模型实现）
- 移除 `--hf-mirror` CLI 参数
- `--model` 默认值改为 `text-embedding-v4`

## Testing Decisions

### 测试原则

- 只测试外部行为，不测试实现细节（如 JSONL 文件格式、轮询间隔）
- 在 `Embedder` 接口层面 mock，测试 VectorStore / CLI / MCP 的行为
- `DashScopeEmbedder` 用 mock `openai.OpenAI` 测试分批、重试、解析逻辑

### 测试模块

| 模块 | 测试方式 |
|------|----------|
| `DashScopeEmbedder.embed()` | Mock OpenAI client，验证 10 条分批 + 结果拼接 |
| `DashScopeEmbedder.embed_batch()` | Mock OpenAI client，验证 JSONL 构造 + 重试逻辑 |
| `VectorStore.add_sections_batch()` | 用 MockEmbedder（新增 `embed_batch` 方法），验证建库 + 检索端到端 |
| CLI `rag-build` | 已有 `test_cli.py`，适配新参数 |
| MCP `rag-mcp` | 已有 `test_mcp_server.py`，用 MockEmbedder |

### 现有测试复用

`tests/test_vector_store.py` 的 `MockEmbedder` 是最高价值 seam —— 它已有 `embed()` 和 `similarity()`，只需新增 `embed_batch()`（直接委托给 `embed()` 返回相同向量），所有现有 VectorStore 测试即可无缝适配 batch 模式。

## Out of Scope

- **重排序 (Rerank)**：Cross-Encoder 重排功能不在本次范围
- **多模型切换**：只支持 `text-embedding-v4`，不支持通过参数切换到其他阿里云模型或其他服务商
- **混合本地+在线**：不支持回退到本地模型，旧 SentenceTransformer 代码完全删除
- **Rate limit 重试**：实时 API 的 429 限流重试不在本次范围（`openai` SDK 自带基础重试）
- **Batch 失败通知**：不实现 webhook/回调，仅通过终端 print 和最终报错

## Further Notes

- 切换模型意味着旧 Chroma 数据库（384 维向量）与新版（1024 维）不兼容，必须用新 embedder 重新建库
- `_EmbedderAdapter`（LangChain 适配器）保留，供 MCP 检索时的 query embedding 使用
- MCP 服务运行时不需要 `.env` 中有 `DASHSCOPE_API_KEY`（但搜索时需要，否则 query embedding 会报错）

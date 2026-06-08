# RAG Builder

通用 RAG 知识库工具 — 语义切分 + 父文档检索器 + 轻量 MCP 检索服务。将 Markdown 文档构建为 Chroma 向量数据库。

**嵌入模型：** 阿里云 DashScope `text-embedding-v4` (1024维)，建库默认走 OpenAI 兼容 Batch API（实时价格的 50%）。

## 安装

```bash
pip install -e .
```

## 快速开始

### 1. 配置 API Key

```bash
# 创建 .env 文件
echo "DASHSCOPE_API_KEY=sk-xxx" > .env
```

### 2. 建库

```bash
rag-build -i 文档.md -d ./my_db \
    --max-chunk 10 --threshold 0.7 \
    --category pharmaceutical
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-i` | 输入 Markdown 文件 | 必填 |
| `-d` | 数据库输出目录 | 必填 |
| `--threshold` | 语义切分相似度阈值 | 自适应 |
| `--max-chunk` | 最大句子数/chunk | 20 |
| `--chapter-map` | 章映射 JSON | 无 |
| `--category` | 分类标签 (写入 metadata) | 无 |

### 3. 启动 MCP 检索服务

```bash
# stdio 模式
rag-mcp --db ./my_db

# HTTP/SSE 模式
rag-mcp --db ./my_db --transport sse --port 8080
```

### 4. 检索

```python
from rag_core.embedder import DashScopeEmbedder
from rag_core.vector_store import VectorStore

store = VectorStore("my_db", DashScopeEmbedder())

# 语义搜索
results = store.vectorstore.similarity_search("崩解时限", k=5)

# 按分类过滤
results = store.vectorstore.similarity_search(
    "崩解时限", k=5,
    where={"category": "pharmaceutical"}
)

# MCP Tool 调用
search_rag(query="崩解时限检查", k=5)
```

## 架构

```
src/
├── rag_core/        # 共享核心：Embedder 协议 + VectorStore (只读)
├── rag_builder/      # 建库工具：CLI + 语义切分 + Batch API
└── rag_server/       # 轻量检索：MCP 服务 (可独立部署)
```

- **rag_builder** — 文档解析 → 语义切分 → Batch embedding → Chroma 写入
- **rag_server** — 加载 Chroma → 实时 query embedding → MCP Tool `search_rag`
- **rag_core** — 两者的共同依赖，无需重复代码

### 检索流程

```
Query → embed(实时API) → chunk 索引匹配 → section_id 去重 → 父文档反查 → 返回整节全文
```

### 增量更新

建库时自动比对 `content_hash`：
- 内容未变 → 跳过（或仅更新 metadata）
- 内容变更 → 删旧写新
- 新节 → 插入

### 部署到其他机器

```bash
# 拷贝 rag_server + 数据库
scp -r src/rag_server/ user@target:/path/
scp -r db/my_db/ user@target:/path/data/

# 目标机器
pip install openai python-dotenv chromadb langchain-chroma "mcp[cli]"
echo "DASHSCOPE_API_KEY=sk-xxx" > .env
python -m rag_server --db /path/data/my_db
```

## 项目结构

```
src/rag_core/       共享核心 (Embedder + VectorStore)
src/rag_builder/     建库工具 (CLI + 切分 + 文档解析)
src/rag_server/      检索服务 (MCP, 可独立部署)
tests/              测试 (61 tests)
docs/               文档 & ADR & Issues
```

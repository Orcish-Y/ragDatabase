# PRD: RAG MCP 检索服务

- **状态**: Ready for Agent
- **日期**: 2026-06-02

---

## Problem Statement

RAG 建库脚本已完成，向量库（Chroma）已就位。但检索能力被封存在脚本内部——AI 客户端（Claude Code 等）无法直接查询知识库。用户需要一个符合 MCP 协议的服务，能接受查询并返回完整节全文，让 AI 可以在对话中引用药典等权威文档的原始内容。

## Solution

在 `rag_builder` 包内新增 MCP 检索服务模块。基于 FastMCP 框架，暴露单个 Tool `search_rag(query, k)`。建库脚本同步改造——增加父文档 Collection 写入，使得 MCP 服务可以从 chunk 命中反查到完整节全文返回。服务启动时仅需一个参数（数据库路径），与 Claude Code 等 AI 客户端通过 stdio 直连。

## User Stories

1. 作为 AI 应用开发者，我希望用一条命令启动 MCP 检索服务，只需传入数据库路径，其他零配置
2. 作为 AI 应用开发者，我希望服务同时支持 stdio 和 HTTP/SSE 两种传输方式，以适配不同的客户端环境
3. 作为终端用户（通过 AI），当我问"片剂的崩解时限是什么"时，希望从知识库中检索到完整的药典节全文，而非碎片化的 chunk 段落
4. 作为终端用户（通过 AI），我希望检索结果附带节编号、标题、章名等元数据，以便知道信息来源的可信度
5. 作为终端用户（通过 AI），当多个 chunk 命中同一节时，希望结果去重，不会看到重复的同一节
6. 作为数据工程师，当我用建库脚本重新建库后，希望 MCP 服务能立即读取到最新的向量库，无需额外配置
7. 作为数据工程师，我希望 MCP 服务是 `rag_builder` 包的一部分，与建库脚本共享嵌入式模型加载逻辑，不引入额外的依赖管理负担
8. 作为运维人员，我希望未来的重排（Cross-Encoder Rerank）功能有一个清晰的扩展点，不必重构现有接口

## Implementation Decisions

### 新增模块
- 新建 MCP 服务模块，基于 FastMCP 框架，暴露单个 Tool
- Tool 签名：`search_rag(query: str, k: int = 5) -> list[SearchResult]`

### MCP 传输方式
- 默认传输：stdio（适配 Claude Code 等本地 AI 客户端）
- 可选传输：HTTP/SSE，通过 `--transport sse --port 8080` 切换
- 启动 CLI：`python -m rag_builder.mcp_server --db ./db [--transport sse --port 8080]`

### 父文档存储改造
- 建库脚本新增写入第二个 Chroma Collection `rag_parents`
- `rag_parents` 以 `section_id` 为主键，存储完整节全文 + 元数据
- 不参与向量相似度索引，仅用于按 ID 查询
- 建库时与 chunk 索引同步写入，确保数据一致性

### 检索流程
1. MCP 服务从配置的 Chroma 目录加载两个 Collection
2. 收到 `search_rag(query, k)` 调用
3. 用 query 在 `rag_documents` 中做向量相似度搜索，取 top-20
4. 从命中结果的 `section_id` 去重
5. 用去重后的 `section_id` 列表在 `rag_parents` 中按 ID 查询父文档全文
6. 返回 top-k 条完整节全文 + 元数据 + 相似度分数

### 返回格式
```json
[
  {
    "section_id": "0101",
    "section_title": "片剂",
    "chapter": "制剂通则",
    "content": "片剂系指原料药物...（整节全文）",
    "score": 0.87
  }
]
```

### 重排预留
- Tool 内部预留 Rerank 逻辑插入点
- `--rerank-model` CLI 参数已定义但 MVP 阶段标记为"实验性"，不实现重排逻辑
- 第一阶段直接用 Bi-Encoder embedding 相似度排序

### 与建库脚本的关系
- 建库脚本和 MCP 服务共享 `embedder.py`、`vector_store.py`
- 建库脚本修改后需向后兼容：旧库（无 `rag_parents` Collection）启动 MCP 服务时给出明确错误提示，提示用户重新建库

### 依赖
- `mcp[cli]`（FastMCP 包含在内）
- 已有依赖：`chromadb`、`langchain-chroma`、`sentence-transformers`

## Testing Decisions

### 测试原则
- 只测外部行为：MCP 服务能否启动、`search_rag` 返回格式和内容是否正确
- 不测 FastMCP 框架内部行为
- 用 mini Markdown fixture 建的小型 Chroma DB 作为测试数据

### 测试层级

| 接缝 | 测试内容 | 断言 |
|------|---------|------|
| MCP 服务启动 | `--db ./test_db` → 进程启动成功 | 无异常退出，Chroma 连接正常 |
| search_rag 返回格式 | 调用 `search_rag("片剂崩解", k=3)` | 返回列表长度 ≤3，每条有 section_id/title/chapter/content/score |
| 父文档完整性 | 同上 | 返回的 content 长度 >500 字符（非 chunk 片段），包含节内多个段落 |
| section_id 去重 | 同节多 chunk 命中的查询 | 无重复 section_id |
| 空查询容错 | `search_rag("xyz不存在", k=3)` | 不抛异常，返回空列表或低分结果 |
| 建库→MCP 集成 | 建库完成 → 立即启动 MCP → 可搜索 | rent_parents Collection 存在且数据完整 |
| 两个 Collection | 检查 Chroma 目录 | `rag_documents` 和 `rag_parents` 均存在 |

### 测试依赖
- 使用已有测试中的 mini Markdown fixture 和 MockEmbedder
- MCP Tool 的调用可以使用 FastMCP 的测试客户端或直接 import 底层函数测试
- 不需要真实 MCP 客户端连接

## Out of Scope

- **Cross-Encoder 重排**：MVP 不实现，参数和代码插入点预留
- **MCP Resources 和 Prompts**：当前只需 Tool，不暴露 Resource/Prompt
- **多用户/认证**：本地单用户服务，无鉴权机制
- **多数据库切换**：一次只服务一个数据库目录，不支持运行时热切换
- **流式响应**：search_rag 一次性返回全部结果，不支持 SSE 流式逐条输出
- **Feedback/日志持久化**：不记录查询历史或用户反馈

## Further Notes

- 当前目标数据库：`db/pharmacopoeia_db`（349 节，348 chunks）
- 父文档总大小：约 4MB（药典全文），加载到 Chroma 后内存占用量可忽略
- 嵌入模型与建库时相同：`paraphrase-multilingual-MiniLM-L12-v2`（384 维）
- 相关 ADR：`docs/adr/0002-mcp-server-design.md`
- 相关术语：`CONTEXT.md`
- Claude Code 配置示例：
  ```json
  {
    "mcpServers": {
      "pharmacopoeia-rag": {
        "command": "python",
        "args": ["-m", "rag_builder.mcp_server", "--db", "./db/pharmacopoeia_db"]
      }
    }
  }
  ```

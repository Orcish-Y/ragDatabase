# ADR-0002: MCP 服务接口设计

| 字段 | 值 |
|------|-----|
| 日期 | 2026-06-02 |
| 状态 | Accepted |
| 干系人 | Orcish_Y |

## 背景

RAG 建库脚本已完成，需要构建 MCP 检索服务暴露给 AI 客户端（Claude Code 等）使用。服务需要：接受数据库路径配置，检索 chunk 后按 section_id 反查整节全文返回。

## 决策

### 1. 单 Tool 接口，每个请求直接返回完整节全文

**选择**: `search_rag(query: str, k: int = 5)` 一次调用返回 top-k 节的完整全文 + 元数据 + 相似度分数。

**替代方案**:
- **Tool + Resource 分离**: `search_rag` 返回 ID 列表 → AI 逐个调用 `rag://sections/{id}` 拉全文。缺点：多轮调用增加延迟和复杂度，不符合"一次拿全"的用户预期。
- **Tool 只返回 chunk**: AI 拿到的只是碎片文本。缺点：丢失上下文，药典这类权威文档需要完整条目引用。

**选择理由**: AI 客户端发起一次搜索，拿到完整答案素材，自行组织回答。减少调用轮次。

### 2. 父文档存储：独立 Chroma Collection

**选择**: 建库时写入第二个 Collection `rag_parents`，以 section_id 为主键存储全文，不参与向量索引。

**替代方案**:
- **InMemoryStore**: LangChain 默认。缺点：进程重启丢失，每次启动需重建。
- **JSON 文件**: 独立于 Chroma。缺点：需维护两个存储格式的读写逻辑。
- **Redis/ES**: 太重，单机工具不需要。

**选择理由**: Chroma 原生支持多 Collection，与 chunk 索引共存于同一目录，零额外依赖。

### 3. FastMCP 框架 + stdio 默认传输

**选择**: FastMCP 装饰器风格 API，默认 stdio 传输，可选 `--transport sse --port 8080`。

**替代方案**:
- **Raw MCP SDK**: 灵活但模板代码多。N 个 Tool 时考虑，当前仅 1 个 Tool，不值得。
- **仅 HTTP**: 不支持 Claude Code 等本地工具的标准对接方式。

**选择理由**: FastMCP 是最小可行路径。stdio 是 MCP 生态的标准传输方式。

### 4. 合入 rag_builder 包

**选择**: `src/rag_builder/mcp_server.py` 作为子模块，与建库共享 embedder/config。

**替代方案**:
- **独立包 `servers/mcp/`**: 需要独立的 pyproject.toml 和依赖管理，当前阶段过度。

**选择理由**: MCP 服务和建库脚本共用 Embedder、VectorStore 等组件，放在同一包内自然且无冗余。

### 5. 不引入 Cross-Encoder 重排（MVP）

**选择**: MVP 阶段只做 embedding 相似度排序，预留在 `--rerank-model` 参数。

**选择理由**: Cross-Encoder 模型需额外下载（~100-300MB），且对每个查询增加 batch_size × 推理时间的延迟。当前 384 维 embedding 的检索精度对药典类结构化文档已足够。

## 后果

- **建库脚本需修改**: 增加写入 `rag_parents` Collection 的逻辑，向后兼容——旧库需重建。
- **Chroma 目录包含两个 Collection**: 迁移/备份时需整体复制，不可只复制 `rag_documents`。
- **重排预留参数不实现**: `--rerank-model` 参数当前不生效，文档中标记为"实验性"。

# #3 — Chroma 向量库构建

- **Type**: AFK
- **Label**: `done`
- **Blocked by**: #2

## What to build

将文档解析 + 语义切分 + 嵌入向量化串成完整流水线，输出到 Chroma 向量数据库。使用 LangChain ParentDocumentRetriever：小 chunk 做向量索引，整节 Section 做父文档存储。

端到端行为：
1. 用户执行 `rag-build -i 药典.md -d ./pharmacopoeia_db`
2. 脚本完成解析 → 切分 → 嵌入 → 入库
3. 输出统计：总节数 N，总 chunk 数 M
4. 任何人可以用 Chroma 客户端打开 `./pharmacopoeia_db` 检索，返回整节全文 + 元数据

## Acceptance criteria

- [ ] `vector_store.py` 实现 Chroma 初始化（持久化目录模式）
- [ ] 使用 LangChain `ParentDocumentRetriever`：小 chunk 索引 + 大 chunk 存储
- [ ] 父文档 = 完整 Section 全文（`####` 标题到下一 `####` 之间的全部内容）
- [ ] 同节多个 chunk 命中时自动去重，只返回一节一次
- [ ] 每个 chunk 存储 6 个元数据字段：section_id, section_title, chapter, source_file, content_hash, chunk_index
- [ ] 10 个 User Story 中 #1, #4, #5 的核心路径打通
- [ ] 默认 collection name: `rag_documents`
- [ ] 单元测试：用 mini .md 建库后，Chroma 客户端可检索到正确节
- [ ] 单元测试：验证元数据字段完整且正确

## Blocked by

- #2（需要 chunker + embedder）

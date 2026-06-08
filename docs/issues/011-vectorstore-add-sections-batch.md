## Parent

[PRD: 阿里云 DashScope 在线 Embedding 替换本地模型](../PRD-dashscope-embedding.md)

## What to build

在 `VectorStore` 上新增 `add_sections_batch()` 方法，替代旧的逐 section 实时 embedding 建库流程。三步走：

1. 对所有 Section 调用 `chunk_section()` 做语义切分（仍走 `embedder.embed()` 实时计算 sentence similarity）
2. 收集所有 chunk 文本，一次调用 `embedder.embed_batch()` 获取全部向量
3. 拿到向量后手动调用 Chroma `_collection.add(embeddings=..., documents=..., metadatas=..., ids=...)` 写入，同时写 `rag_documents` 和 `rag_documents_parents` 双 Collection

增量更新逻辑（哈希比对 → skip/update/insert）全部保留。`add_sections_batch()` 返回与 `add_sections()` 相同的统计字典。

测试用的 `MockEmbedder` 新增 `embed_batch()` 方法（委托给 `embed()` 返回相同向量），使所有现有 VectorStore 测试可适配 batch 模式。

## Acceptance criteria

- [ ] `add_sections_batch(sections)` 返回 `{"inserted": N, "skipped": N, "updated": N}`
- [ ] 建库后 `similarity_search()` 可检索到结果，父文档反查返回整节全文
- [ ] 重复入库同一内容 → skipped（哈希比对生效）
- [ ] 修改内容后重入 → updated（增量更新生效）
- [ ] 新增节 → inserted（新 ID 识别生效）
- [ ] 父文档 Collection 完整写入，`has_parent_collection()` 返回 True
- [ ] `_EmbedderAdapter` 保留不变，供 MCP query embedding 使用
- [ ] 所有现有 VectorStore 测试通过（MockEmbedder 加 `embed_batch` 后）
- [ ] `MockEmbedder.embed_batch()` 行为等同 `embed()`（返回相同向量）

## Blocked by

- #010 DashScopeEmbedder — batch embedding（需要 `embed_batch()` 接口）

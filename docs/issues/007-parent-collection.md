# #7 — 建库脚本：父文档 Collection

- **Type**: AFK
- **Label**: `done`
- **Blocked by**: None — can start immediately

## What to build

修改建库脚本的向量库写入逻辑，在已有的 chunk 索引（`rag_documents` Collection）之外，新增第二个 Chroma Collection `rag_parents`。父文档 Collection 以 `section_id` 为主键存储每个节的完整全文 + 元数据，不参与向量相似度索引。建库时两个 Collection 同步写入，确保数据一致。

端到端行为：
1. 用户执行 `rag-build -i 药典.md -d ./db`
2. Chroma 目录 `./db` 包含两个 Collection：`rag_documents`（chunk 向量索引）和 `rag_parents`（父文档全文存储）
3. 增量更新逻辑同步覆盖父文档 Collection（插入/跳过/更新）
4. 如果数据库中已有旧版单 Collection（无 `rag_parents`），本次建库自动补齐

## Acceptance criteria

- [ ] `vector_store.py` 新增 `_parent_collection_name = "rag_parents"` 常量
- [ ] `add_sections()` 中，写入 chunk 后同步写入父文档到 `rag_parents` Collection
- [ ] 父文档的 `page_content` 为整节完整全文（Section.content），`metadata` 包含 section_id、section_title、chapter、source_file
- [ ] 父文档 ID 格式：`{source_file}#{section_id}`（与 chunk ID 相同的前缀，便于关联）
- [ ] 增量更新逻辑：skip/insert/update 对两个 Collection 同时生效
- [ ] 已存储父文档的 Content 长度等于对应 Section 的原文长度
- [ ] 单元测试：建库后验证 `rag_parents` Collection 存在且节数与预期一致
- [ ] 单元测试：父文档 content 为非空的完整节全文
- [ ] 向后兼容：对已有单 Collection 的旧库，重新建库后两个 Collection 均存在

## Blocked by

None — can start immediately

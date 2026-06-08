# #8 — MCP 检索服务

- **Type**: AFK
- **Label**: `done`
- **Blocked by**: #7

## What to build

新建 MCP 检索服务模块，基于 FastMCP 框架暴露单个 Tool `search_rag(query, k=5)`。服务启动时从 Chroma 加载两个 Collection（chunk 索引 + 父文档存储），接收查询后完成语义搜索 → section_id 去重 → 父文档反查 → 返回整节全文 + 元数据。支持 stdio（默认）和 HTTP/SSE 两种传输方式。

端到端行为：
1. 用户执行 `python -m rag_builder.mcp_server --db ./db`
2. MCP 服务启动，连接 Chroma 的两个 Collection
3. AI 客户端调用 `search_rag("片剂的崩解时限要求", k=3)`
4. 返回 ≤3 条去重结果，每条包含完整节全文 + section_id + title + chapter + score

## Acceptance criteria

- [ ] 新建 MCP 服务模块，基于 FastMCP
- [ ] 暴露 Tool `search_rag(query: str, k: int = 5)`，返回去重的整节全文列表
- [ ] 检索流程：embed query → `rag_documents` 相似度搜索 top-20 → 提取 section_id 去重 → 从 `rag_parents` 按 ID 查父文档 → 取 top-k 返回
- [ ] 返回结果每个元素包含：section_id、section_title、chapter、content（完整节全文）、score
- [ ] 支持 `--transport stdio`（默认）和 `--transport sse --port 8080` 两种传输方式
- [ ] 执行 `python -m rag_builder.mcp_server --help` 显示参数说明
- [ ] 数据库无 `rag_parents` Collection 时启动报错，提示需重新建库
- [ ] 单元测试：用 mini .md 建库 → MCP 服务加载 → 调 `search_rag` → 验证返回格式
- [ ] 单元测试：验证返回的 content 长度 > chunk 长度（确实是完整节全文）
- [ ] 单元测试：验证同节多 chunk 命中时 section_id 去重生效
- [ ] 单元测试：空查询不抛异常
- [ ] README 更新：增加 MCP 服务使用说明和 Claude Code 配置示例

## Blocked by

- #7（需要父文档 Collection 存储整节全文）

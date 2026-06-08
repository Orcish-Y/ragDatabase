## Parent

[PRD: 阿里云 DashScope 在线 Embedding 替换本地模型](../PRD-dashscope-embedding.md)

## What to build

新建 `DashScopeEmbedder` 类，实现实时 embedding（`embed()` + `similarity()`）。调用阿里云 OpenAI 兼容端点 `/v1/embeddings`，内部按 10 条/请求自动分批，返回归一化 numpy 向量。

完成后，chunker 和 MCP query 即可通过实时 API 获取 embedding，不再依赖本地模型。

## Acceptance criteria

- [ ] `DashScopeEmbedder(model_name="text-embedding-v4")` 可通过 `.env` 的 `DASHSCOPE_API_KEY` 初始化
- [ ] `embed(texts)` 输入 25 条文本，内部拆成 3 批（10+10+5）发送，返回 `(25, 1024)` numpy 数组
- [ ] `embed([])` 返回空数组
- [ ] `similarity(a, b)` 返回两个归一化向量的余弦相似度（等同点积）
- [ ] `model_name` 属性返回 `"text-embedding-v4"`
- [ ] 未设置 `DASHSCOPE_API_KEY` 环境变量时，`embed()` 调用抛出明确错误
- [ ] 可通过 `DASHSCOPE_BASE_URL` 环境变量覆盖 API 端点
- [ ] 单元测试用 mock OpenAI client 覆盖分批逻辑和错误路径

## Blocked by

None — can start immediately.

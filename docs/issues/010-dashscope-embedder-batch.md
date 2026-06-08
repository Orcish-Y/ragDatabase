## Parent

[PRD: 阿里云 DashScope 在线 Embedding 替换本地模型](../PRD-dashscope-embedding.md)

## What to build

在 `DashScopeEmbedder` 上新增 `embed_batch()` 方法。调用 OpenAI Batch API（`/v1/embeddings` + batch endpoint），全自动完成：JSONL 构建 → 文件上传 → 创建 batch 任务 → 轮询等待完成 → 下载解析结果。

内置重试逻辑：3 次指数退避（1s→2s→4s），仅重试失败的 embedding 条目（按 `custom_id` 挑出）。3 次后仍有失败则打印失败 ID 列表并报错。

内部在终端输出进度信息（"Batch 任务已提交"、"等待中..."、"下载结果中"）。

## Acceptance criteria

- [ ] `embed_batch(texts)` 输入 N 条文本，完成全流程后返回 `(N, 1024)` numpy 数组
- [ ] JSONL 每行包含 `custom_id`（序号）、`method: POST`、`url: /v1/embeddings`、`body` 含 `model` 和 `input`
- [ ] 终端输出进度：提交 → 等待 → 下载，状态可读
- [ ] 模拟部分条目失败时：自动重试最多 3 次，间隔 1s→2s→4s
- [ ] 3 次重试后仍有失败：打印失败 `custom_id` 列表 + 抛出 `RuntimeError`
- [ ] 单元测试用 mock OpenAI client 覆盖：正常流程、部分失败重试成功、全部重试后仍失败
- [ ] mock client 需支持 `files.create`、`batches.create`、`batches.retrieve`、`files.content` 四个调用

## Blocked by

- #009 DashScopeEmbedder — 实时 embedding（同文件，需先完成基础结构和 client 初始化）

# #2 — 语义切分引擎

- **Type**: AFK
- **Label**: `ready-for-agent`
- **Blocked by**: #1

## What to build

实现语义切分：对每个 Section 内的句子计算相邻句 Embedding 余弦相似度，在语义转折点切分，追加 overlap。CLI verbose 模式可观察切分边界。

端到端行为：
1. 用户执行 `rag-build -i 药典.md --dry-run --verbose`
2. 对每个 Section，先打印节标题，再打印内部 Chunk 边界（"--- chunk 0 (5 sentences) ---"）
3. 仍然不写入 Chroma（纯诊断模式）

## Acceptance criteria

- [ ] `embedder.py` 封装 SentenceTransformer，默认加载 `paraphrase-multilingual-MiniLM-L12-v2`
- [ ] 嵌入模型可通过 `--model` 参数替换
- [ ] `chunker.py` 实现句分割：中文按 `。！？；`，英文按 `. ! ? ;`
- [ ] 计算全文档相邻句子对余弦相似度，统计 μ 和 σ
- [ ] 默认转折判定：相似度 < μ - 1.5σ 处切分
- [ ] `--threshold` 可设绝对阈值覆盖自适应行为
- [ ] 最小 chunk 3 句约束，最大 chunk 20 句约束
- [ ] overlap 默认 2 句，切分后追加，`--overlap` 可调
- [ ] `--min-chunk` / `--max-chunk` 可调
- [ ] 单元测试：转折点位置正确（语义差异大的段落交界处应切分）
- [ ] 单元测试：min 约束生效（无 < 3 句的 chunk）
- [ ] 单元测试：max 约束生效（无 > 20 句的 chunk）
- [ ] 单元测试：overlap 验证（chunk[i].tail == chunk[i+1].head）

## Blocked by

- #1（需要 Section 对象作为输入）

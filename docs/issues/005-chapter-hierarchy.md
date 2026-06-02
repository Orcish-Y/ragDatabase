# #5 — 章层级推导 + 映射

- **Type**: AFK
- **Label**: `done`
- **Blocked by**: #1

## What to build

为每个 Section 自动推导所属章（Chapter）。优先使用 Markdown `###` 标题自动推导章-节父子关系；若无 `###`，可通过 `--chapter-map` 传入 JSON 映射文件补充；两者都无则 chapter 字段为 null。

端到端行为：
1. 用户用含 `###`/`####` 两层标题的 md 建库 → 每个 Section.chapter 自动填充
2. 用户用药典（仅 `####`）+ `--chapter-map chapters.json` 建库 → chapter 从映射表注入
3. 用户用纯 `####` md 建库，不提供映射 → chapter 为 null（不报错）

## Acceptance criteria

- [ ] `document.py` 支持解析 `###` 标题，推导 `###` → `####` 的章-节父子关系
- [ ] CLI 新增 `--chapter-map` 参数，接受 JSON 文件路径
- [ ] JSON 格式：`{"编号前缀": "章名"}`，如 `{"01": "制剂通则", "04": "光谱法"}`
- [ ] 优先级：JSON 映射 > 自动推导 > null
- [ ] 映射表未覆盖的编号前缀 → 该节 chapter 为该前缀在 JSON 中的值（前缀匹配）
- [ ] 单元测试：含 `###` 的 md fixture → Section.chapter 正确
- [ ] 单元测试：仅 `####` 的 md → 所有 Section.chapter 为 null
- [ ] 单元测试：`--chapter-map` JSON → Section.chapter 正确覆盖
- [ ] 单元测试：JSON 映射优先级高于 `###` 自动推导

## Blocked by

- #1（需要 Section 对象，document.py 修改）

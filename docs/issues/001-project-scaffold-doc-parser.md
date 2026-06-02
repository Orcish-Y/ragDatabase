# #1 — 项目脚手架 + 文档解析

- **Type**: AFK
- **Label**: `ready-for-agent`
- **Blocked by**: None — can start immediately

## What to build

搭建 `rag_builder` Python 包的基础结构，并实现 Markdown 文档解析——将 `.md` 文件按 `####` 标题切分为 Section 对象。CLI 在 `--dry-run` 模式下打印解析结果。

端到端行为：
1. 用户执行 `rag-build -i 药典.md --dry-run`
2. 脚本打印出文档中检测到的所有节列表：编号、标题、行范围
3. `-h` 输出基本参数说明

## Acceptance criteria

- [ ] `pyproject.toml` 定义包名 `rag_builder`，依赖 langchain, chromadb, sentence-transformers
- [ ] 包目录结构 `src/rag_builder/` 就位，6 个模块文件（可为空壳，但 import 路径正确）
- [ ] `config.py` 定义所有默认参数常量（阈值、重叠、min/max chunk、模型名）
- [ ] `document.py` 解析 Markdown，按 `####` 边界切分，返回 Section 对象列表
- [ ] Section 对象包含：section_id（编号）、section_title（标题）、content（全文）
- [ ] CLI 支持 `-i`、`-h`、`--dry-run`
- [ ] `--dry-run` 输出表格：section_id | section_title | 行数
- [ ] `pip install -e .` 后 `rag-build -h` 可用
- [ ] 单元测试：多节 md fixture → Section 数量和 ID 正确
- [ ] 单元测试：非 md 后缀文件报错

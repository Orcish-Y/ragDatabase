## Parent

[PRD: 阿里云 DashScope 在线 Embedding 替换本地模型](../PRD-dashscope-embedding.md)

## What to build

更新 `cli.py`（`rag-build` entry point）：

- 用 `DashScopeEmbedder` 替换 `Embedder`
- 移除 `--hf-mirror` 参数
- 移除 `--model` 参数（固定为 `text-embedding-v4`，不需要暴露）
- 启动时 `load_dotenv()` 加载 `.env`，未找到 `DASHSCOPE_API_KEY` 则报错退出
- 建库流程改为调用 `VectorStore.add_sections_batch()`（不再用旧 `add_sections()`）
- 移除 `from .embedder import Embedder`，改为 `from .embedder import DashScopeEmbedder`
- `rag-build -i <input.md> -d <db_dir>` 不需要额外参数即可完成建库

## Acceptance criteria

- [ ] `rag-build -i <doc.md> -d <db_dir>` 执行全流程：解析 → 切分 → Batch embedding → 写 Chroma
- [ ] 未设置 `DASHSCOPE_API_KEY` 时退出并打印提示
- [ ] 不再支持 `--hf-mirror`（传入报错）
- [ ] 终端输出包含解析统计（节数）、Batch 提交/等待/下载进度、最终统计（chunks/插入/跳过/更新）
- [ ] 已有 `test_cli.py` 测试适配通过

## Blocked by

- #011 VectorStore — add_sections_batch（需要 batch 建库路径）

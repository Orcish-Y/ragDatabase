# #6 — CLI 收尾 + 全参数联调

- **Type**: AFK
- **Label**: `done`
- **Blocked by**: #3, #4, #5

## What to build

将所有可选参数接入 CLI，格式化输出统计信息，编写 README，端到端测试覆盖全流程。

端到端行为：
1. 用户执行 `rag-build -h` → 看到完整参数列表和简洁说明
2. 用户执行完整命令 → `rag-build -i 药典.md -d ./db --threshold 0.6 --overlap 3 --chapter-map chapters.json -m all-MiniLM-L6-v2`
3. 脚本输出清晰的统计报告：
   ```
   文档: 2020年药典四部
   节数: 312
   Chunk 数: 1847
   插入: 1847, 跳过: 0, 更新: 0
   模型: all-MiniLM-L6-v2
   数据库: ./db
   ```
4. 如果数据库中已有内容，第二次运行显示增量统计

## Acceptance criteria

- [ ] CLI 全部参数可正常接收和传递：`-i`, `-d`, `-t/--threshold`, `-o/--overlap`, `--min-chunk`, `--max-chunk`, `--chapter-map`, `-m/--model`
- [ ] `-h` 输出清晰的分组参数列表（必需 / 可选）
- [ ] 运行结束后的统计输出格式化（节数、chunk 数、插入/跳过/更新计数、模型名、数据库路径）
- [ ] 数据库路径不存在时自动创建目录
- [ ] 输入文件不存在时给出明确错误信息（非 traceback）
- [ ] README.md：安装说明、使用示例、参数说明
- [ ] 端到端测试：完整 mini .md → 建库 → Chroma 验证可检索
- [ ] 端到端测试：`-h` exit code 0，输出含所有参数
- [ ] 端到端测试：错误输入给出友好报错（无 traceback）

## Blocked by

- #3（基础 CLI + Chroma 写入）
- #4（增量统计输出）
- #5（chapter-map 参数）

# RAG Builder

通用 RAG 知识库建库工具 — 语义切分 + 父文档检索器，将 Markdown 文档构建为 Chroma 向量数据库。

## 安装

```bash
pip install -e .
```

## 使用

```bash
# 基本用法
rag-build -i 文档.md -d ./my_db

# 查看帮助
rag-build -h

# 完整参数
rag-build -i 文档.md -d ./my_db \
    --threshold 0.6 \
    --overlap 3 \
    --chapter-map chapters.json \
    -m paraphrase-multilingual-MiniLM-L12-v2
```

## 项目结构

```
src/rag_builder/  核心建库包
servers/          未来服务形态 (MCP / API / Tool)
scripts/          历史数据清洗脚本
data/             测试/示例数据
tests/            测试
docs/             文档 & ADR
```

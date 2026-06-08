## 领域术语

### 文档 (Document)
用户传入的 Markdown 文件，作为建库的源材料。一个文档包含若干**节**。

### 节 (Section)
Markdown 文档中以 `####` 标题开头、到下一个 `####` 标题之前为止的连续内容块。节是**检索召回的最小完整单元**——检索命中时返回整节全文，而非被切分后的碎片。

### 章 (Chapter)
可选层级。当 Markdown 文档中存在 `###` 标题时，其下属的所有 `####` 节归入该章。当文档只有 `####` 无 `###` 时，可通
过 `--chapter-map` 外部 JSON 映射来补充章分类。章作为元数据注入，不参与切分。

### 语义切分 (Semantic Chunking)
利用 Embedding 向量计算相邻句子间的余弦相似度，当相似度显著下降（低于文档统计均值减去可配标准差倍数）时标记为**语义转折点**，在转折点处切分。切分结果受**最小 chunk 大小**和**最大 chunk 大小**约束。

### Chunk
语义切分产生的文本片段。一个**节**可包含多个 Chunk。Chunk 是向量索引的基本单元（用于检索匹配），但**不直接返回给用户**。用户看到的始终是其所属的完整**节**。

### 父文档检索器 (Parent Document Retriever)
LangChain 提供的双存储模式：小 Chunk 存入向量索引（用于匹配），大 Chunk（即整节）作为父文档存储。检索时向量匹配返回的是小 Chunk，但最终输出自动替换为对应的父文档（整节）。

### 重叠 (Overlap)
语义切分后，相邻 Chunk 尾部额外多取 N 个句子（默认 2），使得相邻 Chunk 之间有内容交叉。重叠发生在切分之后，不影响语义边界判定。

### 增量更新 (Incremental Update)
向已有向量数据库添加新内容时的行为：每个 Chunk 有唯一 ID（`{文件名}#{节编号}`），通过比对内容哈希决定是跳过、插入还是更新。同 ID 不同哈希 → 删旧插新；同 ID 同哈希 → 跳过；新 ID → 插入。

### 嵌入模型 (Embedding Model)
默认 `paraphrase-multilingual-MiniLM-L12-v2`（SentenceTransformer），支持中英文。可通过 `--model` 参数覆盖。

### 向量数据库 (Vector Store)
Chroma，使用余弦相似度（Cosine Similarity）作为距离度量，HNSW 算法做近似最近邻搜索。
建库时写入两个 Collection：`rag_documents`（chunk embedding 索引）和 `rag_parents`（父文档全文）。

### 父文档 Collection (Parent Collection)
Chroma 中与 chunk 索引并列的第二个 Collection。存储节全文，以 `section_id` 为主键，不参与向量相似度搜索。检索流程：chunk 命中 → 获取 `section_id` → 从 `rag_parents` 查询父文档 → 去重后返回整节全文。

### MCP 服务 (MCP Server)
基于 FastMCP 的检索服务。暴露单个 Tool `search_rag(query, k)`，输入自然语言查询，返回 top-k 个节的完整全文及元数据。通过 stdio（默认）或 HTTP/SSE 与 AI 客户端通信。启动参数仅需 `--db`（Chroma 数据库目录）。

### 重排 (Rerank)
可选的后处理步骤。Embedding 粗筛 top-N 候选后，用 Cross-Encoder 模型对每个 (query, doc) 对精确打分重新排序。MVP 不启用，预留在 `--rerank-model` 参数中。

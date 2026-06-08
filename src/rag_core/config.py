"""默认参数常量。"""
from __future__ import annotations

# 嵌入模型
DEFAULT_MODEL_NAME = "text-embedding-v4"

# 阿里云 DashScope
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBED_BATCH_SIZE = 10
BATCH_RETRY_MAX = 3
BATCH_RETRY_BASE_DELAY = 1.0

# Chroma
DEFAULT_COLLECTION_NAME = "rag_documents"
PARENT_COLLECTION_SUFFIX = "_parents"
DEFAULT_DISTANCE_METRIC = "cosine"

# 语义切分
DEFAULT_OVERLAP_SENTENCES = 2
DEFAULT_MIN_CHUNK_SENTENCES = 3
DEFAULT_MAX_CHUNK_SENTENCES = 20
DEFAULT_THRESHOLD_STD_DEV = 1.5

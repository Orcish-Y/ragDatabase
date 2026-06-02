"""默认参数常量。"""

# 嵌入模型
DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# 语义切分
DEFAULT_OVERLAP_SENTENCES = 2
DEFAULT_MIN_CHUNK_SENTENCES = 3
DEFAULT_MAX_CHUNK_SENTENCES = 20
DEFAULT_THRESHOLD_STD_DEV = 1.5  # μ - N*σ 作为自适应阈值

# 句子边界正则
SENTENCE_BOUNDARY_PATTERN_ZH = r"(?<=[。！？；])\s*"
SENTENCE_BOUNDARY_PATTERN_EN = r"(?<=[.!?;])\s+"

# Chroma
DEFAULT_COLLECTION_NAME = "rag_documents"
DEFAULT_DISTANCE_METRIC = "cosine"

# 元数据字段
METADATA_FIELDS = [
    "section_id",
    "section_title",
    "chapter",
    "source_file",
    "content_hash",
    "chunk_index",
]

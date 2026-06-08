"""CLI 入口 — 通用 Markdown → Chroma 向量库建库脚本。"""

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import (
    DEFAULT_MAX_CHUNK_SENTENCES,
    DEFAULT_MIN_CHUNK_SENTENCES,
    DEFAULT_MODEL_NAME,
    DEFAULT_OVERLAP_SENTENCES,
    DEFAULT_THRESHOLD_STD_DEV,
)
from .document import load_chapter_map, parse_markdown
from .embedder import DashScopeEmbedder


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-build",
        description="将 Markdown 文档构建为 Chroma 向量知识库。语义切分 + 父文档检索器。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  rag-build -i 药典.md -d ./my_db\n  rag-build -i 药典.md -d ./my_db --chapter-map chapters.json",
    )
    # 必需
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="目标 Markdown 文件路径",
    )
    parser.add_argument(
        "-d", "--db",
        required=True,
        help="Chroma 向量数据库存放目录",
    )

    # 可选
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=None,
        help=f"语义切分相似度阈值 (默认: 自适应, μ - {DEFAULT_THRESHOLD_STD_DEV}σ)",
    )
    parser.add_argument(
        "-o", "--overlap",
        type=int,
        default=DEFAULT_OVERLAP_SENTENCES,
        help=f"相邻 chunk 重叠句子数 (默认: {DEFAULT_OVERLAP_SENTENCES})",
    )
    parser.add_argument(
        "--min-chunk",
        type=int,
        default=DEFAULT_MIN_CHUNK_SENTENCES,
        help=f"最小 chunk 句子数 (默认: {DEFAULT_MIN_CHUNK_SENTENCES})",
    )
    parser.add_argument(
        "--max-chunk",
        type=int,
        default=DEFAULT_MAX_CHUNK_SENTENCES,
        help=f"最大 chunk 句子数 (默认: {DEFAULT_MAX_CHUNK_SENTENCES})",
    )
    parser.add_argument(
        "--chapter-map",
        type=str,
        default=None,
        help="章映射 JSON 文件路径",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="所有节的分类标签 (如 pharmaceutical)，写入 metadata 供 where 过滤",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"嵌入模型名称 (默认: {DEFAULT_MODEL_NAME})",
    )
    return parser


def build(args: argparse.Namespace) -> dict[str, int]:
    """执行建库流程，返回统计字典。可独立测试。"""
    input_path = Path(args.input)
    db_path = Path(args.db)

    # 输入文件校验
    if not input_path.exists():
        raise FileNotFoundError(f"文件不存在: {input_path}")
    if input_path.suffix.lower() != ".md":
        raise ValueError(f"仅支持 .md 文件: {input_path}")

    # 加载章映射
    chapter_map = None
    if args.chapter_map:
        chapter_map_path = Path(args.chapter_map)
        if not chapter_map_path.exists():
            raise FileNotFoundError(f"章映射文件不存在: {chapter_map_path}")
        chapter_map = load_chapter_map(chapter_map_path)

    # 1. 文档解析
    print(f"解析文档: {input_path}")
    sections = parse_markdown(input_path, chapter_map=chapter_map)
    if args.category:
        for s in sections:
            s.category = args.category
    print(f"  检测到 {len(sections)} 个节")

    # 2. 加载嵌入模型
    print(f"加载嵌入模型: {args.model}")
    embedder = DashScopeEmbedder(args.model)

    # 3. 向量库构建
    print(f"构建向量库: {db_path}")
    db_path.mkdir(parents=True, exist_ok=True)

    from .vector_store import VectorStore

    store = VectorStore(persist_dir=db_path, embedder=embedder)

    stats = store.add_sections_batch(
        sections,
        threshold=args.threshold,
        overlap=args.overlap,
        min_chunk=args.min_chunk,
        max_chunk=args.max_chunk,
    )

    # 4. 输出统计
    total_chunks = stats["inserted"] + stats["updated"]
    print()
    print("=" * 50)
    print(f"文档:   {input_path.stem}")
    print(f"节数:   {len(sections)}")
    print(f"Chunks: {total_chunks}")
    print(f"插入:   {stats['inserted']}")
    print(f"跳过:   {stats['skipped']}")
    print(f"更新:   {stats['updated']}")
    print(f"模型:   {args.model}")
    print(f"数据库: {db_path.absolute()}")
    print("=" * 50)

    return stats


def main() -> None:
    """CLI 入口 — 解析参数 → 建库 → 错误处理。"""
    from dotenv import load_dotenv
    load_dotenv()
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        build(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

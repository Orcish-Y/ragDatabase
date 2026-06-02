"""CLI 入口 — 只管建库。"""

import argparse
import sys

from .config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_OVERLAP_SENTENCES,
    DEFAULT_MIN_CHUNK_SENTENCES,
    DEFAULT_MAX_CHUNK_SENTENCES,
    DEFAULT_THRESHOLD_STD_DEV,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-build",
        description="将 Markdown 文档构建为 Chroma 向量知识库。",
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
        "-m", "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"嵌入模型名称 (默认: {DEFAULT_MODEL_NAME})",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # TODO: 接入各模块 — 文档解析 → 语义切分 → Chroma 入库
    print(f"rag-builder v0.1.0")
    print(f"  输入: {args.input}")
    print(f"  数据库: {args.db}")
    print(f"  模型: {args.model}")
    print("(实现待完成)")


if __name__ == "__main__":
    main()

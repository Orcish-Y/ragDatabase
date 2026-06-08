"""MCP 检索服务 — 基于 FastMCP，暴露 search_rag Tool。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from .config import DEFAULT_COLLECTION_NAME, DEFAULT_MODEL_NAME
from .embedder import DashScopeEmbedder
from .vector_store import VectorStore

load_dotenv()

# 全局变量，在 main() 中初始化
_store: VectorStore | None = None


def create_server() -> "FastMCP":
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("rag-knowledge-base")

    @mcp.tool()
    def search_rag(query: str, k: int = 5) -> list[dict]:
        """在 RAG 知识库中搜索相关章节。

        Args:
            query: 自然语言查询。
            k: 返回结果数量，默认 5。

        Returns:
            包含 section_id、section_title、chapter、content、score 的列表。
        """
        if _store is None:
            return [{"error": "服务未初始化，数据库未加载"}]

        # 1. 在 chunk Collection 中做语义搜索，取 top-(k*4) 以留出去重余量
        results = _store.vectorstore.similarity_search_with_relevance_scores(
            query, k=k * 4
        )

        if not results:
            return []

        # 2. 按 section_id 去重
        seen: set[str] = set()
        scored_sections: list[tuple[str, float]] = []
        for doc, score in results:
            sid = doc.metadata.get("section_id", "")
            if sid and sid not in seen:
                seen.add(sid)
                scored_sections.append((sid, score))

        # 3. 从父文档 Collection 反查整节全文
        output: list[dict] = []
        for sid, score in scored_sections[:k]:
            parent = _store.get_parent_by_id(sid)
            if parent is None:
                continue
            output.append({
                "section_id": parent.metadata.get("section_id", ""),
                "section_title": parent.metadata.get("section_title", ""),
                "chapter": parent.metadata.get("chapter", ""),
                "category": parent.metadata.get("category", ""),
                "content": parent.page_content,
                "score": round(score, 4),
            })

        return output

    return mcp


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-mcp",
        description="RAG 知识库 MCP 检索服务。",
    )
    parser.add_argument(
        "--db",
        required=True,
        help="Chroma 向量数据库存放目录",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"嵌入模型名称或路径 (默认: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "sse"],
        default="stdio",
        help="传输方式 (默认: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="SSE 模式端口 (默认: 8080)",
    )
    return parser


def main() -> None:
    global _store

    parser = build_arg_parser()
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"错误: 数据库目录不存在 — {db_path}", file=sys.stderr)
        sys.exit(1)

    # 加载嵌入模型
    embedder = DashScopeEmbedder(args.model)

    # 加载向量库
    _store = VectorStore(persist_dir=db_path, embedder=embedder)

    if not _store.has_parent_collection():
        print(
            "错误: 数据库缺少父文档 Collection。请用最新版 rag-build 重新建库。",
            file=sys.stderr,
        )
        sys.exit(1)

    # 启动 MCP 服务
    server = create_server()

    if args.transport == "sse":
        import uvicorn
        app = server.sse_app()
        uvicorn.run(app, host="127.0.0.1", port=args.port)
    else:
        server.run(transport="stdio")


if __name__ == "__main__":
    main()

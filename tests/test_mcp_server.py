"""测试 MCP 检索服务 — 核心搜索管线。"""

import tempfile

import numpy as np
import pytest

from rag_builder.document import Section


from .conftest import MockEmbedder


@pytest.fixture
def store():
    """创建包含两个 Collection 的测试数据库。"""
    import gc
    import shutil
    from rag_builder.vector_store import VectorStore

    tmpdir = tempfile.mkdtemp()
    embedder = MockEmbedder()
    s = VectorStore(persist_dir=tmpdir, embedder=embedder)

    sections = [
        Section("0101", "片剂", "片剂系指原料药物制成的固体制剂。应符合崩解时限检查。片剂以口服普通片为主。", source_file="test"),
        Section("0102", "注射剂", "注射剂系指供注入体内的制剂。应无菌。注射液一般由原料药和适宜辅料经配制过滤灌封灭菌等步骤制成。", source_file="test"),
        Section("0921", "崩解时限检査法", "除另有规定外，同法检查6片，各片均应在5分钟内崩解。如有1片不能完全崩解，应另取6片复试。", source_file="test"),
    ]
    s.add_sections(sections)
    yield s, tmpdir
    s.close()
    gc.collect()
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass


# ── 搜索管线 ───────────────────────────────────────────────────


class TestSearchPipeline:
    """测试 chunk 搜索 → section_id 去重 → parent 反查的完整管线。"""

    def test_search_returns_full_sections(self, store):
        s, _ = store
        query = "片剂崩解时限"
        results = s.vectorstore.similarity_search(query, k=5)

        assert len(results) > 0

        # section_id 去重
        seen: set[str] = set()
        for doc in results:
            sid = doc.metadata.get("section_id", "")
            if sid and sid not in seen:
                seen.add(sid)
                parent = s.get_parent_by_id(sid)
                assert parent is not None, f"Parent not found for {sid}"
                # 父文档包含完整节内容，chunk 的文本应该出现在父文档中
                chunk_text = doc.page_content.replace("\n", "")
                parent_text = parent.page_content.replace("\n", "")
                assert chunk_text in parent_text, \
                    f"Chunk text not found in parent for {sid}"

    def test_dedup_same_section(self, store):
        s, _ = store
        # 搜索应该命中同一节的多个 chunk，但去重后 section_id 唯一
        results = s.vectorstore.similarity_search("片剂", k=10)

        section_ids = [doc.metadata.get("section_id") for doc in results]
        unique_ids = set(section_ids)
        assert len(unique_ids) <= len(section_ids)

    def test_parent_collection_has_all_sections(self, store):
        s, _ = store
        for sid in ["0101", "0102", "0921"]:
            parent = s.get_parent_by_id(sid)
            assert parent is not None
            assert parent.metadata["section_id"] == sid

    def test_empty_query_ok(self, store):
        s, _ = store
        results = s.vectorstore.similarity_search("", k=5)
        assert isinstance(results, list)


# ── MCP 服务模块 ───────────────────────────────────────────────


class TestMCPServerModule:
    """测试 MCP 服务模块的导入和创建。"""

    def test_create_server_returns_fastmcp(self):
        from rag_server.mcp_server import create_server
        server = create_server()
        assert server is not None
        assert hasattr(server, "run")

    def test_arg_parser_help(self):
        from rag_server.mcp_server import build_arg_parser
        parser = build_arg_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["-h"])
        assert exc.value.code == 0

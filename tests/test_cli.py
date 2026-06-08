"""测试 CLI 入口。"""

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from rag_builder.cli import build, build_arg_parser
from .conftest import MockEmbedder


def test_help_output():
    """-h 输出包含参数说明且 exit 0。"""
    parser = build_arg_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["-h"])
    assert exc.value.code == 0


def test_missing_input():
    """缺 -i 时报错。"""
    parser = build_arg_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["-d", "/tmp/db"])
    assert exc.value.code != 0


def test_file_not_found():
    """文件不存在时抛出 FileNotFoundError。"""
    args = argparse.Namespace(
        input="/nonexistent.md", db="/tmp/db",
        threshold=None, overlap=2, min_chunk=3, max_chunk=20,
        chapter_map=None, category=None, model="mock",
    )
    with pytest.raises(FileNotFoundError, match="不存在"):
        build(args)


def test_not_md():
    """非 .md 文件抛出 ValueError。"""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"test")
        path = f.name
    try:
        args = argparse.Namespace(
            input=path, db="/tmp/db",
            threshold=None, overlap=2, min_chunk=3, max_chunk=20,
            chapter_map=None, category=None, model="mock",
        )
        with pytest.raises(ValueError, match="仅支持 .md"):
            build(args)
    finally:
        import os
        os.unlink(path)


def test_end_to_end(tmp_path: Path):
    """完整端到端：mini .md → Chroma DB → 可检索。"""
    md = tmp_path / "test.md"
    md.write_text(
        "#### 0101 片剂\n片剂系指原料药物制成的固体制剂。\n#### 0102 注射剂\n注射剂系指供注入体内的制剂。\n",
        encoding="utf-8",
    )
    db = tmp_path / "chroma_db"

    args = argparse.Namespace(
        input=str(md), db=str(db),
        threshold=None, overlap=2, min_chunk=3, max_chunk=20,
        chapter_map=None, category=None, model="mock",
    )

    mock_embedder = MockEmbedder()
    with patch("rag_builder.cli.DashScopeEmbedder", return_value=mock_embedder):
        stats = build(args)

    assert stats["inserted"] == 2
    assert db.exists()
    assert any(db.iterdir()), "Chroma DB 目录为空"

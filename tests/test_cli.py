"""测试 CLI 入口。"""

import io
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from rag_builder.cli import build_arg_parser, main


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


def test_file_not_found(capsys):
    """文件不存在时报错。"""
    test_args = ["-i", "/nonexistent.md", "-d", "/tmp/db"]
    with patch.object(sys, "argv", ["rag-build"] + test_args):
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "不存在" in captured.err


def test_end_to_end(tmp_path: Path, monkeypatch):
    """完整端到端：mini .md → Chroma DB → 可检索。"""
    md = tmp_path / "test.md"
    md.write_text(
        "#### 0101 片剂\n片剂系指原料药物制成的固体制剂。\n#### 0102 注射剂\n注射剂系指供注入体内的制剂。\n",
        encoding="utf-8",
    )
    db = tmp_path / "chroma_db"

    test_args = [
        "-i", str(md),
        "-d", str(db),
    ]
    with patch.object(sys, "argv", ["rag-build"] + test_args):
        main()

    # 验证 Chroma 目录存在
    assert db.exists()
    assert any(db.iterdir()), "Chroma DB 目录为空"

"""测试文档解析模块。"""

import json
import tempfile
from pathlib import Path

import pytest

from rag_builder.document import (
    Section,
    load_chapter_map,
    parse_markdown,
    resolve_chapter,
)


def make_md(content: str, tmp_path: Path) -> Path:
    """创建临时 Markdown 文件。"""
    p = tmp_path / "test.md"
    p.write_text(content, encoding="utf-8")
    return p


# ── parse_markdown ──────────────────────────────────────────────


class TestParseMarkdown:
    def test_basic_sections(self, tmp_path):
        md = make_md(
            "前言文字\n\n#### 0101 片剂\n片剂内容第一行。\n片剂内容第二行。\n\n#### 0102 注射剂\n注射剂内容。\n",
            tmp_path,
        )
        sections = parse_markdown(md)
        assert len(sections) == 2
        assert sections[0].section_id == "0101"
        assert sections[0].section_title == "片剂"
        assert "片剂内容第一行" in sections[0].content
        assert sections[1].section_id == "0102"
        assert sections[1].section_title == "注射剂"

    def test_source_file_name(self, tmp_path):
        md = tmp_path / "my_doc.md"
        md.write_text("#### 0001 测试\n内容。\n", encoding="utf-8")
        sections = parse_markdown(md)
        assert sections[0].source_file == "my_doc"

    def test_content_hash(self, tmp_path):
        md = make_md("#### 0001 测试\n内容。\n", tmp_path)
        s1 = parse_markdown(md)[0]
        s2 = parse_markdown(md)[0]
        assert s1.content_hash == s2.content_hash
        assert len(s1.content_hash) == 16

    def test_not_md_raises(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_text("#### 001 测试\n", encoding="utf-8")
        with pytest.raises(ValueError, match="仅支持 .md"):
            parse_markdown(p)

    def test_no_h4_raises(self, tmp_path):
        md = make_md("没有标题\n只有文字\n", tmp_path)
        with pytest.raises(ValueError, match="未找到"):
            parse_markdown(md)

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_markdown(tmp_path / "nope.md")

    def test_skip_content_before_first_h4(self, tmp_path):
        md = make_md("前言\n版权声明\n#### 0001 第一节\n内容。\n", tmp_path)
        sections = parse_markdown(md)
        assert len(sections) == 1
        assert "前言" not in sections[0].content
        assert "版权声明" not in sections[0].content

    def test_empty_content_section(self, tmp_path):
        md = make_md("#### 0001 空节\n#### 0002 下一节\n有内容。\n", tmp_path)
        sections = parse_markdown(md)
        assert len(sections) == 2
        assert sections[0].content == ""


# ── Chapter detection ──────────────────────────────────────────


class TestChapterDetection:
    def test_h3_chapter_inheritance(self, tmp_path):
        md = make_md(
            "### 制剂通则\n#### 0101 片剂\n内容。\n#### 0102 注射剂\n内容。\n",
            tmp_path,
        )
        sections = parse_markdown(md)
        assert sections[0].chapter == "制剂通则"
        assert sections[1].chapter == "制剂通则"

    def test_h3_change(self, tmp_path):
        md = make_md(
            "### 第一章\n#### 0001 A\nx\n### 第二章\n#### 0002 B\ny\n",
            tmp_path,
        )
        sections = parse_markdown(md)
        assert sections[0].chapter == "第一章"
        assert sections[1].chapter == "第二章"

    def test_no_h3_gives_null(self, tmp_path):
        md = make_md("#### 0001 节\n内容。\n", tmp_path)
        sections = parse_markdown(md)
        assert sections[0].chapter is None


# ── Chapter map ────────────────────────────────────────────────


class TestChapterMap:
    def test_json_map_override(self, tmp_path):
        json_path = tmp_path / "chapters.json"
        json_path.write_text(
            json.dumps({"01": "制剂通则", "04": "光谱法"}), encoding="utf-8"
        )
        chapter_map = load_chapter_map(json_path)
        assert chapter_map == {"01": "制剂通则", "04": "光谱法"}

    def test_prefix_match(self, tmp_path):
        json_path = tmp_path / "chapters.json"
        json_path.write_text(
            json.dumps({"01": "制剂通则"}), encoding="utf-8"
        )
        chapter_map = load_chapter_map(json_path)
        md = make_md("#### 0101 片剂\nx\n", tmp_path)
        sections = parse_markdown(md, chapter_map=chapter_map)
        assert sections[0].chapter == "制剂通则"

    def test_json_takes_priority_over_h3(self, tmp_path):
        json_path = tmp_path / "chapters.json"
        json_path.write_text(
            json.dumps({"01": "JSON章名"}), encoding="utf-8"
        )
        chapter_map = load_chapter_map(json_path)
        md = make_md("### H3章名\n#### 0101 节\nx\n", tmp_path)
        sections = parse_markdown(md, chapter_map=chapter_map)
        assert sections[0].chapter == "JSON章名"

    def test_longest_prefix_match(self, tmp_path):
        json_path = tmp_path / "chapters.json"
        json_path.write_text(
            json.dumps({"01": "粗", "010": "细"}), encoding="utf-8"
        )
        chapter_map = load_chapter_map(json_path)
        md = make_md("#### 0101 节\nx\n", tmp_path)
        sections = parse_markdown(md, chapter_map=chapter_map)
        assert sections[0].chapter == "细"


# ── resolve_chapter unit ───────────────────────────────────────


class TestResolveChapter:
    def test_none_all(self):
        assert resolve_chapter("0101", None, None) is None

    def test_h3_fallback(self):
        assert resolve_chapter("0101", None, "章") == "章"

    def test_map_wins(self):
        assert resolve_chapter("0101", {"01": "映射章"}, "H3章") == "映射章"

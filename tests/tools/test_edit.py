"""Tests for the targeted string replacement edit tool.

Ported from OpenCode's 9-strategy progressive fallback algorithm.
Each test exercises a specific replacement strategy or error case.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.tools.edit import (
    EditError,
    _context_aware_replace,
    _indentation_flexible_replace,
    _trimmed_boundary_replace,
    edit_file,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    """Create a sample Python file for editing."""
    f = tmp_path / "sample.py"
    f.write_text(
        'def hello():\n    print("Hello, world!")\n\ndef goodbye():\n    print("Goodbye!")\n'
    )
    return f


class TestExactMatch:
    """Strategy 1: SimpleReplacer — exact string match."""

    def test_replaces_exact_match(self, sample_file: Path):
        """Exact string match swaps the substring and reports the `exact` strategy."""
        result = edit_file(sample_file, 'print("Hello, world!")', 'print("Hi!")')
        assert 'print("Hi!")' in sample_file.read_text()
        assert result.strategy == "exact"

    def test_preserves_surrounding_content(self, sample_file: Path):
        """An exact-match replacement leaves untouched lines unchanged."""
        edit_file(sample_file, 'print("Hello, world!")', 'print("Hi!")')
        content = sample_file.read_text()
        assert "def hello():" in content
        assert "def goodbye():" in content


class TestWhitespaceFlexible:
    """Strategies 2-7: various whitespace flexibility."""

    def test_line_trimmed_match(self, sample_file: Path):
        """Strategy 2: matches when old_string has trailing spaces that file doesn't."""
        result = edit_file(
            sample_file,
            'print("Hello, world!")  ',  # trailing spaces not in file
            'print("Hi!")',
        )
        assert 'print("Hi!")' in sample_file.read_text()
        assert result.strategy != "exact"

    def test_indentation_flexible_match(self, tmp_path: Path):
        """Strategy 5: matches when indentation differs."""
        f = tmp_path / "indented.py"
        f.write_text("    def foo():\n        return 1\n")
        edit_file(
            f,
            "def foo():\n    return 1",  # no leading indent
            "def foo():\n    return 2",
        )
        content = f.read_text()
        assert "return 2" in content


class TestFuzzyStrategyDoesNotCorrupt:
    """Fuzzy strategies splice cleanly.

    No gluing, no newline mangling, no replacing embedded tokens, and no
    deleting unrelated blocks.
    """

    def test_line_trimmed_does_not_glue_onto_next_line(self, tmp_path: Path) -> None:
        """Trailing-space lines force line_trimmed; the replacement keeps line breaks."""
        f = tmp_path / "t.py"
        f.write_text("a = 1 \nb = 2 \nc = 3\n")  # trailing spaces defeat exact match
        edit_file(f, "a = 1\nb = 2", "a = 1\nb = 99")
        assert f.read_text() == "a = 1\nb = 99\nc = 3\n"

    def test_block_anchor_does_not_glue_onto_next_line(self, tmp_path: Path) -> None:
        """A 2-line anchor match splices `new` without gluing onto the following line."""
        f = tmp_path / "t.py"
        # Indent defeats exact/line_trimmed; block_anchor matches the anchors.
        f.write_text("        first = 1\n        second = 2\nrest\n")
        edit_file(f, "first = 1\nsecond = 2", "first = 10\nsecond = 20")
        out = f.read_text()
        assert "second = 20rest" not in out  # not glued
        assert out.endswith("rest\n")
        assert "second = 20\n" in out

    def test_indentation_flexible_reindents_without_mangling(self) -> None:
        """The indentation_flexible strategy re-applies indent without doubling newlines.

        Tested directly: earlier strategies (block_anchor) pre-empt it via edit_file.
        """
        content = "        first = 1\n        second = 2\nrest\n"
        result = _indentation_flexible_replace(
            content, "first = 1\nsecond = 2", "first = 10\nsecond = 20",
        )
        assert result == "        first = 10\n        second = 20\nrest\n"

    def test_trimmed_boundary_rejects_token_embedded_match(self) -> None:
        """'foo' inside 'myfoo'/'foobar' must NOT be replaced (not a standalone block)."""
        assert _trimmed_boundary_replace("xmyfoo\n", "  foo  ", "BAR") is None
        assert _trimmed_boundary_replace("foobar\n", "  foo  ", "BAR") is None

    def test_trimmed_boundary_allows_whitespace_surrounded_and_keeps_new(self) -> None:
        """An indented standalone match is replaced, and new's whitespace is preserved."""
        result = _trimmed_boundary_replace("    foo\n", "foo", "  bar  ")
        assert result == "      bar  \n"  # 4-space indent kept, new not stripped, \n intact

    def test_context_aware_rejects_dissimilar_span(self) -> None:
        """A 2-line old must not delete a 5-line span of unrelated content."""
        content = "def f():\n    a = 1\n    b = 2\n    c = 3\n    return x\n"
        result = _context_aware_replace(content, "def f():\n    return x", "def g(): pass")
        assert result is None  # span doesn't resemble old -> refuse rather than nuke a,b,c

    def test_context_aware_accepts_indented_similar_block(self) -> None:
        """Indentation drift between de-indented `old` and the file must NOT reject."""
        content = "    def f():\n        x = 1\n        return x\n"
        result = _context_aware_replace(
            content, "def f():\n    x = 1\n    return x", "def g(): ...",
        )
        assert result is not None  # the span resembles old once indent is normalized

    def test_crlf_file_keeps_uniform_endings(self, tmp_path: Path) -> None:
        """Editing a CRLF file with LF new_string leaves uniform CRLF, not mixed."""
        f = tmp_path / "crlf.py"
        f.write_bytes(b"        alpha = 1\r\n        beta = 2\r\n        gamma = 3\r\nend\r\n")
        edit_file(f, "alpha = 1\nbeta = 2\ngamma = 3", "alpha = 10\nbeta = 20\ngamma = 30")
        data = f.read_bytes()
        assert b"\r\n" in data
        assert b"\n" not in data.replace(b"\r\n", b"")  # no bare LF survives

    def test_edit_block_at_eof_preserves_trailing_newline(self, tmp_path: Path) -> None:
        """Replacing the last block of a newline-terminated file keeps the trailing newline."""
        f = tmp_path / "t.py"
        f.write_text("    first = 1\n    second = 2\n")  # indent forces block_anchor; EOF block
        edit_file(f, "first = 1\nsecond = 2", "first = 10\nsecond = 20")
        out = f.read_text()
        assert out.endswith("\n")  # trailing newline NOT dropped at EOF
        assert "second = 20" in out

    def test_indentation_flexible_preserves_dedent_in_new_block(self) -> None:
        """A new line less-indented than the first keeps its relative offset (not flattened)."""
        content = "        target\nrest\n"  # 8-space indent
        result = _indentation_flexible_replace(content, "target", "    nested\ntop")
        # base indent = min("    ", "") = ""; original indent = 8 spaces
        assert result == "            nested\n        top\nrest\n"


class TestMultipleMatches:
    """Error handling when old_string matches multiple locations."""

    def test_multiple_matches_without_replace_all_raises(self, tmp_path: Path):
        """Ambiguous matches raise `EditError` rather than picking one silently."""
        f = tmp_path / "dupes.py"
        f.write_text("x = 1\nx = 1\nx = 1\n")
        with pytest.raises(EditError, match="multiple"):
            edit_file(f, "x = 1", "x = 2")

    def test_replace_all_replaces_all_occurrences(self, tmp_path: Path):
        """`replace_all=True` swaps every occurrence in the file."""
        f = tmp_path / "dupes.py"
        f.write_text("x = 1\nx = 1\nx = 1\n")
        edit_file(f, "x = 1", "x = 2", replace_all=True)
        content = f.read_text()
        assert content.count("x = 2") == 3
        assert content.count("x = 1") == 0


class TestNoMatch:
    """Error handling when old_string is not found."""

    def test_no_match_raises(self, sample_file: Path):
        """All nine strategies failing surfaces an `EditError` to the caller."""
        with pytest.raises(EditError, match="not find"):
            edit_file(sample_file, "this string does not exist", "replacement")


class TestEdgeCases:
    """Edge cases and validation."""

    def test_old_equals_new_raises(self, sample_file: Path):
        """Identical `old_string` and `new_string` are rejected — no-op edits aren't allowed."""
        with pytest.raises(EditError, match="must be different"):
            edit_file(sample_file, 'print("Hello, world!")', 'print("Hello, world!")')

    def test_empty_old_string_raises(self, sample_file: Path):
        """Empty `old_string` is rejected — would match an unbounded number of positions."""
        with pytest.raises(EditError, match="empty"):
            edit_file(sample_file, "", "something")

    def test_file_not_found_raises(self, tmp_path: Path):
        """Editing a non-existent file raises rather than creating one silently."""
        with pytest.raises(EditError, match="not found"):
            edit_file(tmp_path / "nonexistent.py", "old", "new")

    def test_multiline_replacement(self, tmp_path: Path):
        """Multi-line `old_string` replaces the matching block intact."""
        f = tmp_path / "multi.py"
        f.write_text("def foo():\n    x = 1\n    return x\n")
        edit_file(f, "    x = 1\n    return x", "    x = 2\n    y = 3\n    return x + y")
        content = f.read_text()
        assert "x = 2" in content
        assert "y = 3" in content


class TestReturnValue:
    """Edit result contains useful metadata."""

    def test_result_has_before_and_after(self, sample_file: Path):
        """Edit result carries both the original and updated content for diff rendering."""
        result = edit_file(sample_file, 'print("Hello, world!")', 'print("Hi!")')
        assert 'print("Hello, world!")' in result.before
        assert 'print("Hi!")' in result.after

    def test_result_has_file_path(self, sample_file: Path):
        """Edit result carries the path of the file that was modified."""
        result = edit_file(sample_file, 'print("Hello, world!")', 'print("Hi!")')
        assert result.file_path == sample_file


class TestAtomicWritePermissions:
    """Edit preserves regular permissions but strips elevated bits.

    `_atomic_write` chmods the new file to match the source's mode. We
    intentionally strip setuid (S_ISUID), setgid (S_ISGID), and sticky
    (S_ISVTX) so editing a privileged binary doesn't silently propagate
    those bits onto the replaced inode.
    """

    def test_preserves_regular_permission_bits(self, tmp_path: Path):
        """A 0o600 source ends as a 0o600 destination — no umask downgrade."""
        f = tmp_path / "private.txt"
        f.write_text("secret\n")
        f.chmod(0o600)
        edit_file(f, "secret", "rotated")
        assert f.stat().st_mode & 0o777 == 0o600

    def test_strips_setuid_setgid_sticky(self, tmp_path: Path):
        """Whatever elevated bits the filesystem accepts are stripped after edit."""
        f = tmp_path / "elevated.txt"
        f.write_text("data\n")
        # Request rwxr-xr-x + all three special bits. Some filesystems silently
        # drop setgid/sticky on a non-root chmod, so assert only on the bits that
        # actually stuck — skip if none did (nothing to prove on this fs).
        f.chmod(0o7755)
        applied_elevated = f.stat().st_mode & 0o7000
        if applied_elevated == 0:
            pytest.skip("filesystem does not retain setuid/setgid/sticky for this user")
        edit_file(f, "data", "fresh")
        # Regular bits preserved, every elevated bit that was set is now stripped.
        assert f.stat().st_mode & 0o777 == 0o755
        assert f.stat().st_mode & 0o7000 == 0o0000

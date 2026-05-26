"""Targeted string replacement with progressive fallback.

Ported from OpenCode's edit tool (sst/opencode, TypeScript).
LLMs are bad at reproducing exact whitespace. Instead of failing
on a whitespace mismatch, we try increasingly flexible matching
strategies until one succeeds.

9 strategies in fallback order:
1. Exact match
2. Line-trimmed (strip trailing whitespace per line)
3. Block-anchor (first/last line anchors + Levenshtein similarity)
4. Whitespace-normalized (collapse all whitespace to single spaces)
5. Indentation-flexible (ignore leading indentation)
6. Escape-normalized (handle escape sequence differences)
7. Trimmed-boundary (strip leading/trailing whitespace from blocks)
8. Context-aware (similarity matching on first/last lines)
9. Multi-occurrence (all matches when replace_all=True)
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

# Strategy tunables (Block-anchor and Context-aware fuzzy matchers):
# - Block-anchor and Context-aware both require ≥2 lines to have first+last anchors.
# - 0.5 minimum similarity gates similarity() to short-circuit obviously bad anchors.
# - 0.3 minimum similarity for ambiguous block-anchor candidates (when multiple match).
# - 5-line slack lets the context-aware matcher look slightly past the expected end.
_MIN_ANCHOR_LINES = 2
_SIMILARITY_GATE = 0.5
_AMBIGUOUS_BLOCK_THRESHOLD = 0.3
_CONTEXT_SEARCH_SLACK = 5


class EditError(Exception):
    """Raised when the edit operation cannot be performed."""


@dataclass(frozen=True)
class EditResult:
    """Result of a successful edit operation."""

    file_path: Path
    before: str
    after: str
    strategy: str


def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if len(a) < len(b):
        return levenshtein(b, a)
    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[len(b)]


# Above this length, the O(n·m) Levenshtein DP is too expensive to run on the
# hot edit path (a 50 KB block ≈ 2.5B ops). Fuzzy matching on blocks this large
# is unreliable anyway, so fall back to the cheap length-ratio estimate.
_MAX_SIMILARITY_LEN = 4096


def similarity(a: str, b: str) -> float:
    """Normalized similarity between two strings (0.0 = different, 1.0 = identical)."""
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    min_len = min(len(a), len(b))
    # Fast reject: if lengths differ by more than 50%, skip expensive Levenshtein
    if min_len < max_len * 0.5:
        return min_len / max_len
    # Bound the DP cost: very large blocks use the length-ratio estimate so a
    # pathological edit can't hang the tool after the classifier approved it.
    if max_len > _MAX_SIMILARITY_LEN:
        return min_len / max_len
    return 1.0 - levenshtein(a, b) / max_len


def _line_ending(line: str) -> str:
    """The line ending of a keepends line ('' if it had none, i.e. EOF)."""
    if line.endswith("\r\n"):
        return "\r\n"
    if line.endswith("\n"):
        return "\n"
    if line.endswith("\r"):
        return "\r"
    return ""


def _detect_newline(text: str) -> str:
    r"""The text's dominant line ending — '\r\n' if CRLF outnumbers bare LF, else '\n'."""
    crlf = text.count("\r\n")
    lf = text.count("\n") - crlf
    return "\r\n" if crlf > lf else "\n"


def _normalize_newlines(text: str, ending: str) -> str:
    """Rewrite every line ending in ``text`` to ``ending`` (so edits match the file)."""
    unified = text.replace("\r\n", "\n").replace("\r", "\n")
    return unified if ending == "\n" else unified.replace("\n", ending)


def _splice_lines(
    before_lines: list[str],
    new: str,
    removed_lines: list[str],
    after_lines: list[str],
) -> str:
    """Join a line-based replacement, restoring the newline the block consumed.

    The matched block's lines were sliced out WITH their trailing newlines. If
    ``new`` doesn't end in a newline, re-add the block's line ending so it
    neither glues onto the next line (mid-file) nor drops the file's trailing
    newline (EOF). ``new``'s internal endings are normalized by the caller.
    """
    result = "".join(before_lines) + new
    if new and not new.endswith(("\n", "\r")):
        ending = _line_ending(removed_lines[-1]) if removed_lines else ""
        if after_lines:
            result += ending or "\n"  # mid-file: separate new from the next line
        elif ending:
            result += ending  # EOF: preserve the file's trailing newline
    return result + "".join(after_lines)


def _simple_replace(content: str, old: str, new: str) -> str | None:
    """Strategy 1: exact string match."""
    if old not in content:
        return None
    return content.replace(old, new, 1)


def _line_trimmed_replace(content: str, old: str, new: str) -> str | None:
    """Strategy 2: match after stripping trailing whitespace per line."""
    content_lines = content.splitlines(keepends=True)
    old_lines = old.splitlines()

    # Build trimmed versions for matching
    content_trimmed = [line.rstrip() for line in content_lines]
    old_trimmed = [line.rstrip() for line in old_lines]

    for i in range(len(content_trimmed) - len(old_trimmed) + 1):
        if content_trimmed[i : i + len(old_trimmed)] == old_trimmed:
            removed = content_lines[i : i + len(old_trimmed)]
            before = content_lines[:i]
            after_lines = content_lines[i + len(old_trimmed) :]
            return _splice_lines(before, new, removed, after_lines)
    return None


def _whitespace_normalized_replace(content: str, old: str, new: str) -> str | None:
    """Strategy 4: collapse all whitespace to single spaces before matching."""
    normalized_content = re.sub(r"\s+", " ", content)
    normalized_old = re.sub(r"\s+", " ", old)

    if normalized_old not in normalized_content:
        return None

    # Find the position in normalized space, then map back to original
    idx = normalized_content.index(normalized_old)

    # Map normalized position back to original content
    orig_pos = 0
    norm_pos = 0
    while norm_pos < idx and orig_pos < len(content):
        if content[orig_pos].isspace():
            # Skip whitespace runs in original
            while orig_pos < len(content) and content[orig_pos].isspace():
                orig_pos += 1
            norm_pos += 1  # normalized to single space
        else:
            orig_pos += 1
            norm_pos += 1

    start = orig_pos

    # Find end position similarly
    norm_end = idx + len(normalized_old)
    while norm_pos < norm_end and orig_pos < len(content):
        if content[orig_pos].isspace():
            while orig_pos < len(content) and content[orig_pos].isspace():
                orig_pos += 1
            norm_pos += 1
        else:
            orig_pos += 1
            norm_pos += 1

    end = orig_pos
    return content[:start] + new + content[end:]


def _indentation_flexible_replace(content: str, old: str, new: str) -> str | None:
    """Strategy 5: ignore leading indentation differences."""
    content_lines = content.splitlines(keepends=True)
    old_lines = old.splitlines()

    if not old_lines:
        return None

    # Strip leading whitespace from old for comparison
    old_stripped = [line.lstrip() for line in old_lines]

    for i in range(len(content_lines) - len(old_lines) + 1):
        candidate = [line.rstrip("\n\r").lstrip() for line in content_lines[i : i + len(old_lines)]]
        if candidate == old_stripped:
            # Determine the indentation of the first matched line
            original_indent = ""
            first_line = content_lines[i].rstrip("\n\r")
            stripped_first = first_line.lstrip()
            if stripped_first:
                original_indent = first_line[: len(first_line) - len(stripped_first)]

            # Apply original indentation to new content, preserving relative indent.
            # Strip the line ENDING before lstrip — lstrip on a keepends line keeps
            # the trailing newline inside `stripped`, which both miscomputes the
            # indent slice and doubles the newline on emit.
            new_lines = new.splitlines(keepends=True)
            # Base indent of the new content = the MINIMUM indent across its
            # non-empty lines (not the first line's). Using the first line's
            # indent flattens any later line that is LESS indented than the first
            # (relative would go negative → "").
            new_indents: list[str] = []
            for nl in new_lines:
                core = nl.rstrip("\n\r")
                stripped = core.lstrip()
                if stripped:
                    new_indents.append(core[: len(core) - len(stripped)])
            new_base_indent = min(new_indents, key=len) if new_indents else ""

            indented_new = ""
            for nl in new_lines:
                core = nl.rstrip("\n\r")
                stripped = core.lstrip()
                if not stripped:
                    indented_new += nl  # preserve blank lines as-is
                else:
                    line_indent = core[: len(core) - len(stripped)]
                    relative = (
                        line_indent[len(new_base_indent) :]
                        if line_indent.startswith(new_base_indent)
                        else ""
                    )
                    trailing = nl[len(core) :]
                    indented_new += original_indent + relative + stripped + trailing

            removed = content_lines[i : i + len(old_lines)]
            before = content_lines[:i]
            after_lines = content_lines[i + len(old_lines) :]
            return _splice_lines(before, indented_new, removed, after_lines)
    return None


def _block_anchor_replace(content: str, old: str, new: str) -> str | None:
    """Strategy 3: first/last line anchors with Levenshtein similarity on middle."""
    old_lines = old.splitlines()
    if len(old_lines) < _MIN_ANCHOR_LINES:
        return None

    content_lines = content.splitlines(keepends=True)
    first_target = old_lines[0].strip()
    last_target = old_lines[-1].strip()

    candidates: list[tuple[int, float]] = []

    for i in range(len(content_lines) - len(old_lines) + 1):
        first_line = content_lines[i].rstrip("\n\r").strip()
        last_line = content_lines[i + len(old_lines) - 1].rstrip("\n\r").strip()

        if first_line == first_target and last_line == last_target:
            # Score middle lines. Normalize both sides the same way — strip
            # each line's ending and rejoin with "\n" — so the similarity
            # score reflects real content drift, not line-ending/trailing-
            # whitespace mismatch between the (ending-less) old_lines and the
            # (keepends) content_lines. Without this, whitespace noise can
            # flip which of two same-anchor blocks scores highest → wrong-
            # region edit.
            middle_old = "\n".join(old_lines[1:-1])
            middle_content = "\n".join(
                cl.rstrip("\n\r") for cl in content_lines[i + 1 : i + len(old_lines) - 1]
            )
            score = similarity(middle_old, middle_content)
            candidates.append((i, score))

    if not candidates:
        return None

    # Unique anchor match wins outright (threshold 0). With multiple matches,
    # require enough middle-line similarity to disambiguate.
    threshold = 0.0 if len(candidates) == 1 else _AMBIGUOUS_BLOCK_THRESHOLD
    best = max(candidates, key=lambda c: c[1])

    if best[1] < threshold:
        return None

    i = best[0]
    removed = content_lines[i : i + len(old_lines)]
    before = content_lines[:i]
    after_lines = content_lines[i + len(old_lines) :]
    return _splice_lines(before, new, removed, after_lines)


def _escape_normalized_replace(content: str, old: str, new: str) -> str | None:
    """Strategy 6: normalize escape sequences before matching.

    Uses a sentinel for double-backslash pairs to prevent their replacement
    output from being consumed by subsequent escape processing.
    """
    _sentinel = "\x00BACKSLASH\x00"
    normalized_old = old.replace("\\\\", _sentinel)
    for escaped, unescaped in [
        ("\\n", "\n"),
        ("\\t", "\t"),
        ("\\r", "\r"),
        ('\\"', '"'),
        ("\\'", "'"),
    ]:
        normalized_old = normalized_old.replace(escaped, unescaped)
    normalized_old = normalized_old.replace(_sentinel, "\\")

    if normalized_old in content:
        return content.replace(normalized_old, new, 1)
    return None


def _is_word_char(c: str) -> bool:
    """True for identifier characters (so we don't replace a match embedded in a word)."""
    return c.isalnum() or c == "_"


def _trimmed_boundary_replace(content: str, old: str, new: str) -> str | None:
    """Strategy 7: match the whitespace-stripped block, but only as a standalone span.

    Uses a word-boundary check so ``foo`` doesn't replace the ``foo`` inside
    ``foobar`` / ``myfoo`` (a raw substring replace would). ``new`` is inserted
    verbatim — stripping it would discard intended leading/trailing whitespace.
    """
    trimmed = old.strip()
    if not trimmed:
        return None
    idx = content.find(trimmed)
    if idx == -1:
        return None
    end = idx + len(trimmed)
    before_ok = idx == 0 or not _is_word_char(content[idx - 1])
    after_ok = end == len(content) or not _is_word_char(content[end])
    if not (before_ok and after_ok):
        return None
    return content[:idx] + new + content[end:]


def _context_aware_replace(content: str, old: str, new: str) -> str | None:
    """Strategy 8: similarity matching on first/last lines as context anchors."""
    old_lines = old.splitlines()
    if len(old_lines) < _MIN_ANCHOR_LINES:
        return None

    content_lines = content.splitlines(keepends=True)
    first_target = old_lines[0].strip()
    last_target = old_lines[-1].strip()

    best_i = -1
    best_j = -1
    best_score = 0.0

    for i in range(len(content_lines) - 1):
        first_sim = similarity(content_lines[i].rstrip("\n\r").strip(), first_target)
        if first_sim < _SIMILARITY_GATE:
            continue

        # Look for last line within reasonable range
        max_end = min(i + len(old_lines) + _CONTEXT_SEARCH_SLACK, len(content_lines))
        for j in range(i + 1, max_end):
            last_sim = similarity(content_lines[j].rstrip("\n\r").strip(), last_target)
            if last_sim < _SIMILARITY_GATE:
                continue
            score = (first_sim + last_sim) / 2
            if score > best_score:
                best_score = score
                best_i = i
                best_j = j

    if best_i < 0:
        return None

    # Guard the span: context-aware is the most speculative matcher, so the
    # region it would delete must actually resemble `old` — otherwise a far-apart
    # anchor pair silently nukes unrelated lines between them. Compare with each
    # line stripped (both sides), matching how the anchors are compared, so
    # indentation drift between LLM-supplied `old` and the file doesn't tank the
    # score and reject a legitimate match.
    old_norm = "\n".join(line.strip() for line in old_lines)
    spanned_norm = "\n".join(cl.strip() for cl in content_lines[best_i : best_j + 1])
    if similarity(old_norm, spanned_norm) < _SIMILARITY_GATE:
        return None

    removed = content_lines[best_i : best_j + 1]
    before = content_lines[:best_i]
    after_lines = content_lines[best_j + 1 :]
    return _splice_lines(before, new, removed, after_lines)


def _multi_occurrence_count(content: str, old: str) -> int:
    """Count exact occurrences of old in content."""
    count = 0
    start = 0
    while True:
        idx = content.find(old, start)
        if idx == -1:
            break
        count += 1
        start = idx + len(old)
    return count


def _atomic_write(file_path: Path, content: str) -> None:
    """Write ``content`` to ``file_path`` via tmp file + rename.

    A crash mid-write leaves the original file intact instead of truncated.
    `os.replace` is atomic on POSIX and on Windows (same-filesystem rename).
    Preserves the original file's mode so we don't silently downgrade a
    locked-down file (e.g. 0600) to the default umask on edit.
    """
    tmp = file_path.with_suffix(file_path.suffix + ".edit-tmp")
    try:
        # Explicit utf-8 + newline="" so edits don't depend on the platform's
        # default encoding or rewrite line endings on the round trip.
        tmp.write_text(content, encoding="utf-8", newline="")
        # Inherit only the regular rwx bits — `& 0o777` strips setuid/setgid/
        # sticky so editing a privileged file doesn't silently propagate
        # those bits onto the new inode. (Plain `stat.S_IMODE` preserves
        # them: 0o7777, not 0o777.)
        tmp.chmod(file_path.stat().st_mode & 0o777)
        tmp.replace(file_path)
    finally:
        # After a successful replace, tmp no longer exists (no-op). On any
        # failure path, tmp may be a half-written orphan — drop it.
        tmp.unlink(missing_ok=True)


# Ordered list of strategies to try
type ReplaceFn = Callable[[str, str, str], str | None]

_STRATEGIES: list[tuple[str, ReplaceFn]] = [
    ("exact", _simple_replace),
    ("line_trimmed", _line_trimmed_replace),
    ("block_anchor", _block_anchor_replace),
    ("whitespace_normalized", _whitespace_normalized_replace),
    ("indentation_flexible", _indentation_flexible_replace),
    ("escape_normalized", _escape_normalized_replace),
    ("trimmed_boundary", _trimmed_boundary_replace),
    ("context_aware", _context_aware_replace),
]


def edit_file(
    file_path: Path,
    old_string: str,
    new_string: str,
    *,
    replace_all: bool = False,
) -> EditResult:
    """Edit a file using targeted string replacement with progressive fallback.

    Tries 9 replacement strategies in order until one succeeds.

    Args:
        file_path: Path to the file to edit.
        old_string: Text to find and replace.
        new_string: Replacement text.
        replace_all: If True, replace all occurrences. If False, error on multiple matches.

    Returns:
        EditResult with file path, before/after content, and which strategy matched.

    Raises:
        EditError: If the file doesn't exist, old_string is empty, old equals new,
                   multiple matches found (without replace_all), or no match found.
    """
    if not file_path.exists():
        msg = f"File not found: {file_path}"
        raise EditError(msg)

    if not old_string:
        msg = "old_string must not be empty"
        raise EditError(msg)

    if old_string == new_string:
        msg = "old_string and new_string must be different"
        raise EditError(msg)

    before = file_path.read_text(encoding="utf-8", newline="")

    # Match the file's line endings so an LLM's LF new_string doesn't leave mixed
    # endings when editing a CRLF file (read_text/write_text use newline="" — no
    # translation — so endings round-trip byte-for-byte otherwise).
    new_string = _normalize_newlines(new_string, _detect_newline(before))

    # Check for multiple exact matches
    count = _multi_occurrence_count(before, old_string)
    if count > 1 and not replace_all:
        msg = (
            f"Found multiple ({count}) occurrences of old_string. "
            "Provide more surrounding context to make the match unique, "
            "or use replace_all=True."
        )
        raise EditError(msg)

    if replace_all:
        if count == 0:
            # replace_all requires at least one exact match; the fuzzy
            # strategies only ever produce a single substitution.
            msg = (
                "replace_all=True requires at least one exact match of old_string; "
                "found none."
            )
            raise EditError(msg)
        after = before.replace(old_string, new_string)
        _atomic_write(file_path, after)
        return EditResult(file_path=file_path, before=before, after=after, strategy="replace_all")

    # Try each strategy in order
    for name, strategy_fn in _STRATEGIES:
        result = strategy_fn(before, old_string, new_string)
        if result is not None:
            _atomic_write(file_path, result)
            return EditResult(file_path=file_path, before=before, after=result, strategy=name)

    msg = (
        "Could not find old_string in the file. "
        "It must match exactly, including whitespace, indentation, and line endings."
    )
    raise EditError(msg)

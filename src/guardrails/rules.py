"""Rule-based classification — fast-path allow/block without LLM calls.

Deterministic regex patterns for tool calls that are obviously safe
(read-only operations) or obviously dangerous (destructive commands).

Content-aware rules inspect file content and paths for dangerous patterns:
- A1 Self-Destruction (os.remove(__file__), shutil.rmtree)
- A2 Self-Modification (writes to guardrails/policy/profiles paths)
- A6 Credential Access — both WRITES to and READS/listings/searches of
  credential locations (.env, secrets/, .ssh/, *.pem, credentials.*)

Everything else gets NEEDS_JUDGMENT, which routes to the LLM classifier.
"""

from __future__ import annotations

import fnmatch
import posixpath
import re
from enum import StrEnum
from typing import Any


class Decision(StrEnum):
    """Result of rule-based classification."""

    ALLOW = "allow"
    BLOCK = "block"
    NEEDS_JUDGMENT = "needs_judgment"


# Tools that are always safe — read-only, no side effects.
ALWAYS_ALLOW_TOOLS: frozenset[str] = frozenset({
    "read_file",
    "glob_search",
    "grep_search",
    "list_directory",
})

# Dangerous command patterns — block without an LLM call.
DANGEROUS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"rm\s+(-[a-zA-Z]*f[a-zA-Z]*(\s|$)|.*--force)", re.IGNORECASE),
    re.compile(r"rm\s+(-[a-zA-Z]*r|.*--recursive)", re.IGNORECASE),
    re.compile(r"curl\s+.*\|\s*(env\s+)?(exec\s+)?\S*(bash|sh|zsh)\b", re.IGNORECASE),
    re.compile(r"wget\s+.*\|\s*(env\s+)?(exec\s+)?\S*(bash|sh|zsh)\b", re.IGNORECASE),
    re.compile(r"mkfs\.", re.IGNORECASE),
    re.compile(r"dd\s+if=", re.IGNORECASE),
    re.compile(r":\(\)\s*\{\s*:\|:\s*&\s*\}\s*;"),  # fork bomb
)

# A1 Self-Destruction — content patterns that delete the file or workspace.
SELF_DESTRUCTIVE_CONTENT: tuple[re.Pattern[str], ...] = (
    re.compile(r"os\.remove\s*\(\s*__file__\s*\)"),
    re.compile(r"os\.unlink\s*\(\s*__file__\s*\)"),
    re.compile(r"pathlib\.Path\s*\(\s*__file__\s*\)\s*\.unlink"),
    re.compile(r"shutil\.rmtree\s*\("),
)

# A2 Self-Modification — paths that govern the agent's own behavior.
PROTECTED_PATH_PREFIXES: tuple[str, ...] = (
    "src/guardrails/",
    "src/policy/",
    "src/profiles/",
)

# A6 Credential Access. Read-class tools expose THREE field shapes, but the
# first two share ONE credential vocabulary — a single set of anchored patterns
# matches both real paths (read_file.file_path, grep/glob.path; normpath'd
# first) and file globs (glob.pattern, grep.include; matched verbatim so ``**``
# survives). The shared boundaries ``_LB`` / ``_RB`` each accept a path
# separator, a glob wildcard, OR string start/end — so ``**/id_rsa``,
# ``*id_rsa``, and ``~/.ssh/id_rsa`` all anchor, while ``myid_rsa`` /
# ``*.pemfile`` / ``src/secrets.py`` do NOT (a substring is not a segment). Both
# boundaries must be glob-aware: a glob-aware right with a path-only left lets
# ``*server.key`` slip the gate. The third shape — grep's CONTENT pattern, the
# string searched FOR — is matched by ``_SECRET_CONTENT_RE`` (key material only).
#
# This is a high-signal heuristic, not an exhaustive registry — a credential
# filename it doesn't recognize falls through to ALLOW (the post-6 OpenShell
# policy layer is where exhaustive, allowlist-based control lives). Deliberately
# EXCLUDED to protect the auto-mode payoff (escalating these would prompt on
# everyday reads without catching real secrets):
#   - generic ``*.key`` (i18n/license/config) — only key-ish stems below escalate
#   - public certs ``.crt`` / ``.cer`` / ``.der`` and ``.asc`` (public keys,
#     release signatures) — these carry no private material
# ``.env.example`` and friends DO escalate: env templates are a known
# secret-leak vector and escalation only routes to LLM judgment, not a block.
# Boundaries treat a credential token as a complete path/glob segment or
# extension, never a substring of a longer name. ``_LB`` / ``_RB`` accept
# string start/end, a path separator, or a glob wildcard (``*?[]``) on either
# side, so a real path AND a contiguous-token glob (``*id_rsa``, ``**/*.pem``)
# both anchor. ``_CRB`` additionally tolerates an appended ``.bak`` / ``.old`` /
# ``.enc`` chain so a credential plus a backup suffix (``key.pem.bak``) still
# escalates. These patterns are robust for concrete paths, but a glob can
# interleave wildcards/char-classes/braces INSIDE a token (``i?_rsa``,
# ``*.[p]em``, ``*.{pem,key}``) where no regex boundary can reach — so glob
# fields get a SECOND, glob-engine-based check in ``_glob_targets_credential``
# (fnmatch against credential exemplars). Regex here, fnmatch there.
_LB = r"(?:^|[/*?\[\]])"
_RB = r"(?:[/*?\[\]]|$)"
_CRB = r"(?:\.[^/]*)?(?:[/*?\[\]]|$)"

CREDENTIAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    # env files, two shapes (a literal dot anchors both, so a virtualenv DIR
    # ``project/env/`` and source files ``client.env.ts`` / globs ``*.env.ts``
    # are NOT mistaken for secrets): leading-dot dotenv + ``.env.<mode>``
    # variants (anchored at a real start/separator, not a glob ``*``), and a
    # trailing ``.env`` suffix (``prod.env``, ``*.env``).
    re.compile(r"(^|/)\.env(rc)?(\.[^/]*)?(?:[*?/]|$)", re.IGNORECASE),
    re.compile(r"\.env(?:[*?/]|$)", re.IGNORECASE),
    # secret(s) as a full path segment — not the everyday ``secrets.py`` basename.
    re.compile(rf"{_LB}secrets?{_RB}", re.IGNORECASE),
    # credential directories (ssh / cloud / kube / docker / gpg).
    re.compile(rf"{_LB}\.(ssh|aws|gcp|azure|kube|docker|gnupg){_RB}", re.IGNORECASE),
    # a credentials DATA file — not docs/source merely named ``credentials.*``.
    # Bare (``.aws/credentials``) OR a data extension, then an optional backup
    # suffix; ``credentials.md`` / ``credentials-store.go`` stay allowed.
    re.compile(
        rf"{_LB}credentials"
        rf"(?:{_RB}|\.(?:json|ini|ya?ml|cfg|conf|toml|txt|db|sqlite3?|xml|plist|store){_CRB})",
        re.IGNORECASE,
    ),
    # SSH private keys + trust material (authorized_keys / known_hosts).
    re.compile(rf"{_LB}id_(rsa|dsa|ecdsa|ed25519)(?:\.|{_RB})", re.IGNORECASE),
    re.compile(rf"{_LB}(authorized_keys|known_hosts){_CRB}", re.IGNORECASE),
    # private-key / keystore / encrypted-secret / IaC-state serializations.
    re.compile(
        rf"\.(pem|p12|pfx|p8|jks|kdbx|keystore|ovpn|kubeconfig|gpg|tfstate){_CRB}",
        re.IGNORECASE,
    ),
    # ``*.key`` ONLY when the stem names a private key (server.key, tls.key, …).
    re.compile(
        rf"{_LB}(privkey|private|priv|server|client|tls|ssl|secret)\.key{_CRB}",
        re.IGNORECASE,
    ),
    # credential dotfiles + single-purpose token files.
    re.compile(
        rf"{_LB}\.(npmrc|pypirc|netrc|pgpass|git-credentials|vault-token)(?:\.|{_RB})",
        re.IGNORECASE,
    ),
)

# High-signal secret MATERIAL in a grep content pattern (the string being
# searched FOR). Deliberately narrow — "password"/"api_key" are everyday code
# searches, so only unambiguous key material escalates.
_SECRET_CONTENT_RE = re.compile(r"BEGIN [A-Z ]*PRIVATE KEY", re.IGNORECASE)

# Canonical credential exemplar FILENAMES. A glob field (glob.pattern /
# grep.include) is matched by TESTING the glob against these with the real glob
# engine (``fnmatch``) — NOT by regex-matching the glob string, which any
# mid-token wildcard (``i?_rsa``), char class (``*.[p]em``), or brace defeats.
# Every entry is a basename that is SELF-IDENTIFYING as a credential: credential
# DIRECTORIES (.ssh/, .aws/, .docker/, secrets/, ...) are handled by the regex
# vocabulary instead, because a dir-qualified file with a generic basename
# (.docker/config.json) would otherwise lend ``config.json`` to the basename
# match and over-gate every ``**/config.json``. A consistency test pins each
# exemplar to ``_is_credential_path``; not exhaustive (same heuristic caveat).
# ``.env.local`` / ``.env.production`` are the common dotenv-mode secrets; the
# rarer modes (test/dev/staging) are omitted so a generic ``*.test`` / ``*.dev``
# glob doesn't escalate — ``*.env.*`` still escalates via these two, and a READ
# of any ``.env.<mode>`` is caught by the regex vocabulary.
_CREDENTIAL_EXEMPLARS: tuple[str, ...] = (
    ".env", ".envrc", "prod.env", ".env.local", ".env.production",
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "authorized_keys", "known_hosts",
    "server.pem", "server.key", "private.key", "cert.p12", "cert.pfx", "key.p8",
    "store.jks", "vault.kdbx", "x.keystore", "client.ovpn", "host.kubeconfig",
    "secret.gpg", "terraform.tfstate",
    "credentials", "credentials.json", "credentials.xml", "credentials.yaml",
    ".npmrc", ".pypirc", ".netrc", ".pgpass", ".git-credentials", ".vault-token",
)

# Ordinary files a BROAD glob (``**/*``, ``*``, ``*.json``) also matches. A glob
# that hits a credential exemplar AND one of these is a catch-all, not a
# credential hunt — escalating it would wreck the auto-mode payoff — so it stays
# allowed; a glob that hits ONLY credential exemplars is credential-specific.
# Each entry is a representative file of a COMMON extension that collides with a
# credential exemplar (``data.json``↔``credentials.json``, ``pom.xml``↔
# ``credentials.xml``, ``messages.local``↔``.env.local``) so the everyday glob for
# that extension isn't over-gated. NB: there is deliberately no ``*.env.<x>``
# benign exemplar — a wildcard before ``.env.`` is a secret hunt, not a catch-all.
_BENIGN_EXEMPLARS: tuple[str, ...] = (
    "main.py", "README.md", "index.ts", "app.js", "styles.css",
    "data.json", "config.yaml", "pom.xml", "messages.local", "notes.txt", "report.pdf",
)

# A glob longer than this is treated as adversarial and escalated rather than
# compiled/expanded — bounds both the fnmatch translation and brace fan-out.
_MAX_GLOB_LEN = 500


def _matches_credential(value: str) -> bool:
    """True if a path-or-glob string matches the shared credential vocabulary."""
    return any(p.search(value) for p in CREDENTIAL_PATTERNS)


def _is_credential_path(path: str) -> bool:
    """True if a real filesystem path points at a credential location."""
    return bool(path) and _matches_credential(posixpath.normpath(path))


# Cap on brace-expansion fan-out — a pathological ``{a,b}{c,d}...`` glob blows
# up combinatorially, so past this many candidates we FAIL CLOSED (escalate)
# rather than silently dropping un-vetted alternatives.
_MAX_BRACE_EXPANSION = 64


def _expand_braces(glob: str) -> tuple[list[str], bool]:
    """Expand shell ``{a,b}`` alternations into concrete candidate globs.

    Handles nesting by reprocessing each reassembled candidate (so
    ``{a,{b,c}}`` fully expands). ``fnmatch`` understands ``*?[]`` but not
    braces, so they must be expanded before translation.

    Returns ``(candidates, truncated)``. ``truncated`` is True if the fan-out
    hit ``_MAX_BRACE_EXPANSION`` — the caller fails closed (escalates) rather
    than vet a partial expansion.
    """
    done: list[str] = []
    work = [glob]
    while work:
        current = work.pop()
        # ``[^{}]`` matches the INNERMOST brace group; reassembling and
        # reprocessing the result is what unwinds nested braces.
        match = re.search(r"\{([^{}]*)\}", current)
        if match is None:
            done.append(current)
            continue
        pre, post = current[: match.start()], current[match.end() :]
        work.extend(pre + option + post for option in match.group(1).split(","))
        if len(work) + len(done) > _MAX_BRACE_EXPANSION:
            return done, True
    return done, False


def _glob_matches_exemplar(glob: str, exemplars: tuple[str, ...]) -> bool:
    """True if a single (brace-free) glob could match one of the exemplars.

    Two semantics, because the two glob fields differ: ``glob.pattern`` matches
    full relative paths (pathlib), so we test each exemplar at the root and one
    dir deep (``e`` / ``d/e``); ``grep.include`` matches file BASENAMES, so we
    also test the glob's last segment against each exemplar's basename. The
    basename check also closes an explicit-depth + mid-token-wildcard dodge
    (``a/b/i?_rsa``) that the full-path check, anchored at the glob's literal
    prefix, would miss.
    """
    try:
        full = re.compile(fnmatch.translate(glob), re.IGNORECASE)
    except re.error:
        return False
    if any(full.match(e) or full.match("d/" + e) for e in exemplars):
        return True
    base = posixpath.basename(glob)
    if not base or base == glob:
        return False
    try:
        base_matcher = re.compile(fnmatch.translate(base), re.IGNORECASE)
    except re.error:
        return False
    return any(base_matcher.match(posixpath.basename(e)) for e in exemplars)


def _glob_targets_credential(glob: str) -> bool:
    """True if a file-glob (glob.pattern / grep.include) targets credential files.

    Tests the glob against canonical credential exemplars using ``fnmatch`` —
    the real glob engine — instead of regex-matching the glob STRING, which any
    mid-token wildcard, char class, or brace defeats. A glob that also matches a
    benign exemplar is a broad catch-all (e.g. ``**/*``), not a credential hunt,
    so it stays allowed. Over-length / un-expandable globs fail closed.
    """
    if not glob or len(glob) > _MAX_GLOB_LEN:
        return bool(glob)
    candidates, truncated = _expand_braces(glob)
    if truncated:
        return True
    for candidate in candidates:
        # Two complementary checks: the regex vocabulary catches a contiguous
        # credential token (concrete paths, ``**/*.pem``, ``*.env``), and the
        # fnmatch-exemplar test catches wildcard-obfuscated globs the regex
        # can't (``i?_rsa``, ``*.[p]em``) — escalating only when the glob is
        # credential-SPECIFIC, not a benign-matching catch-all.
        if _matches_credential(candidate):
            return True
        if _glob_matches_exemplar(candidate, _CREDENTIAL_EXEMPLARS) and not _glob_matches_exemplar(
            candidate, _BENIGN_EXEMPLARS,
        ):
            return True
    return False


def _credential_access_escalates(short_name: str, tool_args: dict[str, Any]) -> bool:
    """True if a READ-CLASS tool targets a credential location/glob/content (A6).

    Each tool carries the credential target in a different field shape:
    ``read_file`` / ``list_directory`` a real path; ``glob_search`` a path OR a
    file-glob; ``grep_search`` a path, an include-glob, OR key material in its
    content pattern. A bare read/list/search with no credential target is NOT
    escalated here — it keeps the always-allow fast-path (the auto-mode payoff).
    """
    if short_name in {"read_file", "list_directory"}:
        field = "file_path" if short_name == "read_file" else "dir_path"
        return _is_credential_path(str(tool_args.get(field, "")))
    if short_name == "glob_search":
        return _is_credential_path(str(tool_args.get("path", ""))) or _glob_targets_credential(
            str(tool_args.get("pattern", "")),
        )
    if short_name == "grep_search":
        return (
            _is_credential_path(str(tool_args.get("path", "")))
            or _glob_targets_credential(str(tool_args.get("include", "")))
            or _SECRET_CONTENT_RE.search(str(tool_args.get("pattern", ""))) is not None
        )
    return False


def _check_content_safety(tool_args: dict[str, Any]) -> Decision:
    """Inspect write/edit content for self-destructive patterns (A1).

    ``write_file`` carries content in ``text``; ``edit_file`` carries the
    new content in ``new_string`` — both must be inspected, or an edit
    that injects ``shutil.rmtree(...)`` slips past the deterministic block.
    """
    content = str(
        tool_args.get("text", "")
        or tool_args.get("content", "")
        or tool_args.get("new_string", ""),
    )
    if not content:
        return Decision.NEEDS_JUDGMENT

    for pattern in SELF_DESTRUCTIVE_CONTENT:
        if pattern.search(content):
            return Decision.BLOCK

    return Decision.NEEDS_JUDGMENT


def _check_path_safety(file_path: str) -> Decision:
    """Block writes to protected (A2) or credential (A6) paths."""
    if not file_path:
        return Decision.NEEDS_JUDGMENT

    # Normalize so ``./src/guardrails/x.py``, ``src//guardrails/x.py``, and
    # ``src/./guardrails/x.py`` all collapse to ``src/guardrails/x.py``
    # before the substring match. Without this, those forms slip past the
    # A2 deterministic block and fall through to the (weaker) LLM tier.
    # posixpath is used so Windows backslashes don't matter — tool paths
    # in this stack are always posix-style.
    normalized = posixpath.normpath(file_path)

    # A2: a normalized path containing a protected prefix anywhere is blocked.
    # Case-insensitive (like A6 below) so an uppercased path on a case-folding
    # filesystem can't slip a self-modification past the deterministic gate.
    lowered = normalized.casefold()
    for prefix in PROTECTED_PATH_PREFIXES:
        if prefix.casefold() in lowered:
            return Decision.BLOCK

    # A6: regex patterns handle both relative ("/secrets/x") and bare (".env") forms.
    for pattern in CREDENTIAL_PATTERNS:
        if pattern.search(normalized):
            return Decision.BLOCK

    return Decision.NEEDS_JUDGMENT


def check_rules(tool_name: str, tool_args: dict[str, Any]) -> Decision:
    """Fast-path classification using deterministic rules.

    Checks in order:
    1. Credential reads/listings/searches — escalated, not auto-allowed
       (A6 covers read_file, list_directory, glob_search, grep_search)
    2. Always-allow tools (read-only, no side effects)
    3. Dangerous shell command patterns
    4. Content-aware checks for write/edit tools (A1, A2, A6)

    Args:
        tool_name: The tool being called. A FunctionGroup-namespaced name
            (e.g. ``getglad_tools__read_file``) is normalized to its bare form
            before the allow-set check; otherwise the live middleware path
            would never match.
        tool_args: The tool's arguments.

    Returns:
        ALLOW for obviously safe tools, BLOCK for dangerous patterns,
        NEEDS_JUDGMENT for everything else (requires LLM evaluation).
    """
    short_name = tool_name.rsplit("__", maxsplit=1)[-1]

    # A6 — credential access covers every READ-CLASS tool, not just writes. A
    # read of .env, a listing of .ssh/, a grep restricted to *.env, or a glob
    # for **/.env* would otherwise exfiltrate (or enumerate) secrets with no
    # gate. Listing a credential dir is recon — it reveals key filenames — so it
    # escalates like a read. General searches with no credential target still
    # auto-allow.
    if _credential_access_escalates(short_name, tool_args):
        return Decision.NEEDS_JUDGMENT

    if short_name in ALWAYS_ALLOW_TOOLS:
        return Decision.ALLOW

    if short_name in {"bash", "shell"}:
        command = str(tool_args.get("command", ""))
        for pattern in DANGEROUS_PATTERNS:
            if pattern.search(command):
                return Decision.BLOCK

    if short_name in {"write_file", "edit_file"}:
        file_path = str(tool_args.get("file_path", ""))

        if _check_path_safety(file_path) == Decision.BLOCK:
            return Decision.BLOCK

        if _check_content_safety(tool_args) == Decision.BLOCK:
            return Decision.BLOCK

    return Decision.NEEDS_JUDGMENT

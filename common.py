"""Shared, dependency-free helpers for the per-article Legal-AMR agent.

This module used to be imported by path from the older ``dataset_amr`` project.
It now lives here so that ``agentic_dataset`` runs on its own: model-output
extraction, cheap PENMAN balancing checks, core AMR validation, and the
coverage score used by the validator.

Keep it small: paths, JSON extraction, text normalisation, AMR validation.
"""

from __future__ import annotations

import json
import re
from typing import Any

try:
    import penman
except ImportError:  # pragma: no cover - only before dependencies are installed
    penman = None

# AMR variables are short lowercase names, e.g. p, p2, z10.
VAR_RE = re.compile(r"^[a-z][a-z0-9]*$")
# Concept = lemma or lemma-NN. Be permissive about the lemma body.
CONCEPT_RE = re.compile(r"^[a-z][a-z0-9_-]*(-[0-9]{1,2})?$")
# :ARG0/:ARG1-of are uppercase; named roles are lowercase.
ROLE_RE = re.compile(r"^:(?:ARG[0-9]+(?:-of)?|[a-z][a-z0-9-]*)$")

# AMR 3.0 ":wiki" role: the canonical English Wikipedia article title of a
# named entity, with spaces written as underscores. The bare value "-" means
# "this entity has no Wikipedia page".
WIKI_TITLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_'(),.&#_-]*$")

# Placeholder the model writes for every named entity; the resolver replaces it
# with the real title (or removes it when there is no article).
WIKI_PLACEHOLDER = "null"

TOKEN_RE = re.compile(r"[A-Za-z]{3,}|\d+")
STOPWORDS = {
    "the", "and", "for", "are", "but", "not", "with", "that", "this", "from",
    "has", "have", "had", "was", "were", "been", "being", "its", "their",
    "which", "who", "whom", "whose", "into", "under", "over", "such", "than",
    "then", "they", "them", "there", "these", "those", "shall", "may", "can",
    "will", "would", "could", "should", "also", "any", "all", "his", "her",
    "him", "she", "our", "your", "you", "it", "is", "of", "to", "in", "on",
    "by", "as", "at", "or", "an", "a", "be", "no", "nor", "if", "so", "we",
}


# --------------------------------------------------------------------------- #
# Model-output extraction
# --------------------------------------------------------------------------- #
def strip_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json|penman)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Find one JSON object in a model response, tolerating prose/fences."""
    if not text:
        return None
    text = strip_fence(text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def extract_json_payload(text: str) -> Any | None:
    """Find one JSON object or array in a model response.

    Batch generation returns ``{"results": [...]}`` (or occasionally a bare
    array), so this is the batch-safe sibling of :func:`extract_json_object`.
    """
    if not text:
        return None
    text = strip_fence(str(text))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        return None
    start = min(starts)
    end = text.rfind("}" if text[start] == "{" else "]")
    if end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
# AMR validation
# --------------------------------------------------------------------------- #
def top_level_forms(text: str) -> tuple[int, int]:
    """Count top-level parenthesised forms and the final paren depth.

    ``penman.decode`` silently accepts a second graph after the first one, so
    this cheap scan catches strings that contain more than one top-level form.
    Parentheses inside quoted strings are ignored.
    """
    count = 0
    depth = 0
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "(":
            if depth == 0:
                count += 1
            depth += 1
        elif ch == ")":
            depth -= 1
    return count, depth


def wiki_value_errors(target: str) -> list[str]:
    """Check the value of one AMR 3.0 ``:wiki`` role."""
    text = str(target)
    if text == "-":
        return []
    if text.startswith('"') and text.endswith('"'):
        try:
            title = json.loads(text)
        except json.JSONDecodeError:
            return [":wiki value is not a valid JSON string"]
        if title.strip().lower() == WIKI_PLACEHOLDER:
            return []
        if not title.strip():
            return [":wiki title must not be empty"]
        if any(ch.isspace() for ch in title):
            return [
                ":wiki title must use underscores instead of spaces: "
                f"{title!r}"
            ]
        if not WIKI_TITLE_RE.match(title):
            return [f":wiki title {title!r} is not a valid Wikipedia article title"]
        return []
    if VAR_RE.match(text):
        return [":wiki must be a literal attribute, not a node"]
    return [":wiki value must be a Wikipedia title string or '-'"]


def validate_amr(amr: str | None) -> tuple[bool, list[str], dict[str, Any]]:
    """Check that a string is a well-formed core AMR graph.

    Returns (ok, errors, info). ``info`` is always populated when a graph can
    be parsed, even if later semantic checks fail.
    """
    errors: list[str] = []
    info: dict[str, Any] = {}
    if not amr or not isinstance(amr, str) or not amr.strip():
        return False, ["empty graph"], info
    if penman is None:
        return False, ["penman is not installed (pip install penman)"], info

    amr = amr.strip()
    if not amr.startswith("(") or not amr.endswith(")"):
        errors.append("graph must start with '(' and end with ')'")

    forms, depth = top_level_forms(amr)
    if forms != 1:
        errors.append(
            "graph must be single-rooted: exactly one top-level form; "
            f"found {forms}"
        )
    if depth != 0:
        errors.append(f"graph has unbalanced parentheses (final depth {depth})")

    try:
        graph = penman.decode(amr)
    except Exception as exc:  # noqa: BLE001 - penman raises several types
        return False, errors + [f"PENMAN parse failed: {exc}"], info

    triples = list(graph.triples)
    if not triples:
        return False, errors + ["graph has no triples"], info

    defined: dict[str, str] = {}
    concepts: dict[str, set[str]] = {}
    edges: list[tuple[str, str, str]] = []
    for source, role, target in triples:
        if role in ("instance", ":instance"):
            concepts.setdefault(source, set()).add(str(target))
            defined[source] = str(target)
        else:
            edges.append((str(source), str(role), str(target)))

    for source, found in concepts.items():
        if len(found) > 1:
            errors.append(
                f"variable {source!r} is defined more than once with "
                f"different concepts: {sorted(found)}"
            )

    used: set[str] = {s for s, _, _ in edges}
    for _, _, target in edges:
        if VAR_RE.match(target):
            used.add(target)
    undefined = sorted(used - set(defined))
    if undefined:
        errors.append(f"variables used but never defined: {undefined}")

    for source, role, target in edges:
        if not ROLE_RE.match(role):
            errors.append(
                f"role {role!r} is malformed: use :ARG0-:ARG5 / :ARGn-of "
                "in UPPERCASE, or a named role such as :polarity in lowercase"
            )
        if role == ":wiki":
            errors.extend(wiki_value_errors(target))

    for var, concept in defined.items():
        if not VAR_RE.match(var):
            errors.append(f"variable name {var!r} must match [a-z][a-z0-9]*")
        if not (concept.startswith('"') or CONCEPT_RE.match(concept)):
            errors.append(f"concept {concept!r} on variable {var!r} is not lemma or lemma-NN")

    # AMR must be acyclic. penman accepts self-loops and cycles, which are not
    # valid AMR and would silently poison a training set.
    adjacency: dict[str, list[str]] = {}
    for source, _, target in edges:
        if VAR_RE.match(target):
            adjacency.setdefault(source, []).append(target)
    colour: dict[str, int] = {}

    def visit(node: str) -> bool:
        colour[node] = 1
        for nxt in adjacency.get(node, []):
            state = colour.get(nxt, 0)
            if state == 1 or (state == 0 and visit(nxt)):
                return True
        colour[node] = 2
        return False

    if any(colour.get(node, 0) == 0 and visit(node)
           for node in set(defined) | set(adjacency)):
        errors.append("graph contains a cycle (AMR must be acyclic)")

    incoming: dict[str, int] = {}
    for _, _, target in edges:
        if VAR_RE.match(target):
            incoming[target] = incoming.get(target, 0) + 1

    info = {
        "n_nodes": len(defined),
        "n_edges": len(edges),
        "n_negations": sum(
            1 for _, role, target in edges
            if role == ":polarity" and target == "-"
        ),
        "n_reentrancies": sum(1 for n in incoming.values() if n > 1),
        "concepts": sorted(set(defined.values())),
    }
    return not errors, errors, info


_SUFFIXES = (
    "ations", "ation", "ments", "ment", "ings", "ing", "edly", "ed",
    "ies", "es", "s",
)


def stem(word: str) -> str:
    """A light stemmer, enough to match a surface form to an AMR lemma."""
    text = word
    changed = True
    while changed:
        changed = False
        for suffix in _SUFFIXES:
            if len(text) > len(suffix) + 2 and text.endswith(suffix):
                text = text[:-len(suffix)]
                changed = True
                break
    return text


def coverage(sentence: str, amr: str) -> float:
    """Rough check that the graph kept the sentence's content words."""
    sent = {w for w in TOKEN_RE.findall(sentence.lower()) if w not in STOPWORDS}
    if not sent:
        return 1.0
    got = set(TOKEN_RE.findall(amr.lower()))
    got_stems = {stem(word) for word in got}
    covered = sum(
        1 for word in sent
        if word in got or stem(word) in got_stems
    )
    return covered / len(sent)

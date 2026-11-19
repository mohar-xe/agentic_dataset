"""Sentence rewrites and the graph↔sentence consistency contract.

Rewrites are explicit, verifiable spans: the model (or the mock) returns
``(surface, occurrence, replacement)`` and this module locates the exact span,
applies it, and proves the rest of the sentence is unchanged.

The invariant enforced by ``validate``:

* every specific legal name introduced in ``sentence`` but not literally
  present in ``sentence_original`` must be backed by an edit span; and
* every edit must be backed by a matching specific node in the AMR.

Offsets are never trusted from the model: they are recomputed here.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

_LEGACY = str(Path(__file__).resolve().parent.parent / "dataset_amr" / "scripts")
if _LEGACY not in sys.path:
    sys.path.insert(0, _LEGACY)

from common import extract_json_payload, validate_amr  # noqa: E402
from corpus import GENERIC_WORDS, _SINGULAR, _norm  # noqa: E402
from legal_schema import check_legal_amr  # noqa: E402

LEGAL_CONCEPTS = {
    "statute", "regulation", "provision", "gov-entity", "judgment",
    "notification", "legal-instrument",
}


@dataclass
class Edit:
    surface: str
    occurrence: int
    replacement: str
    resolved_to: str = ""
    evidence: str = ""
    kind: str = ""
    start: int = -1
    end: int = -1

    def to_record(self) -> dict:
        return asdict(self)


def _find_occurrence(text: str, needle: str, occurrence: int) -> tuple[int, int] | None:
    if not needle:
        return None
    start = 0
    seen = 0
    while True:
        idx = text.find(needle, start)
        if idx < 0:
            return None
        seen += 1
        if seen == occurrence:
            return idx, idx + len(needle)
        start = idx + 1


def normalize_edit(raw: dict[str, Any]) -> Edit | None:
    if not isinstance(raw, dict):
        return None
    surface = str(raw.get("surface") or "")
    replacement = str(raw.get("replacement") or "")
    if not surface or not replacement:
        return None
    try:
        occ = int(raw.get("occurrence") or 1)
    except (TypeError, ValueError):
        occ = 1
    return Edit(
        surface=surface, occurrence=max(1, occ), replacement=replacement,
        resolved_to=str(raw.get("resolved_to") or ""),
        evidence=str(raw.get("evidence") or ""),
        kind=str(raw.get("kind") or ""),
    )


def apply_edits(original: str, edits: list[dict[str, Any]],
                cfg: dict[str, Any]) -> tuple[str | None, list[Edit], list[str]]:
    """Locate spans in ``original`` and return ``(rewritten, located, errors)``."""
    located: list[Edit] = []
    errors: list[str] = []
    rw = cfg.get("rewrite", {})
    max_n = rw.get("max_per_sentence", 4)
    max_growth = rw.get("max_growth_chars", 160)

    for raw in edits or []:
        edit = normalize_edit(raw)
        if edit is None:
            errors.append("malformed edit (needs surface, occurrence, replacement)")
            continue
        span = _find_occurrence(original, edit.surface, edit.occurrence)
        if span is None:
            errors.append(
                f"edit span not found: {edit.surface!r} occurrence {edit.occurrence}"
            )
            continue
        edit.start, edit.end = span
        located.append(edit)

    if len(located) > max_n:
        errors.append(f"too many edits: {len(located)} > {max_n}")
    spans = sorted((e.start, e.end) for e in located)
    for (a1, a2), (b1, b2) in zip(spans, spans[1:]):
        if b1 < a2:
            errors.append("overlapping edits")
            break
    if errors:
        return None, located, errors

    rewritten = original
    for edit in sorted(located, key=lambda e: e.start, reverse=True):
        rewritten = rewritten[:edit.start] + edit.replacement + rewritten[edit.end:]
    growth = len(rewritten) - len(original)
    if growth > max_growth:
        errors.append(f"rewrite grows sentence by {growth} chars > {max_growth}")
        return None, located, errors
    return rewritten, located, errors


def _extract_specific_names(amr: str) -> list[tuple[str, str, str]]:
    """``(var, concept, legal-name)`` for named legal nodes."""
    try:
        import penman
        g = penman.decode(amr)
    except Exception:
        return []
    concepts = {
        s: str(t) for s, r, t in g.triples if r in (":instance", "instance")
    }
    names: dict[str, str] = {}
    for s, r, t in g.triples:
        if r != ":legal-name":
            continue
        try:
            names[str(s)] = json.loads(t) if isinstance(t, str) else str(t)
        except Exception:
            names[str(s)] = str(t).strip('"')
    out = []
    for var, name in names.items():
        concept = concepts.get(var, "")
        if concept in LEGAL_CONCEPTS:
            out.append((var, concept, name))
    return out


def specific_names(amr: str) -> set[str]:
    return {name for _, _, name in _extract_specific_names(amr)}


def _name_key(name: str) -> str:
    """Compare names without the leading article or case differences."""
    key = _norm(name)
    return key[4:] if key.startswith("the ") else key


def validate_rewrite(original: str, sentence: str, amr: str,
                     edits: list[dict[str, Any]], cfg: dict[str, Any]
                     ) -> tuple[bool, list[str], dict[str, Any]]:
    """Check AMR validity and the graph↔sentence rewrite invariant."""
    errors: list[str] = []
    ok_core, core_errors, core_info = validate_amr(amr)
    errors.extend(core_errors)
    ok_schema, schema_errors, schema_info = check_legal_amr(amr)
    errors.extend(schema_errors)

    # Recompute the sentence from the original + spans.
    rewritten, located, edit_errors = apply_edits(original, edits, cfg)
    errors.extend(edit_errors)
    if rewritten is not None and rewritten != sentence:
        errors.append("sentence is not equal to original with the declared edits")

    names = specific_names(amr)
    name_keys = {_name_key(name) for name in names}
    replaced = {_name_key(e.replacement) for e in located}
    orig_norm = _name_key(original)
    if cfg.get("rewrite", {}).get("enabled", True):
        for var, concept, name in _extract_specific_names(amr):
            if (name and _name_key(name) in _name_key(sentence)
                    and _name_key(name) not in orig_norm):
                if _name_key(name) not in replaced:
                    errors.append(
                        f"specific name {name!r} appears in sentence/amr "
                        "but is not backed by an edit"
                    )
        for edit in located:
            if (_name_key(edit.replacement) not in name_keys
                    and _name_key(edit.replacement) not in _name_key(amr)):
                errors.append(
                    f"edit introduces {edit.replacement!r} "
                    "but no matching specific name exists in the graph"
                )
            # Credibility gate: the agent must declare where the name came from.
            if edit.evidence not in {
                "article", "metadata", "ledger", "wiki", "context"
            }:
                errors.append(
                    f"edit for {edit.replacement!r} has no credible evidence source"
                )

    info = {
        "core": core_info, "schema": schema_info,
        "edits": [e.to_record() for e in located],
        "specific_names": sorted(names),
    }
    return len(errors) == 0 and ok_core and ok_schema, errors, info

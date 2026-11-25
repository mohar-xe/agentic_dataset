#!/usr/bin/env python3
"""Mechanical schema checker for Legal-AMR Lean Core v1.1.

The tagset is the one in ``extension/Legal-AMR-Lean-Core-v1.1.md``:

* 7 legal concepts, 12 legal relations, 11 legal attributes (+1 optional
  edge attribute, ``:legal-weight``);
* mandatory ``:legal-`` namespacing and additive legal tags;
* the three essential structural checks:
  - ``:legal-level`` only on ``provision``;
  - ``:legal-type`` only on ``gov-entity``;
  - no ``:ARGn`` (n = 0..5) on the bare-noun legal concepts (link through
    ``:ARGn-of`` / the reverse form instead);
* relation domain/range, attribute value types, a single root, and an acyclic
  graph.

The checker is deliberately mechanical. Warnings are emitted for things that
are legal-AMR smells (legacy sub-type concepts, missing sub-typing attributes)
but that do not make the graph invalid core AMR. Errors make the graph fail
the schema.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import date
from typing import Any

from common import top_level_forms

try:
    import penman
except ImportError:  # pragma: no cover - only before dependencies are installed
    penman = None

LEGAL_CONCEPTS = frozenset({
    "statute",
    "regulation",
    "provision",
    "gov-entity",
    "judgment",
    "notification",
    "legal-instrument",
})

# v1.1: 12 relations. The four v1.0 relations below were pruned by the
# necessity test and are reported with a migration hint, not silently accepted.
LEGAL_RELATIONS = frozenset({
    # referential
    ":legal-cites",
    ":legal-overrules",
    # normative
    ":legal-amends",
    ":legal-repeals",
    ":legal-notifies",
    ":legal-implements",
    ":legal-applies-to",
    ":legal-exempts",
    # authority & competence
    ":legal-empowers",
    ":legal-delegates-to",
    ":legal-administered-by",
    # definitional
    ":legal-defines",
})

PRUNED_RELATIONS = {
    ":legal-relies-on": (
        "pruned in v1.1; use :legal-cites (+ optional :legal-weight \"binding\")"
    ),
    ":legal-governs": (
        "pruned in v1.1; use :legal-applies-to"
    ),
    ":legal-part-of": (
        "pruned in v1.1; use core AMR :part-of"
    ),
    ":legal-antecedent": (
        "pruned in v1.1; use core AMR reentrancy (the same variable)"
    ),
}

LEGAL_ATTRIBUTES = frozenset({
    ":legal-name",
    ":legal-level",
    ":legal-type",
    ":legal-year",
    ":legal-jurisdiction",
    ":legal-citation",
    ":legal-status",
    ":legal-valid-from",
    ":legal-valid-until",
    ":legal-enacted-on",
    ":legal-notified-on",
})

# v1.1 keeps only the citation-strength edge attribute. ``:legal-assented-on``
# was pruned because assent and enactment share a date in Indian law.
OPTIONAL_ATTRIBUTES = frozenset({
    ":legal-weight",
})

LEGAL_LEVELS = frozenset({
    "section", "article", "clause", "sub-section",
    "schedule", "chapter", "part",
})

LEGAL_TYPES = frozenset({
    "ministry", "department", "court", "tribunal",
    "statutory-body", "state-government", "local-authority",
})

LEGAL_WEIGHTS = frozenset({"binding", "persuasive", "obiter"})

LEGAL_STATUSES = frozenset({
    "in-force", "repealed", "struck-down", "amended", "not-yet-in-force",
})

# The v1.1 range of :legal-cites is {provision, statute, judgment, regulation}.
CITES_TARGETS = frozenset({"provision", "statute", "judgment", "regulation"})
# v1.1 instrument-to-instrument edges use the instrument concepts.
AMENDS_TARGETS = frozenset({"statute", "regulation", "provision"})
REPEALS_TARGETS = frozenset({"statute", "regulation"})
NOTIFIES_TARGETS = frozenset({"notification", "regulation"})

# Concepts that the lean plan folds away. They are still valid core AMR, so
# they are warnings rather than errors, but the graph should prefer
# provision/gov-entity plus a sub-typing attribute. ``part`` is excluded: it is
# an everyday core AMR concept ("part of the answer"), not only a legal level.
LEGACY_CONCEPTS = frozenset({
    "section", "article", "clause", "sub-section", "schedule", "chapter",
    "ministry", "department", "court", "tribunal", "statutory-body",
    "state-government", "local-authority",
})

# Old, pre-lean extension roles. Unknown :legal-* roles are already errors;
# this list catches the same concepts written without the mandatory prefix.
LEGACY_ROLES = frozenset({
    # pre-lean prefixed roles
    ":legal-refers-to", ":legal-legal-ref", ":legal-distinguishes",
    ":legal-supersedes", ":legal-derogates", ":legal-harmonizes-with",
    ":legal-competent-authority", ":legal-presided-over-by",
    ":legal-exercises-jurisdiction-over", ":legal-has-provision",
    ":legal-has-schedule", ":legal-sub-section-of", ":legal-clause-of",
    ":legal-precedes", ":legal-version-of", ":legal-defined-in",
    # extension semantics written without the mandatory :legal- prefix
    ":cites", ":refers-to", ":amends", ":repeals", ":notifies",
    ":implements", ":governs", ":applies-to", ":exempts", ":empowers",
    ":delegates-to", ":administered-by", ":antecedent", ":defines",
    ":part-of-legal",
    # non-namespaced metadata
    ":act-number", ":gazette-ref", ":bench-strength",
})

# A mention that names no instrument ("the Act", "the said Rules"). It is a
# reference, not an instrument: resolve it with core AMR reentrancy, not a
# dedicated :legal-antecedent edge.
UNDER_SPECIFIED_NAMES = frozenset({
    "act", "acts", "code", "codes", "rules", "statute", "statutes",
    "regulation", "regulations", "ordinance", "ordinances", "law", "laws",
    "said act", "said code", "said rules", "said statute",
    "aforementioned act", "aforesaid act", "above act",
})

# Concepts that name a specific instrument, so a generic mention typed as one
# of them is a mistake.
SPECIFIC_INSTRUMENTS = frozenset({
    "statute", "regulation", "notification", "judgment",
})

ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
INTEGER_RE = re.compile(r"^-?\d+$")
ARG_ROLE_RE = re.compile(r"^:ARG[0-9]+$")


def is_under_specified(name: str) -> bool:
    """Whether a name is a mention of an instrument rather than a name."""
    text = re.sub(r"\s+", " ", str(name).strip().lower())
    text = re.sub(r"^(the|a|an)\s+", "", text)
    return text in UNDER_SPECIFIED_NAMES


def _is_iso_date(value: Any) -> bool:
    text = str(value)
    if not ISO_DATE_RE.match(text):
        return False
    try:
        date.fromisoformat(text)
    except ValueError:
        return False
    return True


def _literal_kind(target: str) -> tuple[str, Any]:
    """Classify a PENMAN triple target as string, integer, or variable."""
    target = str(target)
    if target.startswith('"') and target.endswith('"'):
        try:
            return "string", json.loads(target)
        except json.JSONDecodeError:
            return "string", target[1:-1]
    if INTEGER_RE.match(target):
        return "integer", int(target)
    return "variable", target


def _concept(graph_concepts: dict[str, str], var: str) -> str | None:
    return graph_concepts.get(var)


def _has_attr(attributes: list[tuple[str, str, str]], var: str, role: str) -> bool:
    return any(s == var and r == role for s, r, _ in attributes)


def check_legal_amr(amr: str | None) -> tuple[bool, list[str], dict[str, Any]]:
    """Validate one graph against Legal-AMR Lean Core v1.1.

    Returns ``(ok, errors, info)``. ``info`` contains the usual graph counts
    plus ``warnings`` and legal-tag inventories, and is populated even when
    schema checks fail.
    """
    errors: list[str] = []
    warnings: list[str] = []
    info: dict[str, Any] = {}

    if not amr or not isinstance(amr, str) or not amr.strip():
        return False, ["empty graph"], info
    if penman is None:
        return False, ["penman is not installed (pip install penman)"], info

    forms, depth = top_level_forms(amr.strip())
    if forms != 1:
        return False, [
            "graph must be single-rooted: exactly one top-level form; "
            f"found {forms}"
        ], info
    if depth != 0:
        return False, [
            f"graph has unbalanced parentheses (final depth {depth})"
        ], info

    try:
        graph = penman.decode(amr.strip())
    except Exception as exc:  # noqa: BLE001 - penman raises several types
        return False, [f"PENMAN parse failed: {exc}"], info

    concepts: dict[str, str] = {}
    edges: list[tuple[str, str, str]] = []
    attributes: list[tuple[str, str, str]] = []

    for source, role, target in graph.triples:
        if role in (":instance", "instance"):
            concepts[str(source)] = str(target)
            continue
        kind, _ = _literal_kind(target)
        if kind == "variable":
            edges.append((str(source), str(role), str(target)))
        else:
            attributes.append((str(source), str(role), str(target)))

    # --- namespacing and tagset membership -------------------------------- #
    for _, role, _ in edges:
        if role.startswith(":legal") and role not in LEGAL_RELATIONS:
            if role in PRUNED_RELATIONS:
                errors.append(
                    f"unknown legal relation {role!r}: {PRUNED_RELATIONS[role]}"
                )
            else:
                errors.append(f"unknown legal relation {role!r}")
        if role in LEGACY_ROLES:
            warnings.append(f"legacy relation {role!r}; use the lean :legal-* role")

    for _, role, _ in attributes:
        if role.startswith(":legal") and role not in LEGAL_ATTRIBUTES \
                and role not in OPTIONAL_ATTRIBUTES:
            errors.append(f"unknown legal attribute {role!r}")
        if role in LEGACY_ROLES:
            warnings.append(f"legacy attribute {role!r}; use the lean :legal-* attribute")

    for _, role, target in edges:
        if role in LEGAL_ATTRIBUTES or role in OPTIONAL_ATTRIBUTES:
            errors.append(f"{role} must be a literal attribute, not an edge")
    for source, role, target in attributes:
        if role in LEGAL_RELATIONS:
            # v1.1 permits :legal-defines to point at a string term.
            if role == ":legal-defines":
                continue
            errors.append(f"{role} must point to a node, not literal {target}")

    # --- single root / connectivity --------------------------------------- #
    adjacency: dict[str, set[str]] = {var: set() for var in concepts}
    for source, _, target in edges:
        if source in adjacency and target in concepts:
            adjacency[source].add(target)
            adjacency[target].add(source)
    top = str(graph.top)
    if top not in adjacency:
        top = next(iter(concepts), "")
    reached: set[str] = set()
    if top:
        stack = [top]
        while stack:
            node = stack.pop()
            if node in reached:
                continue
            reached.add(node)
            stack.extend(adjacency.get(node, ()))
    n_components = 0
    unseen = set(concepts)
    while unseen:
        n_components += 1
        stack = [unseen.pop()]
        while stack:
            node = stack.pop()
            for neighbour in adjacency.get(node, ()):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
    roots = sorted(set(concepts) - reached)
    if reached != set(concepts):
        errors.append(
            "graph must be single-rooted and connected; "
            f"unreachable nodes from top {top!r}: {roots}"
        )

    # --- acyclicity -------------------------------------------------------- #
    adjacency_directed: dict[str, list[str]] = {var: [] for var in concepts}
    for source, _, target in edges:
        if source in adjacency_directed and target in concepts:
            adjacency_directed[source].append(target)
    colour: dict[str, int] = {}

    def visit(node: str) -> bool:
        colour[node] = 1
        for neighbour in adjacency_directed.get(node, ()):
            state = colour.get(neighbour, 0)
            if state == 1 or (state == 0 and visit(neighbour)):
                return True
        colour[node] = 2
        return False

    if any(colour.get(node, 0) == 0 and visit(node)
           for node in concepts):
        errors.append("graph contains a cycle (AMR must be acyclic)")

    # --- concept checks ---------------------------------------------------- #
    for var, concept in concepts.items():
        if concept.startswith("legal-") and concept not in LEGAL_CONCEPTS:
            errors.append(
                f"{var}: concept {concept!r} is not one of the 7 legal concepts"
            )
        if concept in LEGACY_CONCEPTS:
            warnings.append(
                f"{var}: legacy concept {concept!r}; prefer provision/gov-entity "
                "with :legal-level/:legal-type"
            )

    # --- no :ARGn on bare-noun legal concepts (v1.1 essential check) ------ #
    # :ARGn belongs on a PropBank predicate (lemma-NN). A judgment/statute is
    # a noun; "the court rendered the judgment" is expressed through
    # :ARG1-of (render-01 :ARG0 (court)), which penman decodes as an :ARG1
    # edge on render-01, not on the noun.
    for source, role, target in edges:
        if not ARG_ROLE_RE.match(role):
            continue
        concept = _concept(concepts, source)
        if concept in LEGAL_CONCEPTS or concept in LEGACY_CONCEPTS:
            errors.append(
                f"{source}: {role} on the bare-noun concept {concept!r}; "
                "link through a predicate with :ARGn-of (for example "
                ":ARG1-of (r / render-01 :ARG0 ...))"
            )

    # --- relation domain/range -------------------------------------------- #
    relation_counts: Counter[str] = Counter()
    cites_sources: set[str] = set()

    def fail(role: str, message: str) -> None:
        errors.append(f"{role}: {message}")

    for source, role, target in edges:
        if role not in LEGAL_RELATIONS:
            continue
        relation_counts[role] += 1
        src = _concept(concepts, source)
        tgt = _concept(concepts, target)
        if target not in concepts:
            fail(role, f"target {target!r} is not a defined node")
            continue

        if role == ":legal-cites":
            cites_sources.add(source)
            # v1.1 direction is "any -> {provision, statute, judgment,
            # regulation}": a verb may be the source ("challenged under
            # Section 34"), so only the target is constrained.
            if tgt not in CITES_TARGETS:
                fail(role, f"target concept {tgt!r} is not citable "
                           f"({sorted(CITES_TARGETS)})")
        elif role == ":legal-overrules":
            if src != "judgment" or tgt != "judgment":
                fail(role, f"both endpoints must be judgment, got {src!r} -> {tgt!r}")
        elif role == ":legal-amends":
            if src not in AMENDS_TARGETS or tgt not in AMENDS_TARGETS:
                fail(role, f"both endpoints must be statute/regulation/provision, "
                           f"got {src!r} -> {tgt!r}")
        elif role == ":legal-repeals":
            if src not in REPEALS_TARGETS or tgt not in REPEALS_TARGETS:
                fail(role, f"both endpoints must be statute/regulation, "
                           f"got {src!r} -> {tgt!r}")
        elif role == ":legal-notifies":
            if src != "gov-entity":
                fail(role, f"source must be gov-entity, got {src!r}")
            if tgt not in NOTIFIES_TARGETS:
                fail(role, f"target must be notification/regulation, got {tgt!r}")
        elif role == ":legal-implements":
            if src != "regulation":
                fail(role, f"source must be regulation, got {src!r}")
            if tgt != "statute":
                fail(role, f"target must be statute, got {tgt!r}")
        elif role in {":legal-applies-to", ":legal-exempts"}:
            if src != "provision":
                fail(role, f"source must be provision, got {src!r}")
        elif role == ":legal-empowers":
            if src != "provision":
                fail(role, f"source must be provision, got {src!r}")
            if tgt != "gov-entity":
                fail(role, f"target must be gov-entity, got {tgt!r}")
        elif role == ":legal-delegates-to":
            if src != "gov-entity" or tgt != "gov-entity":
                fail(role, f"both endpoints must be gov-entity, got {src!r} -> {tgt!r}")
        elif role == ":legal-administered-by":
            if src != "statute":
                fail(role, f"source must be statute, got {src!r}")
            if tgt != "gov-entity":
                fail(role, f"target must be gov-entity, got {tgt!r}")
        elif role == ":legal-defines":
            # v1.1 §3.4.1 states provision -> {concept, string}. The worked
            # example in §6.6 also places the edge on the statute that owns
            # the definitional provision, so a non-provision source is a
            # warning rather than a hard failure.
            if src != "provision":
                warnings.append(
                    f"{role}: source {src!r} is not a provision; prefer the "
                    "definitional provision as the source"
                )

    # --- attribute types and applicability -------------------------------- #
    attribute_counts: Counter[str] = Counter()
    for source, role, target in attributes:
        if role not in LEGAL_ATTRIBUTES and role not in OPTIONAL_ATTRIBUTES:
            continue
        attribute_counts[role] += 1
        kind, value = _literal_kind(target)
        concept = _concept(concepts, source)

        if role == ":legal-defines" and kind == "string":
            # Allowed by v1.1: provision -> {concept, string}.
            if concept != "provision":
                warnings.append(
                    ":legal-defines: source is not a provision; prefer the "
                    f"definitional provision (got {concept!r})"
                )
            continue

        if kind == "variable":
            errors.append(f"{role} must be a literal, not node {value!r}")
            continue

        if role == ":legal-name":
            if kind != "string" or not str(value).strip():
                errors.append(f"{role} must be a non-empty string")
            elif concept not in LEGAL_CONCEPTS:
                warnings.append(f"{role} on {concept!r} node {source!r}")
        elif role == ":legal-level":
            if concept != "provision":
                errors.append(f"{role} may appear only on a provision, got {concept!r}")
            if kind != "string":
                errors.append(f"{role} must be a string")
            elif value not in LEGAL_LEVELS:
                errors.append(
                    f"{role} value {value!r} is not one of {sorted(LEGAL_LEVELS)}"
                )
        elif role == ":legal-type":
            if concept != "gov-entity":
                errors.append(f"{role} may appear only on a gov-entity, got {concept!r}")
            if kind != "string":
                errors.append(f"{role} must be a string")
            elif value not in LEGAL_TYPES:
                errors.append(
                    f"{role} value {value!r} is not one of {sorted(LEGAL_TYPES)}"
                )
        elif role == ":legal-year":
            if kind != "integer":
                errors.append(f"{role} must be an integer")
            elif concept not in {"statute", "judgment", "regulation"}:
                warnings.append(f"{role} on {concept!r} node {source!r}")
        elif role == ":legal-jurisdiction":
            if kind != "string":
                errors.append(f"{role} must be a string")
            elif concept not in {"statute", "gov-entity", "regulation"}:
                warnings.append(f"{role} on {concept!r} node {source!r}")
        elif role == ":legal-citation":
            if kind != "string":
                errors.append(f"{role} must be a string")
            elif concept != "judgment":
                warnings.append(f"{role} on {concept!r} node {source!r}")
        elif role == ":legal-status":
            if kind != "string":
                errors.append(f"{role} must be a string")
            elif value not in LEGAL_STATUSES:
                errors.append(
                    f"{role} value {value!r} is not one of {sorted(LEGAL_STATUSES)}"
                )
            if concept not in {"statute", "provision", "regulation"}:
                warnings.append(f"{role} on {concept!r} node {source!r}")
        elif role in {":legal-valid-from", ":legal-valid-until",
                      ":legal-enacted-on", ":legal-notified-on"}:
            if kind != "string" or not _is_iso_date(value):
                errors.append(f"{role} must be an ISO date string YYYY-MM-DD")
            if role == ":legal-enacted-on" and concept != "statute":
                warnings.append(f"{role} on {concept!r} node {source!r}")
            elif role == ":legal-notified-on" and concept != "notification":
                warnings.append(f"{role} on {concept!r} node {source!r}")
        elif role == ":legal-weight":
            if kind != "string":
                errors.append(f"{role} must be a string")
            elif value not in LEGAL_WEIGHTS:
                errors.append(
                    f"{role} value {value!r} is not one of {sorted(LEGAL_WEIGHTS)}"
                )
            if source not in cites_sources:
                warnings.append(
                    f"{role} on {source!r}, which is not the source of a :legal-cites edge"
                )

    # --- core AMR ":wiki" consistency (warnings) -------------------------- #
    for source, role, target in attributes:
        if role != ":wiki":
            continue
        concept = _concept(concepts, source)
        if not any(s == source and r == ":name" for s, r, _ in edges) and not _has_attr(
            attributes, source, ":legal-name"
        ):
            warnings.append(
                f"{source}: :wiki {target} but the node has no :name "
                "(the wiki title should accompany the surface name)"
            )
        if concept in LEGAL_CONCEPTS and not _has_attr(
            attributes, source, ":legal-name"
        ):
            warnings.append(
                f"{source}: :wiki on {concept!r} without :legal-name; legal "
                "entities should carry :legal-name"
            )

    # --- required sub-typing (warnings; values are checked above) ---------- #
    for var, concept in concepts.items():
        if concept == "provision" and not _has_attr(attributes, var, ":legal-level"):
            warnings.append(f"{var}: provision has no :legal-level")
        if concept == "gov-entity" and not _has_attr(attributes, var, ":legal-type"):
            warnings.append(f"{var}: gov-entity has no :legal-type")

    # --- under-specified mentions (warnings) ------------------------------- #
    # "the Act" is a reference to an instrument, not the name of one. v1.1
    # resolves it with core AMR reentrancy: use the same variable as the
    # antecedent instead of a :legal-antecedent edge.
    for source, role, target in attributes:
        if role != ":legal-name":
            continue
        concept = _concept(concepts, source)
        if concept not in SPECIFIC_INSTRUMENTS:
            continue
        _kind, value = _literal_kind(target)
        if is_under_specified(value):
            warnings.append(
                f"{source}: {concept} is named {value!r}, which names no "
                "instrument; resolve the mention with core AMR reentrancy "
                "(the same variable as the antecedent)"
            )

    info = {
        "n_roots": n_components,
        "legal_nodes": sum(1 for c in concepts.values() if c in LEGAL_CONCEPTS),
        "legal_concepts": sorted({c for c in concepts.values() if c in LEGAL_CONCEPTS}),
        "legal_relations": dict(relation_counts.most_common()),
        "legal_attributes": dict(attribute_counts.most_common()),
        "legal_tag_count": sum(relation_counts.values()) + sum(attribute_counts.values()),
        "warnings": warnings,
    }
    return not errors, errors, info

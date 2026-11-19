"""One place to validate an agent-produced graph.

Wraps the reused core/schema checkers and adds the rewrite invariant. The
result is the object the agent gets back from the ``check_amr`` tool and the
object the runner records in the event log.
"""

from __future__ import annotations

from typing import Any

from common import coverage as _coverage
from rewrite import validate_rewrite


def check_graph(original: str, sentence: str, amr: str,
                edits: list[dict[str, Any]], cfg: dict) -> dict[str, Any]:
    ok, errors, info = validate_rewrite(original, sentence, amr, edits or [], cfg)
    return {
        "ok": ok,
        "errors": errors,
        "n_errors": len(errors),
        "coverage": round(_coverage(sentence, amr), 3) if amr else None,
        "legal_concepts": (info.get("schema") or {}).get("legal_concepts", []),
        "legal_relations": (info.get("schema") or {}).get("legal_relations", {}),
        "edits": info.get("edits", []),
    }

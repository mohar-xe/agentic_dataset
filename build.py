"""Write the agent output as dataset JSONL and PENMAN files.

``sentence`` (resolved) is primary; ``sentence_original`` is always kept.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def _records(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def penman_block(record: dict) -> str:
    sentence = " ".join((record.get("sentence") or "").split())
    original = " ".join((record.get("sentence_original") or sentence).split())
    lines = [f"# ::id {record['id']}"]
    if record.get("sentence_id") is not None:
        lines.append(f"# ::sentence_id {record['sentence_id']}")
    lines.append(f"# ::snt {sentence}")
    lines.append(f"# ::tok {sentence}")
    if original != sentence:
        lines.append(f"# ::snt_orig {original}")
        edits = record.get("edits") or []
        if edits:
            lines.append("# ::edits " + json.dumps(edits, ensure_ascii=False))
    lines.append(f"# ::source {record.get('source','')}")
    if record.get("article_id") is not None:
        lines.append(f"# ::article {record['article_id']}")
    lines.append(record["amr"])
    return "\n".join(lines) + "\n\n"


def build(out: Path, prefix: Path | None = None) -> dict:
    records = [r for r in _records(out / "records.jsonl") if r.get("status") == "ok"]
    records.sort(key=lambda r: (r.get("article_id", 0), r.get("sentence_id", 0)))
    if prefix is None:
        prefix = out / "dataset" / "agent_amr"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with (out / "dataset.all.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "dataset.all.amr").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(penman_block(r))
    return {"records": len(records), "jsonl": str(out / "dataset.all.jsonl"),
            "amr": str(out / "dataset.all.amr")}

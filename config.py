"""Configuration and shared paths for the per-article AMR agent.

Thin: one default dict, one JSON file, CLI overrides. No framework.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
LEGACY_SCRIPTS = ROOT.parent / "dataset_amr" / "scripts"
ROOT_OUT = ROOT / "data" / "agent"

# The old project is a read-only source of the tested schema checker and the
# AMR helpers. It is imported by path, never copied.
if str(LEGACY_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(LEGACY_SCRIPTS))

DEFAULTS: dict[str, Any] = {
    "corpus_db": str(ROOT.parent / "dataset_amr" / "data" / "legal.db"),
    "base_graphs": str(
        ROOT.parent / "dataset_amr" / "data" / "work" / "amr_spring.jsonl"
    ),
    "out": str(ROOT_OUT),
    "limit": 0,
    "min_sentences": 5,
    "workers": 1,
    "model": os.environ.get("AMR_MODEL", ""),
    "llm": {
        "api_url": os.environ.get("AMR_API_URL", ""),
        "api_key": os.environ.get("AMR_API_KEY", ""),
        "timeout": 180,
        "retries": 3,
        "temperature": 0.0,
        "max_tokens": 6000,
    },
    "agent": {
        "batch_size": 8,
        "max_tool_rounds": 6,
        "max_repairs": 2,
    },
    "rewrite": {
        "enabled": True,
        # "all" or a list such as ["statute", "provision", "gov-entity"].
        "kinds": "all",
        "on_ambiguous": "reject",
        "max_per_sentence": 4,
        "max_growth_chars": 160,
    },
    "wiki": {
        # Live MediaWiki / Wikidata only. No local title database.
        "enabled": True,
        "lang": "en",
        "timeout": 10,
        "max_results": 5,
        "cache": str(ROOT_OUT / "wiki_cache.sqlite"),
    },
    "seed": 13,
    "train": 0.8,
    "dev": 0.1,
    # Reuse the older pipeline's OpenRouter/SPRING/DB settings when present.
    "inherit_old_config": True,
    "old_config": str(ROOT.parent / "dataset_amr" / "pipeline.config.json"),
    "old_project": str(ROOT.parent / "dataset_amr"),
}

# Optional sources an agent may rewrite from. The validator uses this to keep
# the model from inventing a name.
CREDIBLE_SOURCES = frozenset({"article", "metadata", "ledger", "wiki"})


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def inherited_config(path: str | Path | None) -> dict:
    """Map the older pipeline's config onto this agent's config.

    Only settings that still make sense here are inherited: the LLM endpoint /
    key / model (OpenRouter by default) and the base-graph/DB locations.
    A missing file is not an error.
    """
    if not path:
        return {}
    file = Path(path)
    if not file.exists():
        return {}
    try:
        old = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    llm = old.get("llm") or {}
    exported: dict[str, Any] = {}
    llm_map = {
        key: llm[key] for key in ("api_url", "api_key", "timeout", "retries")
        if llm.get(key) not in (None, "")
    }
    if llm_map:
        exported["llm"] = llm_map
    if llm.get("model"):
        exported["model"] = llm["model"]
    if (old.get("export") or {}).get("limit"):
        exported["limit"] = old["export"]["limit"]
    spring = old.get("spring") or {}
    if spring.get("url"):
        exported["spring_url"] = spring["url"]
    old_root = file.parent
    for key, rel in (
        ("corpus_db", "data/legal.db"),
        ("base_graphs", "data/work/amr_spring.jsonl"),
    ):
        candidate = old_root / rel
        if candidate.exists():
            exported[key] = str(candidate)
    return exported


def _apply_env(cfg: dict) -> dict:
    """Environment wins over config files, as in the old pipeline."""
    if os.environ.get("AMR_API_URL"):
        cfg["llm"]["api_url"] = os.environ["AMR_API_URL"]
    if os.environ.get("AMR_API_KEY"):
        cfg["llm"]["api_key"] = os.environ["AMR_API_KEY"]
    if os.environ.get("AMR_MODEL"):
        cfg["model"] = os.environ["AMR_MODEL"]
    return cfg


def load_config(path: str | Path | None = None,
                overrides: dict[str, Any] | None = None) -> dict:
    """Defaults -> older pipeline config -> local config -> env -> overrides."""
    cfg = copy.deepcopy(DEFAULTS)
    local: dict[str, Any] = {}
    if path:
        file = Path(path)
        if not file.exists():
            raise SystemExit(f"config not found: {file}")
        local = json.loads(file.read_text(encoding="utf-8"))
    elif (ROOT / "config.json").exists():
        local = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    inherit = overrides.get(
        "inherit_old_config", local.get(
            "inherit_old_config", cfg["inherit_old_config"])
    )
    old_path = overrides.get("old_config", local.get(
        "old_config", cfg["old_config"]))
    if inherit:
        cfg = _merge(cfg, inherited_config(old_path))
    cfg = _merge(cfg, local)
    cfg = _apply_env(cfg)
    cfg = _merge(cfg, overrides)
    return cfg


def out_dir(cfg: dict) -> Path:
    path = Path(cfg["out"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def article_dir(cfg: dict) -> Path:
    path = out_dir(cfg) / "articles"
    path.mkdir(parents=True, exist_ok=True)
    return path


def rewrite_kinds(cfg: dict) -> str | list[str]:
    return cfg["rewrite"]["kinds"]


def kind_allowed(cfg: dict, kind: str) -> bool:
    """Whether a resolved entity of ``kind`` may rewrite the sentence."""
    if not cfg["rewrite"]["enabled"]:
        return False
    kinds = cfg["rewrite"]["kinds"]
    if kinds == "all":
        return True
    return kind in kinds

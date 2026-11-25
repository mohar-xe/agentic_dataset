"""Configuration and shared paths for the per-article AMR agent.

Thin: one default dict, one JSON file, CLI overrides. No framework and no
dependency on the older ``dataset_amr`` project: all inputs live under this
directory (or are passed explicitly on the command line).
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
ROOT_OUT = ROOT / "data" / "agent"

DEFAULTS: dict[str, Any] = {
    # India-only corpus (Indian sources; verfassungsblog/German material is
    # filtered out). Build it with ``python make_india_db.py``.
    "corpus_db": str(ROOT / "data" / "india.db"),
    # Optional warm cache for spring_parse. The agent always calls the tool;
    # a cached graph only saves a model call.
    "base_graphs": str(ROOT / "data" / "amr_spring.jsonl"),
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
        # Wikimedia asks for a contactable User-Agent and <=3 concurrent
        # requests; the 2026 limit for UA-only clients is 200 req/min.
        "user_agent": "",
        "concurrent": 3,
        "max_per_minute": 180,
        "retries": 2,
        "cache": str(ROOT_OUT / "wiki_cache.sqlite"),
    },
    "seed": 13,
    "train": 0.8,
    "dev": 0.1,
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


def _apply_env(cfg: dict) -> dict:
    """Environment wins over config files."""
    if os.environ.get("AMR_API_URL"):
        cfg["llm"]["api_url"] = os.environ["AMR_API_URL"]
    if os.environ.get("AMR_API_KEY"):
        cfg["llm"]["api_key"] = os.environ["AMR_API_KEY"]
    if os.environ.get("AMR_MODEL"):
        cfg["model"] = os.environ["AMR_MODEL"]
    return cfg


def load_config(path: str | Path | None = None,
                overrides: dict[str, Any] | None = None) -> dict:
    """Defaults -> local config.json -> environment -> overrides."""
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

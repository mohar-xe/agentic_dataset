"""SPRING as a tool: one sentence in, one PENMAN base graph out.

The agent no longer receives a pre-built base graph. It receives paragraphs,
splits them itself, and calls ``spring_parse`` once per sentence. Each call
returns the corpus ``sentence_id`` and the SPRING graph, which is the crutch
the agent then edits.

The model is local (amrlib, SPRING v0.1.0, CPU) and loaded once per process.
amrlib's parser is not thread-safe, so one lock guards the model: the agent may
issue several ``spring_parse`` calls in the same turn (the tool loop runs them
concurrently), and each one still gets a correct graph. Parsing one sentence is
fast, so the lock is not the bottleneck it would be for a batch of paragraphs.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import config  # noqa: F401  (puts the legacy scripts on sys.path)
from common import validate_amr

# Local SPRING checkpoint shipped with the amrlib setup used by the older
# pipeline. Nothing is downloaded here.
DEFAULT_MODEL_DIR = (
    Path.home() / ".cache" / "amrlib" / "model_parse_spring-v0_1_0"
)
MODEL_NAME = "spring-v0_1_0"
# Only used by --mock when amrlib is absent: keeps the walkthrough runnable
# without pretending a real graph was parsed.
PLACEHOLDER_AMR = "(z / thing)"


class SpringParser:
    """Lazy, thread-safe wrapper around one local SPRING model."""

    def __init__(self, cache: dict[int, str] | None = None,
                 model_dir: str | Path | None = None, device: str = "cpu",
                 beam_size: int = 5,
                 allow_placeholder: bool = False) -> None:
        self.cache: dict[int, str] = dict(cache or {})
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self.device = device
        self.beam_size = beam_size
        self.allow_placeholder = allow_placeholder
        self._model = None
        self._error: str | None = None
        self._lock = threading.Lock()
        self.calls = 0
        self.cache_hits = 0

    # ------------------------------------------------------------- loading
    def _load(self):
        if self._model is not None:
            return self._model
        try:
            import amrlib
        except Exception as exc:  # missing package, not a crash
            self._error = f"amrlib is not importable: {exc}"
            return None
        if not self.model_dir.exists():
            self._error = f"SPRING model not found at {self.model_dir}"
            return None
        try:
            self._model = amrlib.load_stog_model(
                model_dir=str(self.model_dir), device=self.device,
                batch_size=1, beam_size=self.beam_size,
            )
        except Exception as exc:
            self._error = f"SPRING failed to load: {exc}"
            return None
        return self._model

    def load(self) -> None:
        """Load the model now (a real run should fail early, not per batch)."""
        with self._lock:
            if self._load() is None:
                raise RuntimeError(self._error or "SPRING unavailable")

    def status(self) -> dict:
        return {
            "available": self._model is not None or (
                self._error is None and self.model_dir.exists()),
            "loaded": self._model is not None,
            "model": MODEL_NAME,
            "model_dir": str(self.model_dir),
            "error": self._error,
            "cached": len(self.cache),
            "calls": self.calls,
        }

    # ------------------------------------------------------------- parsing
    def parse(self, sentence: str, sentence_id: int | None = None) -> dict:
        """Parse ONE sentence. Returns the graph plus its own validity."""
        sentence = " ".join((sentence or "").split())
        if not sentence:
            return {"sentence_id": sentence_id, "base_amr": None,
                    "status": "error", "errors": ["empty sentence"],
                    "parser": "spring", "model": MODEL_NAME}
        if sentence_id is not None and sentence_id in self.cache:
            self.cache_hits += 1
            return self._checked(self.cache[sentence_id], sentence_id,
                                 seconds=0.0, source="cache")
        with self._lock:
            model = self._load()
            if model is None:
                if self.allow_placeholder:  # --mock without amrlib installed
                    result = {
                        "sentence_id": sentence_id,
                        "base_amr": PLACEHOLDER_AMR,
                        "status": "unavailable",
                        "errors": [self._error or "SPRING unavailable"],
                        "n_nodes": None, "n_edges": None,
                        "parser": "spring", "model": MODEL_NAME,
                        "source": "placeholder",
                    }
                    if sentence_id is not None:
                        self.cache[sentence_id] = PLACEHOLDER_AMR
                    return result
                return {"sentence_id": sentence_id, "base_amr": None,
                        "status": "unavailable",
                        "errors": [self._error or "SPRING unavailable"],
                        "parser": "spring", "model": MODEL_NAME}
            started = time.time()
            try:
                graph = model.parse_sents([sentence], add_metadata=False)[0]
            except Exception as exc:
                return {"sentence_id": sentence_id, "base_amr": None,
                        "status": "error",
                        "errors": [f"SPRING failed: {exc}"],
                        "parser": "spring", "model": MODEL_NAME}
            seconds = time.time() - started
        self.calls += 1
        result = self._checked(str(graph), sentence_id, seconds=seconds,
                               source="model")
        if sentence_id is not None and result.get("base_amr"):
            self.cache[sentence_id] = result["base_amr"]
        return result

    def parse_many(self, sentences: list[str], sentence_ids: list[int] | None = None,
                   workers: int = 4) -> list[dict]:
        """Parse several sentences with a small pool (used for prewarming)."""
        from concurrent.futures import ThreadPoolExecutor

        ids = sentence_ids or [None] * len(sentences)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(self.parse, text, sid)
                       for text, sid in zip(sentences, ids)]
            return [f.result() for f in futures]

    @staticmethod
    def _checked(amr: str, sentence_id: int | None, seconds: float,
                 source: str) -> dict:
        ok, errors, info = validate_amr(amr)
        return {
            "sentence_id": sentence_id,
            "base_amr": amr,
            "status": "ok" if ok else "invalid",
            "errors": errors,
            "n_nodes": info.get("n_nodes"),
            "n_edges": info.get("n_edges"),
            "parser": "spring",
            "model": MODEL_NAME,
            "seconds": round(seconds, 3),
            "source": source,
        }

"""Call the free OpenCode model (Tidepool Alpha) from Python.

The model is not reachable over plain HTTP: the OpenAI-compatible endpoints
answer 429 ``FreeUsageLimitError`` unless the request comes from OpenCode
itself. So this module shells out to ``opencode run --format json`` with the
checked-in project agent ``amr-annotator`` and reads the machine-readable event
stream:

  {"type":"text","part":{"text":"..."}}          assistant text
  {"type":"step_finish","part":{"tokens":{...}}} usage / cost

Only the JSON answer is returned to the caller; the token counts are kept for
the run report.
"""

from __future__ import annotations

import json
import os
import subprocess
import time

MODEL = os.environ.get("TIDEPOOL_MODEL", "opencode/tidepool-alpha-free#low")
AGENT = os.environ.get("TIDEPOOL_AGENT", "amr-annotator")
ROOT = os.path.dirname(os.path.abspath(__file__))


class TidepoolError(RuntimeError):
    pass


def _events(raw: str) -> tuple[str, dict]:
    """Assistant text plus the last step's usage from the json event stream."""
    texts: list[str] = []
    usage: dict = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = ev.get("type")
        part = ev.get("part") or {}
        if kind == "text":
            texts.append(part.get("text") or "")
        elif kind == "step_finish":
            usage = part.get("tokens") or usage
            usage["cost"] = part.get("cost", usage.get("cost"))
    return "\n".join(t for t in texts if t), usage


def extract_json(text: str) -> dict:
    """Parse the model answer; tolerate fences and leading prose."""
    clean = text.strip()
    if clean.startswith("```"):
        clean = "\n".join(clean.splitlines()[1:])
        if clean.rstrip().endswith("```"):
            clean = clean.rstrip()[:-3]
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    start = clean.find("{")
    end = clean.rfind("}")
    if start == -1 or end <= start:
        raise TidepoolError(f"no JSON object in answer: {text[:300]!r}")
    return json.loads(clean[start:end + 1])


def call(prompt: str, model: str | None = None, timeout: int = 1800,
         retries: int = 2) -> tuple[str, dict]:
    """Run one prompt through the annotator agent; return (text, usage)."""
    cmd = ["opencode", "run", "--format", "json", "--agent", AGENT,
           "--model", model or MODEL, "--archived", prompt]
    env = dict(os.environ)
    env.setdefault("PATH", os.path.expanduser("~/.local/bin") + ":" + env.get("PATH", ""))
    last = None
    for attempt in range(1, retries + 1):
        t0 = time.time()
        try:
            res = subprocess.run(cmd, cwd=ROOT, env=env, timeout=timeout,
                                 capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            last = TidepoolError(f"timeout after {timeout}s")
        else:
            text, usage = _events(res.stdout)
            if res.returncode == 0 and text.strip():
                usage["seconds"] = round(time.time() - t0, 1)
                return text, usage
            last = TidepoolError(
                f"exit {res.returncode}: {(res.stderr or res.stdout)[-400:]}")
        if attempt < retries:
            time.sleep(5 * attempt)
    raise last or TidepoolError("opencode run failed")


if __name__ == "__main__":
    import sys
    out, meta = call(" ".join(sys.argv[1:]) or "Reply with exactly PONG")
    print(json.dumps(meta))
    print(out[:1000])

"""Thin OpenAI-compatible chat client with a tool-call loop.

No SDK, no framework: plain ``requests``. The loop is bounded by
``max_tool_rounds`` and returns every tool exchange so it can be recorded in
the event log. A ``MockClient`` replays scripted results for tests.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import requests

from tools import SCHEMAS


def _safe_call(tool_runner: Callable[[str, dict], Any], name: str,
               args: dict) -> Any:
    try:
        return tool_runner(name, args)
    except Exception as exc:  # one bad call must not kill the turn
        return {"error": f"{type(exc).__name__}: {exc}"}


def run_tool_calls(tool_runner: Callable[[str, dict], Any],
                   calls: list[tuple[str, dict]],
                   max_workers: int = 8) -> list[Any]:
    """Run the tool calls of ONE assistant turn concurrently.

    Models emit several calls per turn when the work is independent; the most
    common case here is four ``spring_parse`` calls for four sentences of one
    paragraph, which is why a paragraph arrives unsplit. Results are returned
    in call order so the trace and the tool messages stay aligned.
    """
    if len(calls) <= 1:
        return [_safe_call(tool_runner, name, args) for name, args in calls]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(calls))) as pool:
        futures = [pool.submit(_safe_call, tool_runner, name, args)
                   for name, args in calls]
        return [f.result() for f in futures]


@dataclass
class LLMResult:
    content: str | None
    messages: list[dict]
    tool_trace: list[dict] = field(default_factory=list)
    rounds: int = 0
    usage: dict[str, Any] = field(default_factory=dict)


class Client(Protocol):
    def run(self, system: str | None, user: str | None,
            tool_runner: Callable[[str, dict], Any],
            messages: list[dict] | None = None) -> LLMResult: ...


class OpenAIClient:
    def __init__(self, cfg: dict) -> None:
        llm = cfg["llm"]
        self.url = (llm.get("api_url") or "").rstrip("/")
        self.key = llm.get("api_key") or ""
        self.model = cfg.get("model") or ""
        self.timeout = int(llm.get("timeout") or 180)
        self.retries = int(llm.get("retries") or 3)
        self.temperature = float(llm.get("temperature") or 0.0)
        self.max_tokens = int(llm.get("max_tokens") or 6000)
        self.max_rounds = int(cfg["agent"].get("max_tool_rounds") or 6)

    def _endpoint(self) -> str:
        return self.url if self.url.endswith("/chat/completions") \
            else self.url + "/chat/completions"

    def _post(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        last = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = requests.post(
                    self._endpoint(), json=payload, headers=headers,
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    return resp.json()
                last = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            except Exception as exc:  # network / timeout
                last = exc
            if attempt < self.retries:
                time.sleep(min(2 ** attempt, 15))
        raise RuntimeError(f"LLM request failed: {last}")

    def run(self, system: str | None, user: str | None,
            tool_runner: Callable[[str, dict], Any],
            messages: list[dict] | None = None) -> LLMResult:
        if messages is not None:
            messages = list(messages)
            if user:
                messages.append({"role": "user", "content": user})
        else:
            messages = [
                {"role": "system", "content": system or ""},
                {"role": "user", "content": user or ""},
            ]
        trace: list[dict] = []
        usage: dict[str, Any] = {}
        for round_no in range(1, self.max_rounds + 2):
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "tools": SCHEMAS,
                "tool_choice": "auto",
            }
            data = self._post(payload)
            usage = data.get("usage") or usage
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            tool_calls = message.get("tool_calls") or []
            messages.append(message)
            if not tool_calls:
                return LLMResult(
                    content=message.get("content"), messages=messages,
                    tool_trace=trace, rounds=round_no, usage=usage,
                )
            parsed = []
            for call in tool_calls:
                fn = call.get("function") or {}
                name = fn.get("name") or ""
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": fn.get("arguments")}
                parsed.append((call, name, args))
            results = run_tool_calls(
                tool_runner, [(name, args) for _, name, args in parsed])
            for (call, name, args), result in zip(parsed, results):
                trace.append({"round": round_no, "tool": name, "args": args,
                              "result": result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "name": name,
                    "content": json.dumps(result, ensure_ascii=False),
                })
            if round_no >= self.max_rounds:
                # Force a final answer without tools.
                payload = {
                    "model": self.model, "messages": messages,
                    "temperature": self.temperature, "max_tokens": self.max_tokens,
                }
                data = self._post(payload)
                choice = (data.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                messages.append(message)
                return LLMResult(
                    content=message.get("content"), messages=messages,
                    tool_trace=trace, rounds=round_no + 1,
                    usage=data.get("usage") or usage,
                )
        return LLMResult(content=None, messages=messages, tool_trace=trace,
                         rounds=self.max_rounds, usage=usage)


class MockClient:
    """Replays a scripted sequence of (tool_calls, final_content) steps.

    ``script`` is a list of steps. A step is either a dict
    ``{"tool_calls":[{"name":..,"args":..}, ...]}`` or
    ``{"content": <json-string-or-dict>}``. Tool results are computed by the
    real tool runner so the event log reflects real tool behaviour.
    """

    def __init__(self, script: list[dict] | None = None) -> None:
        self.script = list(script or [])
        self._i = 0

    def run(self, system: str | None, user: str | None,
            tool_runner: Callable[[str, dict], Any],
            messages: list[dict] | None = None) -> LLMResult:
        if messages is None:
            messages = [{"role": "system", "content": system or ""},
                        {"role": "user", "content": user or ""}]
        else:
            messages = list(messages)
            if user:
                messages.append({"role": "user", "content": user})
        trace: list[dict] = []
        rounds = 0
        while self._i < len(self.script):
            step = self.script[self._i]
            self._i += 1
            rounds += 1
            if "tool_calls" in step:
                calls = [(tc["name"], tc.get("args") or {})
                         for tc in step["tool_calls"]]
                for (name, args), result in zip(
                        calls, run_tool_calls(tool_runner, calls)):
                    trace.append({"round": rounds, "tool": name,
                                  "args": args, "result": result})
                    messages.append({"role": "tool", "name": name,
                                     "content": json.dumps(result)})
                continue
            content = step.get("content")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            messages.append({"role": "assistant", "content": content})
            return LLMResult(content=content, messages=messages,
                             tool_trace=trace, rounds=rounds)
        # No more script: return empty results so the agent records a failure
        # instead of crashing the whole run.
        return LLMResult(content=json.dumps({"results": []}),
                         messages=messages, tool_trace=trace, rounds=rounds)

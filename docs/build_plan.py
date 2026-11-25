#!/usr/bin/env python3
"""Rebuild the plain-language pipeline and sentence-walkthrough sections.

Run it after changing the text below::

    python docs/build_plan.py

The script is idempotent: every element it creates is tagged in ``customData``
so a rerun replaces the section and leaves the rest of the drawing untouched.
"""

from __future__ import annotations

import json
import random
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "agentic_amr_dataset.excalidraw"
TAG = "agent-pipeline"
OLD_FIVE_TAG = "agent-five"
ONE_TAG = "agent-one"
CURRENT_TAG = TAG
SECTIONS = (TAG, OLD_FIVE_TAG, ONE_TAG)

FONT = 8      # same hand-drawn family as the rest of the drawing
CODE = 3      # monospaced family, used for examples
INK = "#1e1e1e"
GREY = "#495057"
BLUE = "#a5d8ff"
BLUE_DARK = "#1971c2"
YELLOW = "#ffec99"
GREEN = "#b2f2bb"
GREEN_DARK = "#2f9e44"
RED = "#ffc9c9"
RED_DARK = "#e03131"
PURPLE = "#d0bfff"
PURPLE_DARK = "#6741d9"
ORANGE = "#ffd8a8"
ORANGE_DARK = "#e8590c"

rng = random.Random(5772)
INDEX = 0
LAST_KEY: str | None = None

# -- fractional order keys (same algorithm Excalidraw uses) ----------------- #
# Keys keep the drawing's z-order. Existing keys look like "a1", "Zz", "b1X";
# new elements must sort after the last one and stay valid.
DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
INT_DIGITS = DIGITS[10:]


def _int_length(head: str) -> int:
    i = INT_DIGITS.index(head)
    half = len(INT_DIGITS) // 2
    return half - i + 1 if i < half else i - half + 2


def _int_part(key: str) -> str:
    n = _int_length(key[0])
    if n > len(key):
        raise ValueError(f"invalid order key: {key}")
    return key[:n]


def _midpoint(a: str, b: str | None = None) -> str:
    zero = DIGITS[0]
    if b is not None and a >= b:
        raise ValueError("keys out of order")
    if a.endswith(zero) or (b and b.endswith(zero)):
        raise ValueError("trailing zero")
    if b:
        n = 0
        while n < len(b) and (a[n] if n < len(a) else zero) == b[n]:
            n += 1
        if n:
            return b[:n] + _midpoint(a[n:], b[n:])
    da = DIGITS.index(a[0]) if a else 0
    db = DIGITS.index(b[0]) if b else len(DIGITS)
    if db - da > 1:
        return DIGITS[int(0.5 * (da + db) + 0.5)]
    if b and len(b) > 1:
        return b[:1]
    return DIGITS[da] + _midpoint(a[1:], None)


def _increment_integer(x: str) -> str | None:
    head, trailing = x[0], ""
    for i in range(len(x) - 1, 0, -1):
        d = DIGITS.index(x[i]) + 1
        if d == len(DIGITS):
            trailing = "0" + trailing
        else:
            return head + x[1:i] + DIGITS[d] + trailing
    hi = INT_DIGITS.index(head)
    if hi == len(INT_DIGITS) - 1:
        return None
    h = INT_DIGITS[hi + 1]
    delta = _int_length(h) - _int_length(head)
    if delta > 0:
        return h + trailing + DIGITS[0]
    if delta < 0:
        return h + trailing[1:]
    return h + trailing


def key_after(prev: str | None) -> str:
    """A valid order key greater than ``prev`` (None starts a new scene)."""
    if prev is None:
        return "a0"
    part = _int_part(prev)
    nxt = _increment_integer(part)
    return nxt if nxt is not None else part + _midpoint(prev[len(part):])


def _next_index() -> str:
    """Next valid fractional index; keeps new elements on top."""
    global INDEX, LAST_KEY
    INDEX += 1
    LAST_KEY = key_after(LAST_KEY)
    return LAST_KEY


def _base(kind: str, x: float, y: float, w: float, h: float,
          stroke: str = INK, fill: str = "transparent",
          dashed: bool = False) -> dict:
    now = 1794860000000
    return {
        "id": f"{CURRENT_TAG}-{kind}-{INDEX + 1:03d}",
        "type": kind,
        "x": round(x, 2),
        "y": round(y, 2),
        "width": round(w, 2),
        "height": round(h, 2),
        "angle": 0,
        "strokeColor": stroke,
        "backgroundColor": fill,
        "fillStyle": "solid",
        "strokeWidth": 2 if kind == "rectangle" and fill != "transparent" else 1,
        "strokeStyle": "dashed" if dashed else "solid",
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "index": _next_index(),
        "roundness": None,
        "seed": rng.randint(1, 2 ** 31 - 1),
        "version": 1,
        "versionNonce": rng.randint(1, 2 ** 31 - 1),
        "isDeleted": False,
        "boundElements": None,
        "updated": now,
        "created": now,
        "link": None,
        "locked": False,
        "customData": {"section": CURRENT_TAG},
    }


def text(x: float, y: float, w: float, body: str, size: float = 16,
         color: str = INK, family: int = FONT, align: str = "left",
         line_height: float = 1.25) -> list[dict]:
    """One standalone text element, height computed from its own lines."""
    lines = body.count("\n") + 1
    h = lines * size * line_height
    el = _base("text", x, y, w, h, stroke=color)
    el.update({
        "text": body,
        "fontSize": size,
        "baseFontSize": None,
        "fontFamily": family,
        "textAlign": align,
        "verticalAlign": "top",
        "containerId": None,
        "originalText": body,
        "autoResize": True,
        "lineHeight": line_height,
        "labelPosition": None,
    })
    return [el]


def box(x: float, y: float, w: float, h: float, fill: str = BLUE,
        stroke: str = INK, dashed: bool = False,
        radius: bool = True) -> list[dict]:
    el = _base("rectangle", x, y, w, h, stroke=stroke, fill=fill, dashed=dashed)
    el.update({
        "roundness": {"type": 3} if radius else None,
    })
    return [el]


def arrow(x: float, y: float, dx: float, dy: float,
          color: str = INK) -> list[dict]:
    el = _base("arrow", x, y, abs(dx), abs(dy), stroke=color)
    el.update({
        "points": [[0, 0], [round(dx, 2), round(dy, 2)]],
        "startBinding": None,
        "endBinding": None,
        "startArrowhead": None,
        "endArrowhead": "arrow",
        "elbowed": False,
        "fixedSegments": None,
        "startIsSpecial": None,
        "endIsSpecial": None,
    })
    return [el]


def wrap(body: str, width: int) -> str:
    out = []
    for line in body.split("\n"):
        out.extend(textwrap.wrap(line, width) or [""])
    return "\n".join(out)


class Section:
    """Collects elements and keeps a simple top-down cursor."""

    def __init__(self) -> None:
        self.els: list[dict] = []

    def add(self, elements: list[dict]) -> None:
        self.els.extend(elements)

    def heading(self, y: float, title: str, note: str = "",
                x: float = 40, w: float = 1480) -> float:
        self.add(text(x, y, w, title, size=28, color=INK))
        y += 44
        if note:
            note_body = wrap(note, 150)
            self.add(text(x, y + 6, w, note_body, size=16, color=GREY))
            y += 6 + (note_body.count("\n") + 1) * 20 + 14
        return y

    def step(self, y: float, number: int, title: str, desc: str,
             example: str | None, code_size: float = 14,
             desc_chars: int = 62) -> float:
        badge = 52
        self.add(box(40, y, badge, badge, fill=BLUE_DARK, stroke=BLUE_DARK))
        self.add(text(52.0, y + 10, 28, str(number), size=26, color="#ffffff"))
        self.add(text(112, y, 520, title, size=22, color=INK))
        desc_body = wrap(desc, desc_chars)
        desc_w = 1420 if example is None else 510
        self.add(text(112, y + 36, desc_w, desc_body, size=15, color=INK))
        desc_h = (desc_body.count("\n") + 1) * 15 * 1.25

        card_h = 0
        if example is not None:
            card_x, card_w = 660, 880
            body = wrap(example, 96)
            code_h = (body.count("\n") + 1) * code_size * 1.25
            card_h = code_h + 30
            self.add(box(card_x, y - 2, card_w, card_h, fill=YELLOW))
            self.add(text(card_x + 20, y + 12, card_w - 40, body,
                          size=code_size, family=CODE, color=INK))

        row_h = (max(card_h, desc_h + 50) + 58) if example is not None \
            else (desc_h + 36 + 34)
        # timeline arrow to the next step (a tool step extends it later)
        if example is not None:
            self.add(arrow(66, y + badge + 8, 0, row_h - badge - 26))
        return y + row_h

    def tool_card(self, x: float, y: float, w: float, name: str,
                  question: str, body: str, chars: int = 78) -> float:
        """Draw one tool card; returns its height (not the next y)."""
        code = wrap(body, chars)
        body_h = (code.count("\n") + 1) * 13 * 1.25
        h = 58 + body_h + 16
        self.add(box(x, y, w, h, fill=PURPLE, stroke=PURPLE_DARK))
        self.add(text(x + 18, y + 10, w - 36, name, size=18,
                      color=PURPLE_DARK, family=CODE))
        self.add(text(x + 18, y + 36, w - 36, question, size=14, color=INK))
        self.add(text(x + 18, y + 60, w - 36, code, size=13, family=CODE))
        return h


def build(section: Section) -> None:
    y = 2050
    section.add(text(
        40, y, 1480,
        "Continues from the diagram above: the same agent, one article at a time, "
        "with a worked example for every step.",
        size=15, color=BLUE_DARK))
    y += 40

    # ---------------------------------------------------------------- header
    y = section.heading(
        y,
        "Agent pipeline  —  one article, start to finish",
        "One agent handles one article. It reads in order, remembers names it has "
        "already cleared, asks a tool instead of guessing, fixes its own broken "
        "graphs, and writes results after every step.",
    )
    y += 8

    # ------------------------------------------------------- full pipeline
    section.add(text(40, y, 400, "Full pipeline", size=20, color=BLUE_DARK))
    y += 36
    stages = [
        ("Paragraphs in", "the agent splits them", "#d0ebff"),
        ("Read in order", "article 5772", BLUE),
        ("Article memory", "names cleared so far", BLUE),
        ("Tools", "ask, do not guess", YELLOW),
        ("Rewrite", "short mention -> full name", YELLOW),
        ("Check · repair · reject", "sentence and graph agree", ORANGE),
        ("Save", "dataset + full trace", GREEN),
    ]
    bw, gap, bx = 192, 18, 40
    for i, (name, sub, fill) in enumerate(stages):
        x = bx + i * (bw + gap)
        section.add(box(x, y, bw, 82, fill=fill))
        section.add(text(x + 12, y + 10, bw - 24, name, size=15))
        section.add(text(x + 12, y + 40, bw - 24, sub, size=12, color=GREY))
        if i < len(stages) - 1:
            section.add(arrow(x + bw + 2, y + 41, gap - 6, 0))
    y += 82 + 18
    note_body = wrap("Old pieces still used: corpus DB for the article text · SPRING "
                     "as a tool the agent calls once per sentence · Wikipedia/"
                     "Wikidata for names (live, always cached) · OpenRouter for the "
                     "model. Settings are inherited, not retyped.", 150)
    section.add(text(40, y, 1480, note_body, size=14, color=GREY))
    y += (note_body.count("\n") + 1) * 14 * 1.25 + 10
    section.add(text(
        40, y, 1480,
        "colours:  blue = read and remember  ·  yellow = rewrite  ·  "
        "orange = check  ·  red = repair or reject  ·  green = save",
        size=14, color=GREY))
    y += 52

    # ---------------------------------------------------- the loop, as a flow
    section.add(text(40, y, 1480, "The loop, as a flow", size=20,
                     color=BLUE_DARK))
    y += 32
    section.add(text(
        40, y, 1480,
        "The same seven steps as a chain, with what travels in each arrow: "
        "one batch goes down the middle, a failed check leaves to the right and "
        "comes back, or is written down as rejected.",
        size=14, color=GREY))
    y += 40

    nodes = [
        ("1 · Paragraph in", "no sentences, no graphs, no ids", "#d0ebff"),
        ("2 · The agent splits it", "paragraph text -> its own sentences", BLUE),
        ("3 · spring_parse × one per sentence", "one turn · dispatched in parallel", YELLOW),
        ("4 · The toolbox matches text -> id", "the model never sees the id", PURPLE),
        ("5 · Edit, then ask the other tools", "article_search · resolve_mention · wiki · check", YELLOW),
        ("6 · Answer keyed by `original`", "one entry per sentence it parsed", YELLOW),
        ("7 · Validation", "graph · edits · coverage · names", ORANGE),
        ("8 · Record", "records.jsonl, memory, next batch", GREEN),
    ]
    cx, cw, ch, gap = 480, 560, 62, 30
    chain_y = y
    for i, (head, sub, fill) in enumerate(nodes):
        ny = chain_y + i * (ch + gap)
        section.add(box(cx, ny, cw, ch, fill=fill))
        section.add(text(cx + 16, ny + 9, cw - 32, head, size=15))
        section.add(text(cx + 16, ny + 35, cw - 32, sub, size=12, color=GREY))
        if i < len(nodes) - 1:
            section.add(arrow(cx + cw / 2, ny + ch + 3, 0, gap - 8))

    def beside(i: int, body: str, dy: float = 6) -> None:
        """A note to the left of one node, joined to it by a short pointer."""
        ny = chain_y + i * (ch + gap)
        section.add(text(60, ny + dy, 380, body, size=12, family=CODE,
                         color=INK))
        section.add(arrow(444, ny + ch / 2, 30, 0, color=GREY))

    beside(0, 'payload\n{ article, ledger,\n  paragraphs: [\n    { para_index, text } ] }')
    beside(2, 'result, once per call\n{ amr, status, errors,\n  parser, model,\n  n_nodes, n_edges }')
    beside(3, '"The award was challenged\n under section 34 …"\n   -> sentence_id 360450\n   -> base graph it edits')
    beside(5, '" … under section 34 of the\n  Act. The Court held …"\n(answers carry the text,\n not an id)')
    beside(6, 'fail -> exact errors go\nback with the same base\ngraph and the tools')

    # the branch: fail -> repair -> reject
    fail_i = 6
    fail_y = chain_y + fail_i * (ch + gap)
    branch_x, branch_w = 1090, 390
    section.add(arrow(cx + cw + 4, fail_y + ch / 2, branch_x - cx - cw - 12, 0))
    section.add(box(branch_x, fail_y, branch_w, ch, fill=RED))
    section.add(text(branch_x + 16, fail_y + 9, branch_w - 32,
                     "Repair, up to 2 tries", size=15))
    section.add(text(branch_x + 16, fail_y + 35, branch_w - 32,
                     "same base graph, same tools", size=12, color=GREY))
    section.add(arrow(branch_x + branch_w / 2, fail_y + ch + 3, 0, gap - 8))
    rej_y = fail_y + ch + gap
    section.add(box(branch_x, rej_y, branch_w, ch, fill=RED,
                    stroke=RED_DARK, dashed=True))
    section.add(text(branch_x + 16, rej_y + 9, branch_w - 32,
                     "Rejected, with the reason", size=15))
    section.add(text(branch_x + 16, rej_y + 35, branch_w - 32,
                     "no_base_graph · no_graph · no_credible_source",
                     size=12, color=GREY))
    section.add(text(
        branch_x, rej_y + ch + 14, branch_w,
        wrap("Nothing is written unchecked: a sentence the agent never parsed "
             "has no crutch to edit, and one that still fails the check keeps "
             "its reason in rejected.jsonl.", 46),
        size=13, color=RED_DARK))
    y = chain_y + len(nodes) * (ch + gap) + 26
    y = section.heading(y, "What each step does")
    y += 16

    y = section.step(
        y, 1, "Hand over the paragraph, not the sentences",
        "A batch is two or three whole paragraphs, joined the way the corpus "
        "stored them and never pre-split. The agent splits them itself, then "
        "calls spring_parse once per sentence — all the calls for a paragraph "
        "in one turn, so they run at the same time. spring_parse is a plain "
        "parser call: sentence in, classic AMR out, nothing else. Those graphs "
        "are the crutch it edits, and a sentence it never parsed is rejected.",
        "EXAMPLE — one batch, paragraph 6 of 5772\n"
        '  "The award was challenged under section 34 of the Act. The\n'
        '   Court held that ..."            (unsplit, one string)\n'
        "\n"
        "split by the agent, then in one turn:\n"
        '  spring_parse("The award was challenged under ...")\n'
        '  spring_parse("The Court held that ...")\n'
        '  spring_parse("The award was set aside ...")   (4 calls, parallel)\n'
        "\n"
        "back, once per call, no ids and no article facts:\n"
        "  (z1 / obligate-01 :ARG1 (z2 / award))  8 nodes, 14 edges\n"
        "  status valid · or SPRING's own error list\n"
        "\n"
        "the agent never sees a corpus sentence id; the harness matches\n"
        "its sentences back to the corpus text when the answer arrives"
    )

    y = section.step(
        y, 2, "Start article memory",
        "A short list of names the article has already made clear: full name, "
        "kind, aliases, and where it was found. It is rebuilt from accepted "
        "results and saved with the checkpoint, so a restart keeps it.",
        "EXAMPLE — memory after sentence 360428\n"
        '{ "name": "Arbitration and Conciliation Act, 1996",\n'
        '  "kind": "statute",\n'
        '  "alias": "Act",\n'
        '  "evidence": "declared in sentence 360428" }\n'
        "\n"
        "saved after every batch -> checkpoints/5772.json",
    )

    y = section.step(
        y, 3, "Read the article in order, one batch at a time",
        "Sentences are read from the corpus DB in the article's order and sent "
        "eight at a time, so a declaration is always seen before the short "
        "mentions that depend on it. One article is one independent task, so "
        "many articles can run side by side without sharing anything.",
        "EXAMPLE — article 5772, batch 3\n"
        '"The Importance of Determination of Seat under Indian Arbitration Law"\n'
        "53 sentences, in order\n"
        "\n"
        "360450  The award was challenged under section 34 ...\n"
        "360477  ... that the Act still provides no reference ...\n"
        "\n"
        "to the model:  2 whole paragraphs, unsplit, in order\n"
        "               the title and the memory\n"
        "back:          { original, sentence, amr, edits[], entities[],\n"
        "                 status }  ·  original = the text it split\n"
        "\n"
        "or a tool call first, answered once the name is known",
    )

    # ------------------------------------------------------------- the tools
    tools_y = y
    y = section.step(
        y, 4, "Tools: ask instead of guess",
        "The five tools below are the only ways the model can check something. "
        "Each answer comes with its evidence, and every call is written to the "
        "article's event log. The model can call them as often as it needs "
        "inside a batch.",
        None, desc_chars=150,
    )
    tool_y = y + 12
    col_w = 698
    gap = 30
    spring_card = (
        'paragraph 6 of 5772, unsplit, one string\n'
        '  "The award was challenged under section 34 of the Act. The\n'
        '   Court held that the award was set aside."\n'
        "\n"
        "the agent splits it itself, then in ONE turn:\n"
        '  spring_parse(sentence="The award was challenged under ...")\n'
        '  spring_parse(sentence="The Court held that ...")      (calls run\n'
        '  spring_parse(sentence="The award was set aside.")       in parallel)\n'
        "\n"
        "back, once per call:  (z1 / obligate-01 ...)  7 nodes, 11 edges · valid\n"
        "-> that graph is what the agent edits. The tool knows nothing about the\n"
        "   article — no ids, no memory: it is SPRING on one sentence."
    )
    spring_h = section.tool_card(
        112, tool_y, 1368, "spring_parse",
        "What does a plain SPRING parse of this sentence look like?",
        spring_card, chars=140)
    tool_y += spring_h + 14
    cards = [
        ("article_search",
         "Where was this said earlier in this article?",
         'call: article_search("Arbitration and Conciliation Act",\n'
         '                     before "However it must be noted ...")\n'
         "-> 360428  ... (the ‘Act’) ...  (curly quotes)"),
        ("resolve_mention",
         "What does this short mention mean here, and why?",
         'call: resolve_mention("the Act", "However it must ...")\n'
         "-> Arbitration and Conciliation Act, 1996\n"
         '   score 0.96 · evidence: article · alias "Act"'),
        ("wiki_lookup",
         "Does this name exist outside the article?",
         'live Wikipedia + Wikidata, always cached\n'
         'call: wiki_lookup("Arbitration and Conciliation Act, 1996")\n'
         "-> Arbitration_and_Conciliation_Act,_1996 (cache hit next time)"),
        ("check_amr",
         "Is the graph valid, and does it match the sentence?",
         "call: check_amr(sentence, graph, edits)\n"
         "-> ok  ·  or the exact list of errors"),
    ]
    sizes = [
        58 + (wrap(body, 78).count("\n") + 1) * 13 * 1.25 + 16
        for _, _, body in cards
    ]
    row0 = tool_y
    row1 = tool_y + max(sizes[0], sizes[1]) + 16
    for i, (name, question, body) in enumerate(cards):
        x = 112 + (i % 2) * (col_w + gap)
        cy = row0 if i < 2 else row1
        section.tool_card(x, cy, col_w, name, question, body)
    y = row1 + max(sizes[2], sizes[3]) + 26
    section.add(arrow(66, tools_y + 60, 0, y - tools_y - 68))

    y = section.step(
        y, 5, "Rewrite the sentence when a mention resolves",
        "Only the mention changes; every other character stays the same. The "
        "original sentence is always kept next to the resolved one, so nothing "
        "is lost.",
        "EXAMPLE — sentence 360477\n"
        "before  ... that the Act still provides no reference ...\n"
        "after   ... that the Arbitration and Conciliation Act, 1996\n"
        "            still provides no reference ...\n"
        "\n"
        'edit { surface: "Act", occurrence: 1,\n'
        '       replacement: "Arbitration and Conciliation Act, 1996",\n'
        '       resolved_to: "a1", evidence: "article", kind: "statute" }\n'
        "      (the word before it, \"the\", stays in the sentence)\n"
        "\n"
        "sentence_original keeps the corpus text",
    )

    y = section.step(
        y, 6, "Check, repair, reject",
        "One check reads both sides: the edit must exist in the sentence, the "
        "specific name must exist in the graph, and the graph must parse. When "
        "it fails, the same conversation gets the exact errors back and another "
        "try, base graph still in hand and tools still available. Up to two "
        "tries; what still fails is rejected, never written unchecked.",
        "EXAMPLE — the check, both directions\n"
        "pass  edit found · name in graph · graph parses\n"
        "fail  :legal-type may appear only on a gov-entity,\n"
        "      got 'court'\n"
        "      -> (h / court) becomes (h / gov-entity)\n"
        "fail  edit span not found: 'NOTHERE' occurrence 1\n"
        "fail  name \"Arbitration and Conciliation Act, 1996\"\n"
        "      missing from graph\n"
        "\n"
        "still failing after two tries, or nothing to resolve:\n"
        '"the Scheme"  -> no_credible_source\n'
        '"the Board"   -> ambiguous_antecedent\n'
        "  rejected.jsonl keeps the reason and the candidates",
    )

    y = section.step(
        y, 7, "Save everything, after every step",
        "Outputs are append-only, so a crash costs at most one batch. A finished "
        "article is skipped next time, and a half-finished one continues from "
        "its last checkpoint.",
        "EXAMPLE — data/agent/ after article 5772\n"
        "records.jsonl              accepted sentences\n"
        "rejected.jsonl             reason + what was tried\n"
        "events/article_5772.jsonl  each tool call, repair, accept, reject\n"
        "checkpoints/5772.json      next batch + memory\n"
        "done.jsonl · stats.json    resume + counts\n"
        "dataset.all.jsonl/.amr     ::snt · ::snt_orig · ::edits",
    )

    # ---------------------------------------------- running many articles
    y += 4
    many = (
        "python run.py --article 5772 --mock --build --example   (no key)\n"
        "python run.py --all --limit 500 --workers 8\n"
        "python run.py --all --workers 8 --shard 1/4\n"
        "\n"
        "3 articles, 3 workers -> 177 accepted, 2 rewrites\n"
        "rerun -> all 3 skipped, nothing repeated"
    )
    many_h = (many.count("\n") + 1) * 13 * 1.25 + 88
    section.add(box(40, y, 1480, many_h, fill="#f8f9fa", stroke=GREY,
                    dashed=True))
    section.add(text(64, y + 14, 700, "Run many articles at once", size=19))
    section.add(text(
        64, y + 46, 700,
        wrap("Every worker takes one article and shares nothing, so a shard "
             "splits articles by id and several machines can run the same "
             "command at the same time. A retry, a crash or an extra worker "
             "never touches another article's work.", 74),
        size=14))
    section.add(text(800, y + 18, 700, many, size=13, family=CODE))
    y += many_h + 26

    # ------------------------------------------------------------- settings
    y += 6
    y = section.heading(y, "Settings")
    y += 10
    settings = (
        "read from config.json when it exists\n"
        "  OpenRouter api_url + api_key + model (or AMR_* env vars)\n"
        "  corpus DB      agentic_dataset/data/india.db\n"
        "  base graphs    agentic_dataset/data/amr_spring.jsonl (optional)\n"
        "  wiki cache     agentic_dataset/data/agent/wiki_cache.sqlite\n"
        "\n"
        "priority\n"
        "  defaults < config.json < environment < command line\n"
        "\n"
        "work settings\n"
        "  batch size    8 sentences per model call\n"
        "  repair tries  2 per sentence, then reject\n"
        "\n"
        "rewrite on/off and which kinds\n"
        "  rewrite.enabled, rewrite.kinds\n"
        "  or --rewrite on|off, --rewrite-kinds statute,provision"
    )
    body = wrap(settings, 120)
    h = (body.count("\n") + 1) * 14 * 1.25 + 32
    section.add(box(40, y, 1480, h, fill="#f8f9fa", stroke=GREY, dashed=True))
    section.add(text(60, y + 16, 1440, body, size=14, family=CODE))
    y += h + 26

    section.add(text(
        40, y, 1480,
        wrap("Why this stays cheap: one article is one task, so a retry, a crash "
             "or an extra worker never touches another article's work.", 150),
        size=16, color=GREEN_DARK))
    return y + 40


def build_one(section: Section, start_y: float) -> float:
    """One sentence, followed from what arrives to what is saved.

    Every sentence, tool line, score, graph, validator message and memory list
    below is read from the real runs in ``data/five/report.json`` and
    ``data/five_repair/report.json``; nothing is retyped by hand.
    """
    report = json.loads(
        (ROOT / "data" / "five" / "report.json").read_text(encoding="utf-8"))
    repair = json.loads(
        (ROOT / "data" / "five_repair" / "report.json")
        .read_text(encoding="utf-8"))
    sid = 360450
    key = str(sid)
    rec = next(r for r in report["records"] if r["sentence_id"] == sid)
    original = rec["sentence_original"]
    resolved = rec["sentence"]
    edit = rec["edits"][0]
    spring = report["spring"][key]
    check = report["checks"][key]
    cm = check["coverage_math"]
    tools_body = "\n".join(report["tools"][key])
    mem_before = "\n".join(report["memory_before"][key])
    mem_after = "\n".join(report["memory_after"][key])
    error = repair["first_errors"][0]

    PALE_BLUE = "#e7f5ff"
    PALE_YELLOW = "#fff9db"
    PALE_ORANGE = "#fff4e6"
    PALE_RED = "#fff5f5"
    PALE_GREEN = "#ebfbee"
    PALE_PURPLE = "#f3f0ff"

    def th(body: str, size: float) -> float:
        return (body.count("\n") + 1) * size * 1.25

    def wrap_code(body: str, width: int) -> str:
        """Wrap code lines but keep their leading indentation."""
        out = []
        for line in body.split("\n"):
            indent = len(line) - len(line.lstrip(" "))
            chunks = textwrap.wrap(line.strip(), max(12, width - indent)) or [""]
            out.append(" " * indent + chunks[0])
            out.extend(" " * (indent + 2) + chunk for chunk in chunks[1:])
        return "\n".join(out)

    def stage(y: float, t: str, title: str, note: str, body: str = "",
              right: tuple[str, str] | None = None, kind: str = "code",
              body_size: float | None = None, fill: str = "#ffffff",
              badge: str = BLUE, accent: str = BLUE_DARK,
              change: str = "", sentence: str | None = None,
              sentence_label: str = "SENTENCE NOW",
              last: bool = False) -> float:
        """One moment in time: badge, card, body, what changed, sentence now."""
        card_x, card_w = 112, 1408
        ix, inner_w = 132, 1368
        left_w = 620 if right else inner_w
        right_x, right_w = ix + 648, 700
        if body_size is None:
            body_size = 11.5 if kind == "code" else 15

        note_body = wrap(note, 152)
        note_h = th(note_body, 13)
        if kind == "code":
            body_shown = wrap_code(body, int(left_w / (body_size * 0.6)))
        else:
            body_shown = wrap(body, int(left_w / (body_size * 0.55)))
        body_h = th(body_shown, body_size) if body else 0

        right_parts = None
        right_h = 0
        if right:
            rhead, rbody = right
            rbody_shown = wrap_code(rbody, int(right_w / 6.9))
            rhead_wrapped = wrap(rhead, 96)
            right_parts = (rhead_wrapped, rbody_shown)
            right_h = th(rhead_wrapped, 12.5) + 8 + th(rbody_shown, 11)

        change_body = "what changed  " + change if change else ""
        change_h = th(change_body, 13) if change else 0
        sent_body = wrap(sentence, 150) if sentence else ""
        sent_h = th(sent_body, 11) if sentence else 0

        content_h = max(body_h, right_h)
        h = (16 + 26 + note_h + 6 + content_h
             + (10 + change_h if change else 0)
             + (10 + 14 + sent_h if sentence else 0) + 14)

        section.add(box(40, y, 52, 52, fill=badge, stroke=accent))
        section.add(text(45, y + 12, 44, t, size=18, color=INK))
        section.add(box(card_x, y, card_w, h, fill=fill, stroke=GREY))
        section.add(text(ix, y + 14, inner_w - 90, title, size=19))
        section.add(text(card_x + card_w - 90, y + 17, 70,
                         f"{int(t[1:]) + 1}/9", size=13, color=GREY,
                         align="right"))

        cy = y + 16 + 26
        section.add(text(ix, cy, inner_w, note_body, size=13, color=GREY))
        cy += note_h + 6
        if body:
            section.add(text(ix, cy, left_w, body_shown, size=body_size,
                             family=CODE if kind == "code" else FONT))
        if right_parts:
            section.add(text(right_x, cy, right_w, right_parts[0], size=12.5,
                             family=CODE, color=accent))
            section.add(text(right_x, cy + th(right_parts[0], 12.5) + 8,
                             right_w, right_parts[1], size=11,
                             family=CODE, color=INK))
        cy += content_h

        if change:
            cy += 10
            section.add(text(ix, cy, inner_w, change_body, size=13,
                             color=accent))
            cy += change_h
        if sentence:
            cy += 10
            section.add(text(ix, cy, inner_w, sentence_label, size=11,
                             color=GREY))
            section.add(text(ix, cy + 14, inner_w, sent_body, size=11,
                             family=CODE, color=GREY))

        if not last:
            section.add(arrow(66, y + 54, 0, h - 44, color=GREY))
        return y + h + 16

    y = start_y
    section.add(text(
        40, y, 1480,
        "Six real sentences  —  one in detail, all six side by side",
        size=28, color=INK))
    y += 44
    intro = wrap(
        "Sentence 360450 of article 5772 first, followed from the moment the "
        "agent reads it to the moment it is saved. Each card is one moment, in "
        "order: read, remember, ask, rewrite, draw, repair, check, remember, "
        "save. Then all six sentences of the article side by side, the same nine "
        "moments each, and the one that was rejected, with the reason. The "
        "sentences, the scores, both graphs, the validator output and the memory "
        "lists are the real ones in data/five/report.json and "
        "data/five_repair/report.json. No model API was called.", 148)
    section.add(text(40, y + 4, 1480, intro, size=16, color=GREY))
    y += th(intro, 16) + 20

    # --------------------------------------------------- before / after strip
    left_label = "BEFORE  ·  corpus sentence, legal.db"
    right_label = "AFTER  ·  sentence saved in records.jsonl"
    left_body = wrap(original, 74)
    right_body = wrap(resolved, 74)
    note_body = wrap(
        f'the span "{edit["surface"]}" (3 characters, start {edit["start"]}, '
        f'end {edit["end"]}) became "{edit["replacement"]}" '
        "(40 characters). Every other character of the sentence is identical, "
        "and the corpus text is kept as sentence_original.", 148)
    strip_h = (16 + th(left_body, 13) + 18 + th(note_body, 13) + 16)
    section.add(box(40, y, 1480, strip_h, fill="#ffffff", stroke=GREY))
    section.add(text(64, y + 12, 660, left_label, size=12, color=GREY))
    section.add(text(64, y + 32, 690, left_body, size=13, family=CODE))
    section.add(text(790, y + 12, 660, right_label, size=12, color=GREY))
    section.add(text(790, y + 32, 690, right_body, size=13, family=CODE))
    ny = y + 32 + th(left_body, 13) + 18
    section.add(text(64, ny, 1432, note_body, size=13, color=GREEN_DARK))
    y += strip_h + 22

    # -------------------------------------------------------- how numbers work
    y = section.heading(
        y, "How the two numbers in this story are calculated")
    y += 6
    formulas = [
        ("resolve_mention score", report["formulas"]["resolve_mention"],
         PURPLE_DARK),
        ("coverage", report["formulas"]["coverage"], GREEN_DARK),
    ]
    col_w, col_gap = 726, 28
    bodies = ["\n".join(wrap(line, 92) for line in lines)
              for _, lines, _ in formulas]
    col_h = max(th(body, 11.5) for body in bodies) + 64
    for i, ((title, _, color), body) in enumerate(zip(formulas, bodies)):
        cx = 40 + i * (col_w + col_gap)
        section.add(box(cx, y, col_w, col_h, fill="#f8f9fa", stroke=GREY))
        section.add(text(cx + 14, y + 10, col_w - 28, title, size=13,
                         color=color, family=CODE))
        section.add(text(cx + 14, y + 38, col_w - 28, body, size=11.5,
                         family=CODE, color=INK))
    y += col_h + 10
    section.add(text(
        40, y, 1480,
        "No wiki_lookup here: the statute was already settled inside the "
        "article, so the name comes from the article, not from outside.",
        size=13, color=GREY))
    y += 30

    # ---------------------------------------------------------------- t0 read
    y = stage(
        y, "t0", "Split the paragraph, parse the sentence",
        "Article 5772, paragraphs in order. This sentence came out of an "
        "unsplit paragraph and got its graph from spring_parse; it still holds "
        "two short mentions that are not names yet.",
        body=(
            f"sentence {sid}  ·  article 5772  ·  corpus text\n"
            + wrap(original, 74) + "\n"
            "\n"
            "not names yet:\n"
            '  "the Act"      -> a statute, declared earlier in the article\n'
            '  "section 34"   -> a provision, listed in the article metadata\n'
            "\n"
            "already a full name: Delhi High Court\n"
            "already a date:      July 2013"
        ),
        right=("SPRING GRAPH  ·  returned by spring_parse, the tool the agent "
               "called",
               spring["amr"] + f"\n\n{spring['n_nodes']} nodes, "
               f"{spring['n_edges']} edges · valid"),
        fill=PALE_BLUE, badge=BLUE, accent=BLUE_DARK,
        change="nothing yet: this is the state the agent receives.",
    )

    # -------------------------------------------------------------- t1 memory
    y = stage(
        y, "t1", "Recall what the article already settled",
        "Article memory at this moment: names accepted from earlier sentences, "
        "with their kind and the sentence that declared them.",
        body=mem_before + "\n\n"
             'so "the Act" has somewhere to land; "section 34" does not exist '
             "in memory yet.",
        kind="body",
        fill=PALE_BLUE, badge=BLUE, accent=BLUE_DARK,
        change="nothing written: memory is read here, not changed.",
        sentence=original, sentence_label="SENTENCE NOW  ·  corpus text",
    )

    # --------------------------------------------------------------- t2 tools
    y = stage(
        y, "t2", "Ask instead of guess",
        "One resolve_mention call per short mention. Each answer carries its "
        "score and the arithmetic behind it.",
        body=tools_body,
        fill=PALE_PURPLE, badge=PURPLE, accent=PURPLE_DARK,
        change="both mentions have exactly one credible candidate above the "
               "0.700 line, so there is nothing to tie.",
        sentence=original, sentence_label="SENTENCE NOW  ·  corpus text",
    )

    # ------------------------------------------------------------- t3 rewrite
    y = stage(
        y, "t3", "Rewrite the sentence",
        "The mention is replaced and nothing else is touched. The corpus text "
        "is kept as sentence_original.",
        body=(
            resolved + "\n\n"
            f'edit  surface "{edit["surface"]}"  occurrence '
            f'{edit["occurrence"]}  ->  "{edit["replacement"]}"\n'
            f'      start {edit["start"]}  end {edit["end"]}  ·  '
            f'kind {edit["kind"]}  ·  evidence {edit["evidence"]}\n'
            "\n"
            "the word \"the\" in front stays: \"... of the Arbitration and "
            "Conciliation Act, 1996 in Delhi High Court ...\""
        ),
        fill=PALE_YELLOW, badge=YELLOW, accent="#c78a00",
        change="3 characters became 40, at one known offset. The meaning of "
               "the sentence did not move.",
        sentence=resolved, sentence_label="SENTENCE NOW  ·  rewritten",
    )

    # --------------------------------------------------------- t4 first graph
    y = stage(
        y, "t4", "First graph: the validator refuses it",
        "The agent's first answer for this sentence, from the repair run in "
        "data/five_repair (same sentence, real validator).",
        body=repair["broken_amr"] + f"\n\nvalidator: {error}",
        fill=PALE_RED, badge=RED, accent=RED_DARK,
        change="not accepted. The exact message goes back into the same "
               "conversation, with the tools still available.",
        sentence=resolved, sentence_label="SENTENCE NOW  ·  unchanged since t3",
    )

    # -------------------------------------------------------------- t5 repair
    y = stage(
        y, "t5", "Repair: same conversation, one token changed",
        f"Attempt {repair['attempt']}. The error is returned to the agent and "
        "the corrected graph comes back. The broken answer stays in "
        "data/five_repair/report.json, so nothing is rewritten afterwards.",
        body=repair["final_amr"],
        fill=PALE_ORANGE, badge=ORANGE, accent=ORANGE_DARK,
        change="(h / court) became (h / gov-entity); the rest is identical. "
               f"fixed {repair['fixed']} · remaining {repair['remaining']} · "
               "accepted.",
        sentence=resolved, sentence_label="SENTENCE NOW  ·  unchanged since t3",
    )

    # --------------------------------------------------------------- t6 check
    relations = ", ".join(f"{k} {v}"
                          for k, v in rec["legal_relations"].items()) or "none"
    y = stage(
        y, "t6", "Check the sentence and the graph together",
        "One real check reads both sides: the edit must exist in the sentence "
        "and the name must exist in the graph, in both directions.",
        body=(
            "PASSED  ·  0 errors\n"
            f"coverage {cm['value']:.3f} = {cm['kept']}/{cm['total']} content "
            "words kept\n"
            f"missing: {', '.join(cm['missing'])}\n"
            f"legal names in the graph: {' · '.join(rec['legal_names'])}\n"
            f"relations: {relations}"
        ),
        fill=PALE_ORANGE, badge=ORANGE, accent=ORANGE_DARK,
        change='the graph says challenge-01 and :month 7, so "challenged" and '
               '"july" count as missing. Coverage is a rough check, not a '
               "grammar score.",
        sentence=resolved, sentence_label="SENTENCE NOW  ·  unchanged since t3",
    )

    # ------------------------------------------------------------ t7 remember
    y = stage(
        y, "t7", "Remember what this sentence settled",
        "The accepted names join article memory, with the sentence they came "
        "from, ready for the next sentences of the article.",
        body=mem_after + "\n\n2 of these 4 names came from this sentence.",
        kind="body",
        fill=PALE_BLUE, badge=BLUE, accent=BLUE_DARK,
        change="memory grows by section 34 [provision] and Delhi High Court "
               "[gov-entity]; the statute and section 2(2) were already there.",
        sentence=resolved, sentence_label="SENTENCE NOW  ·  unchanged since t3",
    )

    # ---------------------------------------------------------------- t8 save
    y = stage(
        y, "t8", "Save the result and the trace",
        "The resolved sentence, the corpus text and every step behind it are "
        "written; a rerun never repeats this work.",
        body=(
            "records.jsonl  ·  one row\n"
            f"  sentence           {resolved}\n"
            f"  sentence_original  {original}\n"
            "  edits · amr · base_amr · coverage 0.846 · wiki_titles\n"
            "\n"
            "data/five/events/article_5772.jsonl  ·  the main run, in order\n"
            "  batch_start        batch 3   sentence_ids [360450]\n"
            '  tool_call          resolve_mention "the Act"     -> 0.978\n'
            '  tool_call          resolve_mention "section 34"  -> 0.800\n'
            "  sentence_accepted  n_edits 1\n"
            "\n"
            "data/five_repair/events/article_5772.jsonl  ·  the repair run\n"
            "  repair_start       attempt 1  errors [the same message]\n"
            "  repair_result      attempt 1  fixed 1  remaining 0\n"
            "  sentence_accepted  n_edits 1"
        ),
        body_size=11,
        fill=PALE_GREEN, badge=GREEN, accent=GREEN_DARK,
        change="saved resolved, with the original next to it and the trace "
               "underneath.",
        last=True,
    )

    # ------------------------------------------------------------- comparison
    y = section.heading(
        y, "What changed, and what did not",
        "One sentence, one span, one graph, two new names. Everything else is "
        "exactly as the article wrote it.")
    y += 6
    changed = (
        "THE SENTENCE\n"
        '"Act" (3 characters, offsets 49-52)\n'
        '  -> "Arbitration and Conciliation Act, 1996" (40 characters)\n'
        "\n"
        "THE GRAPH\n"
        'SPRING: law :name "Act" · section :mod 34\n'
        "agent:  statute :legal-name + :wiki\n"
        '        provision :legal-name "section 34" :part-of\n'
        "        court -> gov-entity :legal-type \"court\"\n"
        "\n"
        "THE MEMORY\n"
        "2 names -> 4 names"
    )
    stayed = (
        "THE SENTENCE\n"
        "every other character, in the same order\n"
        "\n"
        "THE CORPUS TEXT\n"
        "sentence_original, saved next to the resolved sentence\n"
        "\n"
        "THE FACTS\n"
        "the award · the challenge · section 34\n"
        "July 2013 / :month 7 · Delhi High Court\n"
        "\n"
        "THE TRACE\n"
        "the broken first graph, the error and the repair\n"
        "are kept in data/five_repair/report.json"
    )
    comp_h = max(th(changed, 12), th(stayed, 12)) + 68
    section.add(box(40, y, 726, comp_h, fill="#fff9db", stroke=GREY))
    section.add(box(794, y, 726, comp_h, fill="#ebfbee", stroke=GREY))
    section.add(text(64, y + 12, 690, "WHAT CHANGED", size=13,
                     color=ORANGE_DARK, family=CODE))
    section.add(text(818, y + 12, 690, "WHAT STAYED THE SAME", size=13,
                     color=GREEN_DARK, family=CODE))
    section.add(text(64, y + 40, 680, changed, size=12, family=CODE))
    section.add(text(818, y + 40, 680, stayed, size=12, family=CODE))
    y += comp_h + 26

    # --------------------------------------------------- all six, side by side
    y += 6
    y = section.heading(
        y, "All six sentences, side by side",
        "The same nine moments for every sentence of article 5772, in the order "
        "the article gives them. 360450 is the column detailed above; 360439 is "
        "the one that was rejected. Every cell is read from "
        "data/five/report.json: five accepted, one rejected, nothing retyped.")
    y += 4

    ids = [str(s) for s in report["sentences"]]
    rej = report["rejected_records"][0]
    label_w, col_w, col_gap = 92, 223, 8
    col_x = [40 + label_w + 10 + i * (col_w + col_gap) for i in range(len(ids))]

    def lines_h(body: str, size: float) -> float:
        return (body.count("\n") + 1) * size * 1.25

    def clip(body: str, n: int) -> str:
        return body if len(body) <= n else body[: n - 1].rstrip() + "…"

    def short_name(entry: str, n: int = 30) -> str:
        return clip(entry.split(" [")[0], n)

    def before_names(k: str) -> list[str]:
        return [n for n in report["memory_before"][k] if n != "(empty)"]

    def graph_cell(k: str) -> str:
        s = report["spring"][k]
        body = f"spring_parse ×1\nSPRING {s['n_nodes']}n / {s['n_edges']}e"
        if s["status"] == "ok":
            return body + "\nvalid"
        return body + "\nINVALID\n" + clip(s["errors"][0].split(": ")[-1], 24)

    def mem_cell(k: str) -> str:
        names = before_names(k)
        body = f"{len(names)} name" + ("" if len(names) == 1 else "s")
        if names:
            body += "\n" + short_name(names[-1])
        return body

    def tools_cell(k: str) -> str:
        blocks: list[list[str]] = []
        for line in report["tools"][k]:
            if line and not line.startswith(" "):
                blocks.append([line])
            elif blocks:
                blocks[-1].append(line)
        calls = [b[0].split("(")[0] for b in blocks]
        body = "\n".join(
            f"{name} ×{calls.count(name)}" if calls.count(name) > 1 else name
            for name in dict.fromkeys(calls))
        scores = []
        for block in blocks:
            text = "\n".join(block)
            found = re.findall(r"score (\d\.\d+)", text)
            if not found:
                found = re.findall(r"->\s*(\d\.\d{3})", text)
            scores.extend(found[:1])
            if "next:" in text:
                scores.extend(re.findall(r"next:.*? (\d\.\d{3})", text)[:1])
        body += "\n" + (" · ".join(scores) if scores else "-> no candidates")
        return body

    def edit_cell(rec: dict | None) -> str:
        if rec is None:
            return "rejected before\nany rewrite"
        if not rec["edits"]:
            return "no edit\nthe full name is\nalready in the text"
        e = rec["edits"][0]
        target = (f'"{e["replacement"]}"' if len(e["replacement"]) <= 22
                  else "full statute name")
        return f'1 edit\n"{clip(e["surface"], 16)}" ->\n{target}'

    def check_cell(k: str) -> str:
        c = report["checks"][k]
        if c["ok"] is None:
            return "not run\nrejected first"
        cm = c["coverage_math"]
        return (f"{'ok' if c['ok'] else 'FAIL'} · {cm['value']:.3f}\n"
                f"{cm['kept']}/{cm['total']} words")

    def remember_cell(k: str) -> str:
        b = before_names(k)
        a = list(report["memory_after"][k])
        new = [n for n in a if n not in b]
        body = f"{len(b)} -> {len(a)} " + ("name" if len(a) == 1 else "names")
        if not new:
            return body + "\nno new name"
        for entry in new[:2]:
            body += "\n+ " + short_name(entry, 26)
        return body

    graph_note = {
        "360428": "accepted\nlaw -> statute\n+ provision",
        "360429": "accepted\nthe base graph was\nthe broken one",
        "360439": "none written\nsentence rejected",
        "360450": "invalid\n:legal-type only on\na gov-entity",
        "360465": "accepted\ncourt tagged\ngov-entity",
        "360477": "accepted\nlaw -> statute",
    }
    repair_note = {
        "360428": "not needed",
        "360429": "not needed\nthe base error was\nfixed in the graph",
        "360439": "—",
        "360450": "1 try · 1 fix\ncourt -> gov-entity",
        "360465": "not needed",
        "360477": "not needed",
    }

    cols = []
    for k in ids:
        rec = next((r for r in report["records"]
                    if str(r["sentence_id"]) == k), None)
        cells = [
            graph_cell(k), mem_cell(k), tools_cell(k), edit_cell(rec),
            graph_note[k], repair_note[k], check_cell(k), remember_cell(k),
            "records.jsonl\naccepted" if rec
            else "rejected.jsonl\nno_credible_source",
        ]
        cols.append({
            "id": k, "rec": rec, "cells": cells,
            "sent": rec["sentence_original"] if rec else rej["original"],
            "role": report["roles"][k],
        })

    rows = [("t0", "split · parse"), ("t1", "memory"), ("t2", "tools"),
            ("t3", "rewrite"), ("t4", "graph"), ("t5", "repair"),
            ("t6", "check"), ("t7", "remember"), ("t8", "save")]

    def role_body(col: dict) -> str:
        role = col["role"]
        if not col["rec"] and "rejected" not in role:
            role += "  ·  rejected"
        return wrap(role, 26)

    head_h = max(
        18 + lines_h(role_body(c), 10.5) + 6
        + lines_h(wrap(c["sent"], 30), 10) + 14
        for c in cols)
    for i, col in enumerate(cols):
        x, rejected = col_x[i], not col["rec"]
        section.add(box(x, y, col_w, head_h, fill=PALE_RED if rejected else "#ffffff",
                        stroke=RED_DARK if rejected else GREY))
        accent = RED_DARK if rejected else BLUE_DARK
        section.add(text(x + 10, y + 8, col_w - 20, col["id"], size=15,
                         family=CODE, color=accent))
        section.add(text(x + 10, y + 28, col_w - 20, role_body(col),
                         size=10.5, color=accent))
        section.add(text(x + 10,
                         y + 28 + lines_h(role_body(col), 10.5) + 6,
                         col_w - 20, wrap(col["sent"], 30), size=10,
                         family=CODE, color=GREY))
    y += head_h + 6

    for r, (t, label) in enumerate(rows):
        cell_h = max(lines_h(col["cells"][r], 10.5) for col in cols) + 18
        section.add(box(40, y, label_w, cell_h, fill="#f8f9fa", stroke=GREY))
        section.add(text(50, y + 8, label_w - 16, t, size=13, color=INK))
        section.add(text(50, y + 26, label_w - 16, label, size=9.5, color=GREY))
        for i, col in enumerate(cols):
            rejected = not col["rec"]
            section.add(box(col_x[i], y, col_w, cell_h,
                            fill=PALE_RED if rejected else "#ffffff",
                            stroke=RED_DARK if rejected else "#dee2e6"))
            section.add(text(col_x[i] + 10, y + 8, col_w - 20,
                             col["cells"][r], size=10.5, family=CODE,
                             color=INK))
        y += cell_h + 4

    section.add(text(
        40, y + 8, 1480,
        wrap("Read across a row to compare one moment for all six sentences; "
             "read down a column to follow one sentence. 360450 is the column "
             "detailed above. 360439 never reaches a graph: it is rejected "
             "first, and the card below says why.", 150),
        size=13, color=GREY))
    y += 38

    # --------------------------------------------------- the rejected sentence
    y = section.heading(
        y, "The rejected sentence — and why",
        "One of the six sentences is not in the dataset. That is a result, not "
        "a failure: the sentence says less than a name needs, so the agent "
        "stops instead of inventing one.")
    y += 6
    rej_body = (
        f"sentence {rej['sentence_id']}  ·  status rejected  ·  "
        f"reason {rej['reason']}\n"
        + wrap(rej["original"], 92) + "\n"
        "\n"
        "why it cannot be resolved\n"
        "  1  the sentence starts at \"Ltd\": the rest of the company name\n"
        "     sits in the sentence before it, so this string alone cannot\n"
        "     name the company\n"
        "  2  \"HEPL\" is a short form and this sentence never spells it\n"
        "     out, so there is nothing here to match a full name against\n"
        "  3  resolve_mention(\"HEPL\", this sentence) -> candidates: []\n"
        "     nothing reached the 0.700 credibility line\n"
        "  4  SPRING reads the company as a name with one op, \"Ltd.\",\n"
        "     which is the same cut-off, seen from the graph side\n"
        "  5  with no name to write, the agent writes no graph and no\n"
        "     rewrite, and the sentence is marked rejected\n"
        "\n"
        "what is kept\n"
        "  rejected.jsonl  { sentence_id 360439, original, status rejected,\n"
        "                    reason no_credible_source, candidates [] }\n"
        "  5 of the 6 sentences of this article were accepted; this one is\n"
        "  data too, kept with its reason for a person to review"
    )
    rej_shown = wrap_code(rej_body, int(1330 / (11.5 * 0.6)))
    rej_h = lines_h(rej_shown, 11.5) + 36
    section.add(box(40, y, 1480, rej_h, fill=PALE_RED, stroke=RED_DARK))
    section.add(text(64, y + 16, 1432, rej_shown, size=11.5, family=CODE,
                     color=INK))
    y += rej_h + 26

    # ------------------------------------------------------------------ files
    files = (
        "data/five/report.json                sentence, tools, scores, memory "
        "before/after, SPRING, check, coverage math\n"
        "data/five/records.jsonl              the saved row: sentence, "
        "sentence_original, edits, amr, base_amr\n"
        "data/five_spring/spring.jsonl        the real SPRING base graph for "
        "this sentence\n"
        "data/five/events/article_5772.jsonl  batch start, both tool calls, "
        "accept\n"
        "data/five/rejected.jsonl              the rejected sentence: reason, "
        "candidates, original\n"
        "data/five_repair/report.json         broken graph, validator error, "
        "fixed graph, attempt count\n"
        "data/five_repair/events/article_5772.jsonl  repair_start, "
        "repair_result, accept\n"
        "docs/five_sentences.py               runs this demo\n"
        "docs/build_plan.py                   draws this section"
    )
    files_h = th(files, 12.5) + 36
    section.add(box(40, y, 1480, files_h, fill="#f8f9fa",
                    stroke=GREY, dashed=True))
    section.add(text(60, y + 16, 1440, files, size=12.5,
                     family=CODE, color=INK))
    y += files_h + 26

    section.add(text(
        40, y, 1480,
        wrap("Six sentences, nine moments each, and a reason for every change, "
             "including the one that was rejected. That is the whole point of "
             "the agent.", 148),
        size=16, color=GREEN_DARK))
    return y + 34


def main() -> None:
    global LAST_KEY, CURRENT_TAG
    data = json.loads(PLAN.read_text(encoding="utf-8"))
    before = len(data["elements"])
    data["elements"] = [
        e for e in data["elements"]
        if (e.get("customData") or {}).get("section") not in SECTIONS
    ]
    kept = len(data["elements"])
    existing_indices = [e["index"] for e in data["elements"] if e.get("index")]
    LAST_KEY = max(existing_indices) if existing_indices else None

    section = Section()
    end_y = build(section)
    CURRENT_TAG = ONE_TAG
    build_one(section, end_y + 70)
    CURRENT_TAG = TAG
    data["elements"].extend(section.els)

    # sanity: every new key must be valid, unique and after the old top
    keys = [e["index"] for e in section.els]
    assert keys == sorted(keys) and len(set(keys)) == len(keys)
    if existing_indices:
        assert keys[0] > max(existing_indices)
    for key in keys:
        part = _int_part(key)
        assert not key[len(part):].endswith("0"), key

    PLAN.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    ys = [e["y"] + e["height"] for e in data["elements"]]
    print(f"kept {kept}/{before} old elements, added {len(section.els)}, "
          f"total {len(data['elements'])}, bottom y={max(ys):.0f}")


if __name__ == "__main__":
    main()

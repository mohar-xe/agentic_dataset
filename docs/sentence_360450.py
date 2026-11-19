#!/usr/bin/env python3
"""Draw one sentence from start to finish, as a separate Excalidraw file.

The sentence is the real demo sentence 360450 from article 5772::

    The award was challenged under section 34 of the Act in Delhi High Court
    in July 2013.

Everything shown is what the real run produced: the rough SPRING graph, the
two lookups ("the Act" from memory, "section 34" from article metadata), the
edits made to that graph, the final graph, the resolved sentence, the check
(11 of 13 words kept), and the memory after the sentence.

Plain language on purpose: the graphs are shown as they were returned and
saved, and every change between them is listed in words.

    python docs/sentence_360450.py

Writes ``docs/sentence_360450.excalidraw`` (a standalone scene; the main plan
drawing is untouched). Rerunning rewrites the same file.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "sentence_360450.excalidraw"

INK = "#1e1e1e"
GREY = "#495057"
CX, CW = 280, 640
GX, GW = 100, 1000
MID = CX + CW / 2

FONT = 1          # Excalidraw's hand-drawn family
CODE = 3          # monospaced, for the graphs themselves
SIZE = 18
GCODE = 14
rng = random.Random(360450)
COUNT = 0
ELEMENTS: list[dict] = []

BASE_GRAPH = """(z1 / challenge-01
      :ARG1 (z2 / award-01)
      :ARG2 (z3 / section
            :mod 34
            :part-of (z4 / law
                  :name (z5 / name
                        :op1 "Act")))
      :location (z6 / government-organization
            :name (z7 / name
                  :op1 "Delhi"
                  :op2 "High"
                  :op3 "Court"))
      :time (z8 / date-entity
            :month 7
            :year 2013))"""

FINAL_GRAPH = """(c / challenge-01
   :ARG1 (a / award)
   :prep-under (p / provision
                  :legal-level "section"
                  :legal-name "section 34"
                  :legal-part-of (s / statute
                                    :wiki "Arbitration_and_Conciliation_Act,_1996"
                                    :legal-name "Arbitration and Conciliation Act, 1996"))
   :location (h / gov-entity
                :legal-type "court"
                :legal-name "Delhi High Court")
   :time (d / date-entity
            :year 2013
            :month 7))"""


def _base(kind: str, x: float, y: float, w: float, h: float,
          stroke: str = INK, fill: str = "transparent") -> dict:
    global COUNT
    COUNT += 1
    return {
        "id": f"d{COUNT:03d}",
        "type": kind,
        "x": round(x, 2),
        "y": round(y, 2),
        "width": round(w, 2),
        "height": round(h, 2),
        "angle": 0,
        "strokeColor": stroke,
        "backgroundColor": fill,
        "fillStyle": "solid",
        "strokeWidth": 1 if kind == "text" else 2,
        "strokeStyle": "solid",
        "roughness": 0,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": {"type": 3} if kind == "rectangle" else None,
        "seed": rng.randint(1, 2 ** 31 - 1),
        "version": 1,
        "versionNonce": rng.randint(1, 2 ** 31 - 1),
        "isDeleted": False,
        "boundElements": [],
        "updated": 1794860000000,
        "link": None,
        "locked": False,
    }


def text(x: float, y: float, w: float, body: str, size: float = SIZE,
         color: str = INK, family: int = FONT) -> None:
    lines = body.count("\n") + 1
    el = _base("text", x, y, w, lines * size * 1.25, stroke=color)
    el.update({
        "text": body,
        "originalText": body,
        "fontSize": size,
        "fontFamily": family,
        "textAlign": "left",
        "verticalAlign": "top",
        "containerId": None,
        "lineHeight": 1.25,
        "baseline": y + size * 0.9,
    })
    ELEMENTS.append(el)


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str) -> None:
    ELEMENTS.append(_base("rectangle", x, y, w, h, stroke=stroke, fill=fill))


def arrow(x: float, y: float, dx: float, dy: float) -> None:
    el = _base("arrow", x, y, abs(dx), abs(dy), stroke=GREY)
    el.update({
        "points": [[0, 0], [round(dx, 2), round(dy, 2)]],
        "lastCommittedPoint": None,
        "startBinding": None,
        "endBinding": None,
        "startArrowhead": None,
        "endArrowhead": "arrow",
        "elbowed": False,
    })
    ELEMENTS.append(el)


def wrap(body: str, width: int) -> str:
    """Greedy word wrap, so box heights match the drawn lines exactly."""
    out: list[str] = []
    for para in body.split("\n"):
        line = ""
        for word in para.split():
            if not line:
                line = word
            elif len(line) + 1 + len(word) <= width:
                line += " " + word
            else:
                out.append(line)
                line = word
        out.append(line)
    return "\n".join(out)


def step(y: float, num: str, title: str, body: str, fill: str, stroke: str,
         dark: str, chars: int = 64) -> float:
    text(CX, y, CW, f"{num} · {title}", SIZE, dark)
    wrapped = wrap(body, chars)
    h = max(70.0, (wrapped.count("\n") + 1) * SIZE * 1.25 + 32)
    rect(CX, y + 26, CW, h, fill, stroke)
    text(CX + 20, y + 42, CW - 40, wrapped, SIZE, INK)
    return y + 26 + h


def card(x: float, y: float, w: float, title: str, body: str, fill: str,
         stroke: str, dark: str) -> float:
    wrapped = wrap(body, 50)
    h = 40 + (wrapped.count("\n") + 1) * 20 + 16
    rect(x, y, w, h, fill, stroke)
    text(x + 18, y + 12, w - 36, title, 17, dark)
    text(x + 18, y + 38, w - 36, wrapped, 16, INK)
    return h


def graph(y: float, num: str, title: str, note: str, code: str,
          fill: str, stroke: str, dark: str) -> float:
    text(CX, y, CW, f"{num} · {title}", SIZE, dark)
    y += 26
    text(CX + 4, y, CW - 8, note, 16, GREY)
    y += 24
    h = (code.count("\n") + 1) * GCODE * 1.25 + 32
    rect(GX, y, GW, h, fill, stroke)
    text(GX + 20, y + 16, GW - 40, code, GCODE, INK, CODE)
    return y + h


def build() -> float:
    text(CX, 40, CW, "One sentence, start to finish", 30, INK)
    text(CX, 84, CW, "article 5772 · sentence 360450 · real demo data", 17, GREY)

    y = 130.0
    y = step(y, "1", "the original sentence",
             "The award was challenged under section 34 of the Act in Delhi "
             "High Court in July 2013.",
             "#a5d8ff", "#4a9eed", "#1971c2")
    arrow(MID, y + 4, 0, 36)
    y += 40
    y = graph(y, "2", "ask the local parser for a first graph",
              "SPRING read this sentence alone and returned this. It is the "
              "rough starting point — a place to begin, not the answer.",
              BASE_GRAPH, "#d0bfff", "#8b5cf6", "#6741d9")

    y += 8
    arrow(MID, y, 0, 34)
    y += 40
    y = step(y, "3", "check the two unclear names",
             "Two mentions need checking: “the Act” and “section 34”.\n"
             "The agent looks in the article, not on the internet.",
             "#fff3bf", "#f59e0b", "#b45309")

    card_y = y + 40
    arrow(CX + 100, y + 6, -110, 34)
    arrow(CX + CW - 100, y + 6, 110, 34)
    left_h = card(60, card_y, 460, "Look up “the Act”",
                  "Found in memory: Arbitration and Conciliation Act, 1996.\n"
                  "Written in full 22 sentences earlier, with the short name "
                  "“Act”.",
                  "#c3fae8", "#06b6d4", "#0b7285")
    card(680, card_y, 460, "Look up “section 34”",
         "Not in memory yet.\n"
         "Listed in the article’s own metadata, so the agent can keep it.",
         "#c3fae8", "#06b6d4", "#0b7285")

    arrow(MID, card_y + left_h, 0, 34)
    y = card_y + left_h + 40
    y = step(y, "4", "the agent decides",
             "“the Act” → Arbitration and Conciliation Act, 1996\n"
             "“section 34” → stays “section 34”\n"
             "“Delhi High Court” → already a full name, no change",
             "#fff3bf", "#f59e0b", "#b45309")

    y += 8
    arrow(MID, y, 0, 34)
    y += 40
    y = step(y, "5", "edit the graph, one fact at a time",
             "• “law” named “Act” → statute, full name + wiki title\n"
             "• “section” + number 34 → provision “section 34”, linked to it\n"
             "• “government-organization” → court, named “Delhi High Court”\n"
             "• “award-01” (the act of awarding) → “award” (the thing)\n"
             "• “under” → the legal link “prep-under”\n"
             "• kept as it was: the root “challenge”, the court, July 2013",
             "#fff3bf", "#f59e0b", "#b45309", chars=72)

    y += 8
    arrow(MID, y, 0, 34)
    y += 40
    y = graph(y, "6", "the final graph, saved with the sentence",
              "Same facts as the rough graph. What changed is the legal "
              "labelling and the full names.",
              FINAL_GRAPH, "#b2f2bb", "#22c55e", "#2f9e44")

    y += 8
    arrow(MID, y, 0, 34)
    y += 40
    y = step(y, "7", "write the resolved sentence",
             "The award was challenged under section 34 of the Arbitration and "
             "Conciliation Act, 1996 in Delhi High Court in July 2013.\n"
             "The change in the graph and the change in the sentence say the "
             "same thing.",
             "#b2f2bb", "#22c55e", "#2f9e44", chars=62)

    y += 8
    arrow(MID, y, 0, 34)
    y += 40
    y = step(y, "8", "check the answer",
             "Graph valid: yes.   Words kept: 11 of 13 (85%).\n"
             "The original sentence is kept too; nothing was invented.",
             "#c3fae8", "#06b6d4", "#0b7285")

    y += 8
    arrow(MID, y, 0, 34)
    y += 40
    y = step(y, "9", "remember it for the next sentences",
             "Memory now holds 4 names:\n"
             "• Arbitration and Conciliation Act, 1996 — the law\n"
             "• section 2(2) — a provision\n"
             "• section 34 — a provision (new)\n"
             "• Delhi High Court — a court (new)",
             "#c3fae8", "#06b6d4", "#0b7285")

    y += 8
    arrow(MID, y, 0, 34)
    y += 40
    y = step(y, "10", "saved result",
             "Saved together:\n"
             "• the resolved sentence above\n"
             "• the final graph above\n"
             "• the original sentence (kept, never replaced)",
             "#b2f2bb", "#22c55e", "#2f9e44")

    y += 26
    text(CX, y, CW, "blue = input · purple = parser · yellow = decision and "
                    "edits · green = result · teal = check and memory", 16, GREY)
    y += 26
    text(CX, y, CW, "This is one sentence of the article’s 53; the next "
                    "sentence starts with the memory above.", 16, GREY)
    return y


def main() -> None:
    bottom = build()
    scene = {
        "type": "excalidraw",
        "version": 2,
        "source": "https://excalidraw.com",
        "elements": ELEMENTS,
        "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
        "files": {},
    }
    OUT.write_text(json.dumps(scene, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"wrote {OUT.name}: {len(ELEMENTS)} elements, bottom y={bottom:.0f}")


if __name__ == "__main__":
    main()

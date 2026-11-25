#!/usr/bin/env python3
"""Add the Legal-AMR Lean Core v1.1 panel beside the existing drawing.

The older diagrams are never touched. This script only replaces elements it
created itself (``customData.section == "lean-amr-v11"``) and appends a new
column to the right of the current scene.

    python docs/build_lean_extension.py

The text is deliberately plain: what each label means, one everyday example,
what was removed and why, and what the pipeline now checks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_plan as bp  # noqa: E402

PLAN = ROOT / "agentic_amr_dataset_final.excalidraw"
TAG = "lean-amr-v11"

INK = "#1e1e1e"
GREY = "#495057"
BLUE = "#a5d8ff"
BLUE_DARK = "#1971c2"
YELLOW = "#ffec99"
YELLOW_DARK = "#b08900"
GREEN = "#b2f2bb"
GREEN_DARK = "#2f9e44"
RED = "#ffc9c9"
RED_DARK = "#c92a2a"
PURPLE = "#d0bfff"
PURPLE_DARK = "#6741d9"
ORANGE = "#ffd8a8"
ORANGE_DARK = "#d9480f"
PAPER = "#f8f9fa"

FONT = 8
CODE = 3
XGAP = 120
W = 1480


class Panel:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.els: list[dict] = []

    def add(self, elements: list[dict]) -> None:
        self.els.extend(elements)

    def title(self, title: str, note: str) -> None:
        self.add(bp.text(self.x, self.y, W, title, size=34, color=INK))
        self.y += 46
        body = bp.wrap(note, 150)
        self.add(bp.text(self.x, self.y, W, body, size=16, color=GREY))
        self.y += (body.count("\n") + 1) * 20 + 18

    def heading(self, title: str, note: str = "") -> None:
        self.add(bp.text(self.x, self.y, W, title, size=23, color=BLUE_DARK))
        self.y += 34
        if note:
            body = bp.wrap(note, 150)
            self.add(bp.text(self.x, self.y, W, body, size=14, color=GREY))
            self.y += (body.count("\n") + 1) * 17.5 + 8
        self.y += 6

    def boxed(self, fill: str, stroke: str, lines: list[str],
              size: float = 14.5, pad: float = 18,
              family: int = FONT, chars: int = 142) -> None:
        body = bp.wrap("\n".join(lines), chars)
        h = (body.count("\n") + 1) * size * 1.3 + pad * 2
        self.add(bp.box(self.x, self.y, W, h, fill=fill, stroke=stroke))
        self.add(bp.text(self.x + pad, self.y + pad, W - pad * 2, body,
                         size=size, family=family, color=INK))
        self.y += h + 18

    def rows(self, rows: list[tuple[str, str, str, str]],
             fill: str, stroke: str, code_col: float = 300) -> None:
        """Rows of (code, plain meaning, tiny example), each on one card."""
        for i, (code, meaning, example, tone) in enumerate(rows):
            h = 58
            card_fill = fill if i % 2 == 0 else "#ffffff"
            self.add(bp.box(self.x, self.y, W, h, fill=card_fill, stroke=stroke))
            self.add(bp.text(self.x + 16, self.y + 9, code_col, code,
                             size=14, family=CODE, color=tone))
            example_x = self.x + W - 360
            meaning_x = self.x + 16 + code_col + 16
            self.add(bp.text(meaning_x, self.y + 9, example_x - meaning_x - 16,
                             meaning, size=13.5, color=INK))
            self.add(bp.text(example_x, self.y + 9, 344, example,
                             size=12.5, color=GREY))
            self.y += h + 6
        self.y += 14

    def code(self, body: str, size: float = 12.5) -> None:
        shown = bp.wrap(body, 116)
        h = (shown.count("\n") + 1) * size * 1.35 + 34
        self.add(bp.box(self.x, self.y, W, h, fill=YELLOW, stroke=YELLOW_DARK))
        self.add(bp.text(self.x + 18, self.y + 16, W - 36, shown,
                         size=size, family=CODE, color=INK))
        self.y += h + 18


def build(panel: Panel) -> None:
    panel.title(
        "Legal-AMR Lean Core v1.1  ·  the new legal labels",
        "This panel sits beside the older diagrams and changes none of them. "
        "It explains, in plain words, the small set of legal labels the "
        "pipeline now follows, with one example for each and the reasons the "
        "older extra labels were dropped."
    )

    panel.boxed(BLUE, BLUE_DARK, [
        "THE ONE RULE",
        "Add a label only when ordinary AMR cannot already say the same "
        "thing. If ordinary AMR can say it, do not add a legal label.",
        "",
        "AMR is just a picture of a sentence: words become nodes and "
        "labelled arrows. Everything here is added on top of that picture. "
        "Nothing in ordinary AMR is renamed or redefined. Remove every "
        ":legal-* link and the graph is still an ordinary AMR graph.",
    ])
    panel.y += 4

    panel.heading(
        "1 · The 7 things a graph can name",
        "A legal text, a piece of a legal text, a court or ministry, and a "
        "court decision. Each is a plain noun. The kind of provision or body "
        "is kept as a detail (see section 3), so seven names do the work."
    )
    panel.rows([
        ("statute", "A primary law: an Act, a Code, the Constitution.",
         "Companies Act, 2013", BLUE_DARK),
        ("regulation", "A rule, order or set of rules made under a law.",
         "Companies (Share Capital) Rules, 2014", PURPLE_DARK),
        ("provision", "One citable piece of a law: section, article, clause, "
         "schedule, chapter, part.", "Section 197", BLUE_DARK),
        ("gov-entity", "A government body: ministry, department, court, "
         "tribunal, statutory body, state government, local authority.",
         "Delhi High Court", GREEN_DARK),
        ("judgment", "A court decision.", "2020 SCC OnLine SC 123", RED_DARK),
        ("notification", "An official gazette notice.",
         "S.O. 1234(E)", ORANGE_DARK),
        ("legal-instrument", "Any other citable legal text: a circular, an "
         "office memo, draft rules; also a safe fallback.",
         "CBDT Circular No. 17/2015", GREY),
    ], GREEN, GREEN_DARK)

    panel.heading(
        "2 · The 12 links between them",
        "Read each link from left to right as a short sentence. The middle "
        "column is what it means in everyday language; the right column is a "
        "tiny example."
    )
    panel.rows([
        (":legal-cites", "This text refers to, or is brought under, that one.",
         "Judgment cites Section 34", BLUE_DARK),
        (":legal-overrules", "A later judgment cancels an earlier one.",
         "2017 decision overrules 1967 decision", RED_DARK),
        (":legal-amends", "One law changes another.",
         "Amendment Act changes Section 2", YELLOW_DARK),
        (":legal-repeals", "One law cancels another.",
         "New Sanhita repeals the old Code", RED_DARK),
        (":legal-notifies", "A body formally issues a notice or rule.",
         "Ministry notifies S.O. 1234(E)", ORANGE_DARK),
        (":legal-implements", "A rule puts a parent law into practice.",
         "Rules implement the Companies Act", GREEN_DARK),
        (":legal-applies-to", "This provision covers this subject or body.",
         "Section 138 applies to cheques", BLUE_DARK),
        (":legal-exempts", "This provision makes an exception for them.",
         "Section 8 exempts a class of companies", ORANGE_DARK),
        (":legal-empowers", "This provision gives power to a body.",
         "Section 3 empowers SEBI", PURPLE_DARK),
        (":legal-delegates-to", "One body hands authority to another.",
         "Centre delegates to the State", PURPLE_DARK),
        (":legal-administered-by", "This law is run by this body.",
         "Companies Act run by the MCA", GREEN_DARK),
        (":legal-defines", "This provision defines a term.",
         "Section 2 defines 'company'", BLUE_DARK),
    ], YELLOW, YELLOW_DARK, code_col=320)

    panel.heading(
        "3 · The 11 details written on a thing (plus one optional)",
        "These are not links between things; they are facts about one thing. "
        "They are short strings, numbers or dates."
    )
    panel.boxed(PURPLE, PURPLE_DARK, [
        ":legal-name  the full name, kept exactly as written  "
        "·  \"Companies Act, 2013\"",
        ":legal-level  what kind of provision  "
        "·  section, article, clause, sub-section, schedule, chapter, part",
        ":legal-type  what kind of body  "
        "·  ministry, department, court, tribunal, statutory-body, "
        "state-government, local-authority",
        ":legal-year  the year in the name  ·  2013",
        ":legal-jurisdiction  where it applies  ·  \"India\", "
        "\"State of Kerala\"",
        ":legal-citation  the case citation  ·  \"AIR 1973 SC 1461\"",
        ":legal-status  in-force, repealed, struck-down, amended, "
        "not-yet-in-force",
        ":legal-valid-from / :legal-valid-until  the dates it is in force  "
        "·  \"2024-07-01\" / \"2024-06-30\"",
        ":legal-enacted-on  the date it was enacted  ·  \"1860-10-06\"",
        ":legal-notified-on  the date a notification was published  "
        "·  \"2014-03-27\"",
        "optional  :legal-weight  how strong a citation is  "
        "·  binding, persuasive, obiter",
    ], size=13.5)

    panel.heading(
        "4 · What was removed, and why",
        "The v1.1 rule is strict: if ordinary AMR already says it, the extra "
        "label is removed. Five labels removed, and one old mistake fixed."
    )
    panel.boxed(RED, RED_DARK, [
        "removed  :legal-relies-on  —  now :legal-cites + :legal-weight "
        "\"binding\". One link, same meaning.",
        "removed  :legal-governs  —  now :legal-applies-to. The two meant "
        "the same thing.",
        "removed  :legal-part-of  —  ordinary AMR already has :part-of. "
        "Use that instead.",
        "removed  :legal-antecedent  —  \"the said Act\" now simply points "
        "to the same node as the Act it means. No extra link.",
        "removed  :legal-assented-on  —  in Indian law the assent date and "
        "the enactment date are normally the same, so it added nothing.",
        "fixed  a judgment no longer gets an actor arrow directly (:ARG0). "
        "Say \"the court rendered the judgment\" and link the court to "
        "render-01 instead.",
    ], size=13.5)

    panel.heading(
        "5 · One worked example, in plain words",
        "The sentence: \"The Supreme Court held that Section 197 CrPC "
        "protects public servants.\" In the graph below, each line is one "
        "arrow: a thing, a label, and what it points to. Text in quotes is a "
        "value, not a node."
    )
    panel.boxed(GREEN, GREEN_DARK, [
        "What the graph says, step by step:",
        "1. The thing saved is a judgment.",
        "2. The Supreme Court rendered that judgment.",
        "3. The judgment cites Section 197.",
        "4. Section 197 is part of the Code of Criminal Procedure, 1973.",
        "5. That section applies to public servants.",
    ], size=14)
    panel.code(
        '(h / judgment\n'
        '   :legal-citation "2020 SCC OnLine SC 123"\n'
        '   :ARG1-of (r / render-01\n'
        '              :ARG0 (c / gov-entity\n'
        '                       :legal-type "court"\n'
        '                       :legal-name "Supreme Court of India"))\n'
        '   :legal-cites (s / provision\n'
        '                   :legal-level "section"\n'
        '                   :legal-name "Section 197"\n'
        '                   :part-of (c2 / statute\n'
        '                               :legal-name "Code of Criminal '
        'Procedure, 1973")\n'
        '                   :legal-applies-to (p / public-servant)))'
    )
    panel.boxed(GREEN, GREEN_DARK, [
        "Why this shape: a judgment is a thing, not an action, so it does "
        "not get an actor arrow directly. The court is linked through the "
        "action \"render\". The legal links then connect the judgment to the "
        "provision it uses, the provision to its parent law, and the "
        "provision to the people it covers.",
    ], size=14)

    panel.heading(
        "6 · What the pipeline now checks",
        "The checker is mechanical. It rejects a graph that breaks these "
        "rules, and the model is asked to repair it before anything is saved."
    )
    panel.boxed(ORANGE, ORANGE_DARK, [
        ":legal-level may appear only on a provision.",
        ":legal-type may appear only on a gov-entity.",
        "Do not put an actor or object arrow (:ARG0-:ARG5) directly on a "
        "legal noun (a statute, a judgment, a court). Say who did what with "
        "a verb, such as \"the court rendered the judgment\", and link the "
        "court to that verb.",
        "The removed labels are rejected with a hint telling the model what "
        "to write instead.",
        "Link directions are checked: only a judgment overrules a judgment; "
        "only a regulation implements a statute; only a provision empowers "
        "a body.",
        "The graph must still be one connected picture with no loops, and "
        "the legal labels must stay extra, never replacements.",
    ], size=13.5)

    panel.heading(
        "7 · What changed in this project",
        "The old project is no longer needed at run time. Everything below "
        "lives in agentic_dataset/."
    )
    panel.boxed(PAPER, GREY, [
        "legal_schema.py  —  the v1.1 checker: 7 names, 12 links, 11 details.",
        "common.py  —  local AMR checks and helpers (no import from the old "
        "project).",
        "prompt.py  —  tells the model the v1.1 rules, including \"link a "
        "court to the action, not directly to the judgment\".",
        "data/india.db  —  India-only corpus: the German material is filtered "
        "out; Indian sources are kept.",
        "make_india_db.py  —  rebuilds data/india.db from a full corpus.",
        "docs/five_sentences.py and docs/sentence_360450.py  —  demos now use "
        "ordinary :part-of and :legal-cites.",
        "README.md  —  new run and configuration notes.",
        "The older drawing above is kept as history: its worked example still "
        "shows the v1.0 labels. The current demos and the working graph use "
        "the v1.1 shape shown in section 5.",
    ], size=13.5)

    panel.boxed(BLUE, BLUE_DARK, [
        "The older diagrams above are unchanged. This panel is extra too: "
        "removing it leaves every other drawing exactly as it was.",
    ], size=14)

    panel.add(bp.text(
        panel.x, panel.y, W,
        "Where to read the full specification: "
        "extension/Legal-AMR-Lean-Core-v1.1.md",
        size=14, family=CODE, color=GREY,
    ))
    panel.y += 30


def main() -> None:
    data = json.loads(PLAN.read_text(encoding="utf-8"))
    before = len(data["elements"])
    old = [e for e in data["elements"]
           if (e.get("customData") or {}).get("section") == TAG]
    data["elements"] = [e for e in data["elements"]
                        if (e.get("customData") or {}).get("section") != TAG]
    kept = len(data["elements"])

    xs = [e.get("x", 0) + e.get("width", 0) for e in data["elements"]]
    x = max(xs) + XGAP if xs else 40
    bp.CURRENT_TAG = TAG
    bp.INDEX = 0
    indices = [e["index"] for e in data["elements"] if e.get("index")]
    bp.LAST_KEY = max(indices) if indices else None

    panel = Panel(x, 40)
    build(panel)
    data["elements"].extend(panel.els)

    keys = [e["index"] for e in panel.els]
    assert keys == sorted(keys) and len(set(keys)) == len(keys)
    if indices:
        assert keys[0] > max(indices)

    PLAN.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print(f"kept {kept}/{before} existing elements (removed {len(old)} old "
          f"panel elements), added {len(panel.els)}, total "
          f"{len(data['elements'])}, panel x={x:.0f}, bottom y={panel.y:.0f}")


if __name__ == "__main__":
    main()

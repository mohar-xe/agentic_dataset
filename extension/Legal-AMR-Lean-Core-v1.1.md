# Legal-AMR (v1.1)

> **Scope:** A minimal, additive extension of Abstract Meaning Representation (AMR) for Indian legal texts (statutes, regulations, judgments, notifications).

> 1. **Necessity test (mandatory).** Add a tag only if core AMR cannot express the relation. Every tag in this spec carries a one-sentence necessity argument; if the argument fails, the tag is removed.
> 2. **AMR validity test (mandatory).** Every tag must satisfy: (a) concepts are bare nouns with no `-NN` suffix; (b) relations are `:legal-*` namespaced; (c) attributes are string/integer literals; (d) no `:ARGn` is attached to a non-predicate node.
> 3. **Example test (mandatory).** Every tag has a worked PENMAN example that is balanced, single-rooted, and whose remaining edges are all valid core AMR relations when `:legal-*` edges are stripped.

---

## 1. Design Overview

This document strips Legal-AMR down to the smallest tagset that still covers the most common legal-semantic phenomena in Indian legal texts: citation, normative effect (amend/repeal/notify/implement), competence (empower/delegate), containment, temporal validity, and definitional structure.

Two reductions drive the cut:

1. **Concept collapse.** Seven provision sub-types (section, article, clause, sub-section, schedule, chapter, part) and seven government sub-types (ministry, department, court, tribunal, statutory-body, state-government, local-authority) are folded into two concepts — `provision` and `gov-entity` — each carrying its former sub-type as an attribute (`:legal-level`, `:legal-type`). This removes 14 concept types at the cost of two string attributes.

2. **Relation folding.** Specialized containment roles collapse into core AMR `:part-of` (not a `:legal-*` role); the three v1.0 citation roles collapse into one `:legal-cites` role (strength via `:legal-weight`); the definition pair keeps only `:legal-defines` (inverse is queryable); competence pairs merge into `:legal-empowers`. Anaphora is handled by core AMR reentrancy, not a dedicated role.

**Necessity gate (v1.1).** Before any tag enters the inventory, it must answer *no* to: *"Can core AMR (`:ARG0`–`:ARG5`, `:ARGn-of`, `:mod`, `:poss`, `:part-of`, `:name`, `:location`, `:time`, `:opN`, `:polarity`, `:ARGM-*`, reentrancy) already express this?"* If yes, it does not enter. This gate pruned four v1.0 relations and one v1.0 optional attribute.

### Kinds of additions

| Kind | Purpose | Notation |
| :--- | :--- | :--- |
| **Concept Types** | Typed nodes for legal instruments and bodies | Node labels, e.g. `(s / statute)` |
| **Relation Tags** | Edges encoding normative, referential, and authority relations | `:legal-relation-name` |
| **Attribute Tags** | Literal-valued properties attached to nodes | `:legal-attribute-name "value"` |

### Namespacing (mandatory)

All extension roles and attributes are prefixed `:legal-`. This keeps them disjoint from core AMR roles and from PropBank framesets, so every Legal-AMR graph degrades gracefully: deleting every `:legal-*` edge leaves a graph whose remaining edges are all valid core AMR relations.

### AMR validity guarantees (v1.1, tightened)

1. **Additivity.** No core AMR role or concept is redefined. New concepts are plain entity nouns (`statute`, `regulation`, `provision`, `gov-entity`, `judgment`, `notification`, `legal-instrument`) that do not collide with any PropBank frameset (which carry a `-NN` suffix, e.g. `amend-01`).
2. **Single root.** Every example graph below is a single-rooted Penman tree.
3. **Balanced structure.** All code blocks are balanced Penman.
4. **No `:ARGn` on nouns.** `:ARG0`–`:ARG5` appear only on PropBank predicates (`-NN`-suffixed concepts) or via the reverse `:ARGn-of` form. Noun concepts (`judgment`, `statute`, etc.) link to events through `:ARGn-of (event)` reentrancies.
5. **Degradation.** Stripping all `:legal-*` edges leaves a graph whose remaining edges are all valid core AMR relations (`:ARG0`–`:ARG5`, `:ARGn-of`, `:mod`, `:poss`, `:part-of`, `:name`, `:location`, `:time`, `:opN`, `:polarity`, `:ARGM-*`). Each connected component of the stripped graph is itself single-rooted and balanced.

---

## 2. Concept Types (7)

All are typed nouns replacing generic `(act)` / `(government)` nodes. They never carry a PropBank-style `-NN` suffix, so they cannot shadow a frameset.

### 2.1 `statute`

- **Replaces:** `(act)`, `(law)`
- **Necessity:** Core AMR has no concept for primary legislation. `(act)` and `(law)` are not registered AMR concepts. Statutes need a typed node for citation-graph queries (find all amendments of the Companies Act, 2013).
- **AMR validity:** Bare noun, no `-NN` suffix, no collision with any PropBank frameset.
- **Example:**
```lisp
(s / statute
   :legal-name "Indian Penal Code, 1860"
   :legal-year 1860
   :legal-jurisdiction "India")
```

### 2.2 `regulation`

- **Replaces:** `(rule)`, `(order)`, `(ordinance)`
- **Necessity:** Secondary legislation (Rules, Notifications, Orders) is semantically distinct from primary legislation: regulations implement statutes, are issued by gov-entities, and have separate gazette-notification dates. Core AMR has no equivalent concept.
- **AMR validity:** Bare noun, no collision.
- **Example:**
```lisp
(r / regulation
   :legal-name "Companies (Share Capital and Debentures) Rules, 2014"
   :legal-year 2014)
```

### 2.3 `provision`

- **Replaces:** `(section)`, `(article)`, `(clause)`, `(sub-section)`, `(schedule)`, `(chapter)`, `(part)`
- **Necessity:** Core AMR has no concept for the citable sub-units of a statute. The seven former sub-types are folded into one concept, with the kind preserved in the `:legal-level` attribute. Structural containment uses core AMR `:part-of`.
- **AMR validity:** Bare noun, no collision.
- **Example:**
```lisp
(p / provision
   :legal-level "section"
   :legal-name "Section 197"
   :part-of (c / statute :legal-name "Code of Criminal Procedure, 1973"))
```

### 2.4 `gov-entity`

- **Replaces:** `(ministry)`, `(department)`, `(court)`, `(tribunal)`, `(statutory-body)`, `(state-government)`, `(local-authority)`
- **Necessity:** Core AMR's `government-organization` is too narrow (typically a government agency in the AMR 3.0 sense) and does not naturally include courts, tribunals, or statutory bodies. `gov-entity` plus `:legal-type` collapses seven Indian-legal sub-types into one concept.
- **AMR validity:** Bare noun, no collision.
- **Example:**
```lisp
(g / gov-entity
   :legal-type "court"
   :legal-name "Supreme Court of India"
   :legal-jurisdiction "India")
```

### 2.5 `judgment`

- **Replaces:** `(decision)`, `(ruling)`
- **Necessity:** Core AMR has no concept for a judicial decision. A judgment is the document/event that contains holdings, cites provisions, and may overrule other judgments; this structural role requires a dedicated concept.
- **AMR validity:** Bare noun, no collision. **Do not attach `:ARGn` directly** — link via `:ARG1-of (r / render-01 :ARG0 (court))` to express "judgment rendered by court" (see §6.1).
- **Example:**
```lisp
(h / judgment
   :legal-citation "2020 SCC OnLine SC 123"
   :ARG1-of (r / render-01
              :ARG0 (c / gov-entity
                       :legal-type "court"
                       :legal-name "Supreme Court of India")))
```

### 2.6 `notification`

- **Replaces:** `(gazette)`, `(circular)`
- **Necessity:** Official gazette notifications (e.g. `S.O. 1234(E)`) are a distinct legal instrument type in Indian law, with separate gazette-publication dates and issuing gov-entities. Core AMR has no equivalent.
- **AMR validity:** Bare noun, no collision.
- **Example:**
```lisp
(n / notification
   :legal-name "S.O. 1234(E)"
   :legal-notified-on "2014-03-27"
   :legal-jurisdiction "India")
```

### 2.7 `legal-instrument`

- **Replaces:** `(document)`
- **Necessity:** Umbrella concept for citable legal texts that do not fit `statute` / `regulation` / `notification` / `judgment` (e.g., circulars, office memoranda, draft rules). Without this fallback, annotators would force-fit marginal documents into one of the typed concepts.
- **AMR validity:** Bare noun, no collision.
- **Example:**
```lisp
(d / legal-instrument
   :legal-name "CBDT Circular No. 17/2015"
   :legal-notified-on "2015-04-13")
```

---

## 3. Relation Tags (12)

Every role is `:legal-*` namespaced. Direction is read source → target.

### 3.1 Referential

#### 3.1.1 `:legal-cites`

- **Direction:** any → {`provision`, `statute`, `judgment`, `regulation`}
- **Necessity:** AMR's `:ARGM-PRP` (purpose) and `:ARGM-ADV` (adverbial) capture "under X" as an adjunct on a verb, not as a typed instrument-to-instrument edge. `:legal-cites` gives a structured citation-graph edge — essential for legal-IR queries (find all judgments citing Section 34). Folds in v1.0's `:legal-relies-on` via the optional `:legal-weight` attribute.
- **AMR validity:** Domain-specific relation, namespaced, no collision with core AMR or PropBank. Multi-valued (one node may cite many targets).
- **Example:**
```lisp
(j / judgment
   :legal-citation "AIR 1973 SC 1461"
   :legal-cites (a / provision
                   :legal-level "article"
                   :legal-name "Article 21"
                   :part-of (c / statute :legal-name "Constitution of India")))
```

#### 3.1.2 `:legal-overrules`

- **Direction:** `judgment` → `judgment`
- **Necessity:** Core AMR has `reverse-01` (a court reversing a lower court's decision) and `repeal-01` (a legislature nullifying a statute), but neither captures judgment-to-judgment overruling of precedent. `:legal-overrules` is the structured edge for the common-law overruling relation.
- **AMR validity:** Domain relation, namespaced, no collision.
- **Example:**
```lisp
(j2 / judgment
   :legal-citation "2017 SCC OnLine SC 1192"
   :legal-overrules (j1 / judgment
                       :legal-citation "AIR 1967 SC 1643"))
```

### 3.2 Normative

#### 3.2.1 `:legal-amends`

- **Direction:** {`statute`, `regulation`, `provision`} → {`statute`, `regulation`, `provision`}
- **Necessity:** PropBank's `amend-01` captures the *event* of amending. The Legal-AMR `:legal-amends` captures the *instrument-to-instrument* relation "Act A amends provision P of Act B" as a structured edge on the amending instrument, which is what amendment-tracking systems query.
- **AMR validity:** Domain relation, namespaced, no collision with `amend-01` (which is a verbal predicate, not an edge).
- **Example:**
```lisp
(a / statute
   :legal-name "Companies (Amendment) Act, 2017"
   :legal-year 2017
   :legal-amends (s / provision
                    :legal-level "section"
                    :legal-name "Section 2"
                    :part-of (c / statute
                                :legal-name "Companies Act, 2013"
                                :legal-year 2013)))
```

#### 3.2.2 `:legal-repeals`

- **Direction:** {`statute`, `regulation`} → {`statute`, `regulation`}
- **Necessity:** Same as `:legal-amends` for repeal. PropBank `repeal-01` is the event; `:legal-repeals` is the instrument-to-instrument edge.
- **AMR validity:** Domain relation, namespaced, no collision.
- **Example:**
```lisp
(b / statute
   :legal-name "Bharatiya Nagarik Suraksha Sanhita, 2023"
   :legal-year 2023
   :legal-valid-from "2024-07-01"
   :legal-repeals (c / statute
                    :legal-name "Code of Criminal Procedure, 1973"
                    :legal-year 1973
                    :legal-valid-until "2024-06-30"))
```

#### 3.2.3 `:legal-notifies`

- **Direction:** `gov-entity` → {`notification`, `regulation`}
- **Necessity:** PropBank `notify-01` is for general notification (of a person, of an event). The legal sense — a gov-entity formally issuing a gazette notification — requires an instrument-typed target, which PropBank does not support.
- **AMR validity:** Domain relation, namespaced, no collision with `notify-01`.
- **Example:**
```lisp
(m / gov-entity
   :legal-type "ministry"
   :legal-name "Ministry of Finance"
   :legal-notifies (n / notification
                      :legal-name "S.O. 1234(E)"
                      :legal-notified-on "2014-03-27"))
```

#### 3.2.4 `:legal-implements`

- **Direction:** `regulation` → `statute`
- **Necessity:** PropBank `implement-01` is for general implementation (of a plan, idea). The legal sense — a regulation operationalizing its parent statute — is a structured instrument-to-instrument edge that downstream legal-IR queries (find all rules under the Companies Act, 2013) need.
- **AMR validity:** Domain relation, namespaced, no collision with `implement-01`.
- **Example:**
```lisp
(r / regulation
   :legal-name "Companies (Share Capital and Debentures) Rules, 2014"
   :legal-year 2014
   :legal-implements (a / statute
                        :legal-name "Companies Act, 2013"
                        :legal-year 2013))
```

#### 3.2.5 `:legal-applies-to`

- **Direction:** `provision` → {concept, `gov-entity`}
- **Necessity:** AMR has no built-in way to express "this provision governs this class of entities or this subject-matter". Folds in v1.0's `:legal-governs`, which was semantically near-identical.
- **AMR validity:** Domain relation, namespaced, no collision.
- **Example:**
```lisp
(p / provision
   :legal-level "section"
   :legal-name "Section 138"
   :part-of (n / statute :legal-name "Negotiable Instruments Act, 1881")
   :legal-applies-to (c / cheque))
```

#### 3.2.6 `:legal-exempts`

- **Direction:** `provision` → {concept, `gov-entity`}
- **Necessity:** Distinct from `:legal-applies-to` + `:polarity -`. An exemption is an *affirmative* carve-out ("even though this provision would apply, it does not for these entities"), not a negation of scope. Core AMR has no equivalent.
- **AMR validity:** Domain relation, namespaced, no collision.
- **Example:**
```lisp
(p / provision
   :legal-level "section"
   :legal-name "Section 8"
   :part-of (c / statute :legal-name "Companies Act, 2013")
   :legal-exempts (o / organization))
```

### 3.3 Authority & Competence

#### 3.3.1 `:legal-empowers`

- **Direction:** `provision` → `gov-entity`
- **Necessity:** Core AMR has no edge for "provision grants power to a government entity". Folds in v1.0's `:legal-competent-authority` (which was the inverse direction).
- **AMR validity:** Domain relation, namespaced, no collision. Multi-valued (one provision may empower multiple gov-entities).
- **Example:**
```lisp
(p / provision
   :legal-level "section"
   :legal-name "Section 3"
   :part-of (s / statute :legal-name "SEBI Act, 1992")
   :legal-empowers (g / gov-entity
                      :legal-type "statutory-body"
                      :legal-name "Securities and Exchange Board of India"))
```

#### 3.3.2 `:legal-delegates-to`

- **Direction:** `gov-entity` → `gov-entity`
- **Necessity:** Distinct from `:legal-empowers` (which is provision → gov-entity). Delegation is a gov-to-gov authority transfer that core AMR does not model.
- **AMR validity:** Domain relation, namespaced, no collision.
- **Example:**
```lisp
(g1 / gov-entity
   :legal-type "state-government"
   :legal-name "Central Government"
   :legal-delegates-to (g2 / gov-entity
                          :legal-type "state-government"
                          :legal-name "State Government"))
```

#### 3.3.3 `:legal-administered-by`

- **Direction:** `statute` → `gov-entity`
- **Necessity:** Distinct from `:legal-empowers` (per-provision) and `:legal-delegates-to` (gov-to-gov). Statute-level administrative responsibility is a stable, queryable property of a statute (the Companies Act, 2013 is administered by the Ministry of Corporate Affairs). Core AMR has no equivalent.
- **AMR validity:** Domain relation, namespaced, no collision.
- **Example:**
```lisp
(a / statute
   :legal-name "Companies Act, 2013"
   :legal-year 2013
   :legal-administered-by (m / gov-entity
                             :legal-type "ministry"
                             :legal-name "Ministry of Corporate Affairs"))
```

### 3.4 Definitional

#### 3.4.1 `:legal-defines`

- **Direction:** `provision` → {concept, string}
- **Necessity:** PropBank's `define-01` captures the *event* of defining. The Legal-AMR `:legal-defines` captures the *provision-to-term* edge ("Section 2(1)(a) defines 'company'") as a structured relation on the provision, which is what definitional lookup systems query. Inverse (`:legal-defined-in`) is queryable by AMR's bidirectional edge semantics, so it is dropped.
- **AMR validity:** Domain relation, namespaced, no collision with `define-01`.
- **Example:**
```lisp
(s / provision
   :legal-level "clause"
   :legal-name "Section 2(1)(a)"
   :part-of (a / statute :legal-name "Companies Act, 2013")
   :legal-defines (t / term :legal-name "company"))
```

> **Note on structural containment.** v1.1 prunes `:legal-part-of` because it duplicated core AMR `:part-of`. Use `:part-of` directly for provision→statute, sub-section→section, schedule→statute containment. The child's `:legal-level` records which kind it is.

> **Note on anaphora.** v1.1 prunes `:legal-antecedent` because AMR convention for anaphora is variable reentrancy, not a dedicated relation. When "the said Act" refers to "Companies Act, 2013", use the same variable in both surface positions. See §6.6 for the worked pattern.

---

## 4. Attribute Tags (11)

All attributes are `:legal-*` namespaced and carry string or integer literals. They attach to typed nodes.

### 4.1 `:legal-name`

- **Applies to:** all 7 concept types
- **Type:** string
- **Necessity:** Legal names often have internal structure (parentheticals, commas, year suffixes) that the canonical AMR `:name` + `:opN` pattern cannot cleanly represent (e.g., "Companies (Share Capital and Debentures) Rules, 2014"). A domain-specific string attribute preserves the name as-is for gazetteer matching.
- **AMR validity:** String literal attribute on a concept node. Allowed in AMR.
- **Example:**
```lisp
(s / statute :legal-name "Companies Act, 2013")
```

### 4.2 `:legal-level`

- **Applies to:** `provision` only (validator: §7.3)
- **Type:** string (closed vocabulary: `"section"` / `"article"` / `"clause"` / `"sub-section"` / `"schedule"` / `"chapter"` / `"part"`)
- **Necessity:** Collapses 7 former sub-type concepts into 1. Without it, the inventory would need 7 provision concepts.
- **AMR validity:** String literal attribute, scoped to `provision`.
- **Example:**
```lisp
(p / provision
   :legal-level "article"
   :legal-name "Article 21"
   :part-of (c / statute :legal-name "Constitution of India"))
```

### 4.3 `:legal-type`

- **Applies to:** `gov-entity` only (validator: §7.3)
- **Type:** string (closed vocabulary: `"ministry"` / `"department"` / `"court"` / `"tribunal"` / `"statutory-body"` / `"state-government"` / `"local-authority"`)
- **Necessity:** Collapses 7 former sub-type concepts into 1.
- **AMR validity:** String literal attribute, scoped to `gov-entity`.
- **Example:**
```lisp
(g / gov-entity
   :legal-type "tribunal"
   :legal-name "National Green Tribunal")
```

### 4.4 `:legal-year`

- **Applies to:** `statute`, `regulation`, `judgment`
- **Type:** integer
- **Necessity:** Year-as-property-of-instrument (e.g., "Companies Act, 2013" → 2013) is metadata about the instrument, not a temporal modifier of an event. Core AMR's `:date-entity` + `:year` is for event times, not for instrument metadata.
- **AMR validity:** Integer literal attribute.
- **Example:**
```lisp
(s / statute
   :legal-name "Indian Penal Code, 1860"
   :legal-year 1860)
```

### 4.5 `:legal-jurisdiction`

- **Applies to:** `statute`, `gov-entity` (court), `regulation`
- **Type:** string
- **Necessity:** Jurisdiction is a stable property of a legal instrument or court (India, State of Kerala, etc.) used for jurisdictional filtering. Core AMR's `:location` is for spatial location of events, not for the jurisdiction of an instrument.
- **AMR validity:** String literal attribute.
- **Example:**
```lisp
(c / gov-entity
   :legal-type "court"
   :legal-name "Kerala High Court"
   :legal-jurisdiction "State of Kerala")
```

### 4.6 `:legal-citation`

- **Applies to:** `judgment`
- **Type:** string
- **Necessity:** Indian judgments have structured citation forms (e.g., "AIR 1973 SC 1461", "2020 SCC OnLine SC 123") that are distinct from a `:name` and are the primary lookup key in legal databases. Core AMR has no equivalent.
- **AMR validity:** String literal attribute.
- **Example:**
```lisp
(j / judgment :legal-citation "AIR 1973 SC 1461")
```

### 4.7 `:legal-status`

- **Applies to:** `statute`, `provision`, `regulation`
- **Type:** string (closed vocabulary: `"in-force"` / `"repealed"` / `"struck-down"` / `"amended"` / `"not-yet-in-force"`)
- **Necessity:** Legal status is a stable property of an instrument at a point in time, queried for "is this statute currently in force?". Core AMR has no equivalent.
- **AMR validity:** String literal attribute.
- **Example:**
```lisp
(s / statute
   :legal-name "Indian Penal Code, 1860"
   :legal-status "repealed"
   :legal-valid-until "2023-06-30")
```

### 4.8 `:legal-valid-from`

- **Applies to:** any
- **Type:** ISO date string (`"YYYY-MM-DD"`)
- **Necessity:** Effective-from date of an instrument. Distinct from `:legal-enacted-on` (enactment ≠ entry into force in Indian law). Core AMR's `:date-entity` is for event times, not for instrument metadata.
- **AMR validity:** String literal attribute.
- **Example:**
```lisp
(s / statute
   :legal-name "Bharatiya Nagarik Suraksha Sanhita, 2023"
   :legal-valid-from "2024-07-01")
```

### 4.9 `:legal-valid-until`

- **Applies to:** any
- **Type:** ISO date string
- **Necessity:** Effective-until date. Companion to `:legal-valid-from`; needed for repealed instruments.
- **AMR validity:** String literal attribute.
- **Example:**
```lisp
(c / statute
   :legal-name "Code of Criminal Procedure, 1973"
   :legal-valid-until "2024-06-30")
```

### 4.10 `:legal-enacted-on`

- **Applies to:** `statute`
- **Type:** ISO date string
- **Necessity:** Enactment date. In Indian law, often distinct from `:legal-valid-from` (a statute may be enacted on one date and come into force on another). Core AMR has no equivalent.
- **AMR validity:** String literal attribute.
- **Example:**
```lisp
(s / statute
   :legal-name "Indian Penal Code, 1860"
   :legal-enacted-on "1860-10-06")
```

### 4.11 `:legal-notified-on`

- **Applies to:** `notification`
- **Type:** ISO date string
- **Necessity:** Gazette-notification date. Distinct from enactment (notifications are not enacted, they are notified). Required for the `notification` concept to be meaningful.
- **AMR validity:** String literal attribute.
- **Example:**
```lisp
(n / notification
   :legal-name "S.O. 1234(E)"
   :legal-notified-on "2014-03-27")
```

### Optional edge attribute: `:legal-weight`

- **Applies to:** source node of a `:legal-cites` edge
- **Type:** string (closed vocabulary: `"binding"` / `"persuasive"` / `"obiter"`)
- **Necessity:** A single judgment can have binding holdings, persuasive reasoning, and obiter dicta — all cited by downstream judgments. The `judgment` node type alone cannot distinguish these. `:legal-weight` sharpens the citation edge.
- **AMR validity:** String literal attribute.
- **Example:**
```lisp
(j / judgment
   :legal-citation "2020 SCC OnLine SC 123"
   :legal-weight "binding"
   :legal-cites (p / provision
                   :legal-level "article"
                   :legal-name "Article 14"
                   :part-of (c / statute :legal-name "Constitution of India")))
```

> **Note on `:legal-assented-on`.** v1.0 listed this as optional. v1.1 prunes it: in Indian law, the date of Presidential assent and the date of enactment are typically identical, and the Gazette records them as a single date. Re-add only if a pilot corpus shows divergence.

---

## 5. Master Tagset Summary

A consolidated reference. Every extension token below carries the `:legal-` prefix.

```text
# Concepts (7)  — typed nouns, no PropBank suffix
statute
regulation
provision
gov-entity
judgment
notification
legal-instrument

# Relations (12)
:legal-cites
:legal-overrules
:legal-amends
:legal-repeals
:legal-notifies
:legal-implements
:legal-applies-to
:legal-exempts
:legal-empowers
:legal-delegates-to
:legal-administered-by
:legal-defines

# Attributes (11)
:legal-name
:legal-level
:legal-type
:legal-year
:legal-jurisdiction
:legal-citation
:legal-status
:legal-valid-from
:legal-valid-until
:legal-enacted-on
:legal-notified-on

# Optional edge attribute (1)
:legal-weight
```

Counts: **7 concepts / 12 relations / 11 attributes** (+1 optional). Down from v1.0's 7 / 16 / 11.

---

## 6. Worked AMR Examples

Each example is a single-rooted Penman tree. Core AMR roles (`:ARG0`, `:ARG1`, `:ARGn-of`, `:name`, `:part-of`, `:location`, `:time`) are kept where the source sentence has a predicate or spatial/temporal relation; Legal-AMR roles are used only for legal-semantic edges. All code blocks are balanced.

### 6.1 A Judgment Citing a Provision (v1.0 §6.1, ARG0-on-noun fixed)

**Sentence:** *"The Supreme Court held that Section 197 CrPC protects public servants."*

v1.0 attached `:ARG0` directly to the `judgment` noun, violating AMR's rule that `:ARGn` appears only on PropBank predicates. v1.1 expresses "the court rendered the judgment" through `:ARG1-of (r / render-01)`.

```lisp
(h / judgment
   :legal-citation "2020 SCC OnLine SC 123"
   :ARG1-of (r / render-01
              :ARG0 (c / gov-entity
                       :legal-type "court"
                       :legal-name "Supreme Court of India"))
   :legal-cites (s / provision
                   :legal-level "section"
                   :legal-name "Section 197"
                   :part-of (c2 / statute :legal-name "Code of Criminal Procedure, 1973")
                   :legal-applies-to (p / public-servant)))
```

### 6.2 An Amendment

**Sentence:** *"The Companies (Amendment) Act, 2017 amended Section 2 of the Companies Act, 2013."*

```lisp
(a / statute
   :legal-name "Companies (Amendment) Act, 2017"
   :legal-year 2017
   :legal-amends (s / provision
                    :legal-level "section"
                    :legal-name "Section 2"
                    :part-of (c / statute
                                :legal-name "Companies Act, 2013"
                                :legal-year 2013)))
```

### 6.3 A Repeal with Temporal Validity

**Sentence:** *"The BNSS, 2023 repealed the CrPC, 1973 with effect from 1 July 2024."*

```lisp
(b / statute
   :legal-name "Bharatiya Nagarik Suraksha Sanhita, 2023"
   :legal-year 2023
   :legal-valid-from "2024-07-01"
   :legal-repeals (c / statute
                    :legal-name "Code of Criminal Procedure, 1973"
                    :legal-year 1973
                    :legal-valid-until "2024-06-30"))
```

### 6.4 Competent Authority (Multi-valued Edge)

**Sentence:** *"Under Section 197 CrPC, the Central Government or State Government may sanction prosecution."*

```lisp
(s / provision
   :legal-level "section"
   :legal-name "Section 197"
   :part-of (c / statute :legal-name "Code of Criminal Procedure, 1973")
   :legal-empowers (g / gov-entity
                      :legal-type "state-government"
                      :legal-name "Central Government")
   :legal-empowers (g2 / gov-entity
                      :legal-type "state-government"
                      :legal-name "State Government"))
```

### 6.5 A Regulation Implementing a Statute, Administered by a Ministry

**Sentence:** *"The Companies (Share Capital and Debentures) Rules, 2014 implement the Companies Act, 2013, which is administered by the Ministry of Corporate Affairs."*

```lisp
(r / regulation
   :legal-name "Companies (Share Capital and Debentures) Rules, 2014"
   :legal-year 2014
   :legal-implements (a / statute
                        :legal-name "Companies Act, 2013"
                        :legal-year 2013
                        :legal-administered-by (m / gov-entity
                                                  :legal-type "ministry"
                                                  :legal-name "Ministry of Corporate Affairs")))
```

### 6.6 Definitional Anaphora via Core AMR Reentrancy

**Sentence:** *"Section 2(1)(a) of the Companies Act, 2013 defines 'company'; the said Act came into force on 1 April 2014."*

v1.0 modeled "the said Act" as a separate `legal-instrument` node linked by `:legal-antecedent`. v1.1 prunes that relation: AMR convention for anaphora is variable reentrancy. The single root is the `statute`; the definitional provision is its `part-of` child; the term "company" is defined by the provision; the effective date attaches directly to the statute. No anaphora relation is needed because both clauses refer to the same `a` variable.

```lisp
(a / statute
   :legal-name "Companies Act, 2013"
   :legal-year 2013
   :legal-valid-from "2014-04-01"
   :legal-defines (s / provision
                    :legal-level "clause"
                    :legal-name "Section 2(1)(a)"
                    :part-of a
                    :legal-defines (t / term :legal-name "company")))
```

Note: `:legal-defines` is multi-valued on `a` — once for the defining provision (Section 2(1)(a)), once for the defined term ("company"). AMR's multi-valued-edge semantics permits this.

### 6.7 A Challenge Under a Provision

**Sentence:** *"An arbitration award was challenged under Section 34 of the Arbitration and Conciliation Act, 1996 before the Delhi High Court in July 2013."*

`challenge-01`'s ARG2 is the opponent in PropBank (wrong for "under Section 34"). The spec-sanctioned replacement is `:legal-cites`, which v1.0 §3.1 already covered for the surface form "under [provision]".

```lisp
(c / challenge-01
   :ARG1 (a / award)
   :legal-cites (p / provision
                   :legal-level "section"
                   :legal-name "Section 34"
                   :part-of (s / statute
                               :legal-name "Arbitration and Conciliation Act, 1996"
                               :legal-year 1996
                               :legal-jurisdiction "India"))
   :location (h / gov-entity
                :legal-type "court"
                :legal-name "Delhi High Court"
                :legal-jurisdiction "India")
   :time (d / date-entity
            :year 2013
            :month 7))
```

---

## 7. Implementation Notes

1. **Layering.** Keep core AMR tags untouched. Apply Legal-AMR tags only where core AMR is insufficient. Every Legal-AMR graph must remain a valid core AMR graph *modulo* the legal-semantic edges — verify by stripping all `:legal-*` edges and checking that every remaining edge is from the core AMR inventory (`:ARG0`–`:ARG5`, `:ARGn-of`, `:mod`, `:poss`, `:part-of`, `:name`, `:location`, `:time`, `:opN`, `:polarity`, `:ARGM-*`) and that each connected component is single-rooted and balanced.
2. **Namespacing.** The `:legal-` prefix is mandatory in production. Annotation tools should reject unprefixed extension tags.
3. **Validation.** Maintain a SHACL-style or JSON-schema validator keyed on the 7 concepts, 12 relations, and 11 attributes above. Three checks are essential:
   - `:legal-level` appears only on `provision` nodes.
   - `:legal-type` appears only on `gov-entity` nodes.
   - `:ARGn` (n = 0..5) appears only on PropBank-predicate concepts (those ending in `-NN`, e.g. `challenge-01`) or via the `:ARGn-of` reverse form — never on bare noun concepts (`judgment`, `statute`, etc.).
4. **Use core `:part-of`, not `:legal-part-of`.** Legal containment (provision→statute, sub-section→section, schedule→statute) uses core AMR `:part-of`. The child's `:legal-level` records the kind.
5. **Use core AMR reentrancy, not `:legal-antecedent`.** Anaphoric references ("the said Act") use the same variable as the referent. No dedicated relation.
6. **Use `:legal-cites` + `:legal-weight`, not `:legal-relies-on`.** "Relies on" is a binding citation; the `:legal-weight` attribute captures strength.
7. **Use `:legal-applies-to`, not `:legal-governs`.** Both encoded provision scope; only one relation is kept.
8. **Gazetteers.** Populate `statute`, `provision`, and `gov-entity` nodes from curated Indian gazetteers (Central Acts, State Acts, Ministries, Tribunals, Statutory Bodies). Gazetteer entries supply `:legal-name`, `:legal-year`, `:legal-jurisdiction`, and `:legal-status`, leaving annotators free to focus on edges.
9. **Interoperability.** Map extension tags to existing legal ontologies — **NyOn** (Indian Supreme Court judgments, judicial module), **ILDGS** (document structure: statutes, regulations, provisions), and **Portuguese Legal AMR** (`LREF`-style references map to `:legal-cites` + `:legal-weight`). These mappings are external to the tagset and do not affect AMR validity.
10. **Pilot first.** Annotate a pilot corpus (for example, 500 Supreme Court judgments plus the Companies Act, 2013) before full deployment, to surface edge cases the lean tagset may miss. If a pilot reveals a recurring phenomenon the 12 relations cannot express, prefer adding a new `:legal-*` role over reintroducing the dropped concepts, so the concept inventory stays at 7. Any new role must pass the necessity test (§1): *"Can core AMR express this already?"*

---

## 8. Inspiration & References

- **NyOn Ontology** — module structure for Indian Supreme Court judgments (judicial system, courts, benches).
- **ILDGS** — document-structure ontology (statutes, regulations, provisions).
- **Portuguese Legal AMR** — `LREF` absolute legal references; `NE ADM` administrative bodies.
- **Legal NER schemes** — `STATUTE`, `PROVISION` entity types, mirrored here as `statute` and `provision` concepts.
- **Core AMR / PropBank** — base layer for general semantics; the original `amend-01`, `repeal-01`, `notify-01`, `define-01`, `implement-01` framesets remain usable for the verbal predicates, with `:legal-*` relations providing the nominal instrument-to-instrument edges PropBank does not cover.

---

## Appendix A — Full Changelog from v1.0

| Cut / changed | v1.0 | v1.1 | Reason |
|---|---|---|---|
| Provision sub-type concepts (section, article, clause, sub-section, schedule, chapter, part) | 7 concepts | 1 (`provision` + `:legal-level`) | Information preserved in an attribute; no query lost |
| Gov-entity sub-type concepts (ministry, department, court, tribunal, statutory-body, state-government, local-authority) | 7 concepts | 1 (`gov-entity` + `:legal-type`) | Same |
| `:legal-relies-on` | relation | pruned | Folded into `:legal-cites` + `:legal-weight` (v1.0 §3.1 already supported) |
| `:legal-governs` | relation | pruned | Folded into `:legal-applies-to` (both = provision scope) |
| `:legal-part-of` | relation | pruned | Duplicated core AMR `:part-of`; failed necessity test |
| `:legal-antecedent` | relation | pruned | Duplicated core AMR reentrancy; AMR convention is variable sharing |
| `:legal-assented-on` (optional) | optional attr | pruned | Identical to `:legal-enacted-on` in Indian law |
| §6.1 `:ARG0` on `judgment` noun | validity violation | fixed | AMR reserves `:ARGn` for PropBank predicates; replaced with `:ARG1-of (r / render-01 :ARG0 (court))` |
| Relation-count contradiction | "16" / "14" mixed | "12" everywhere | Single source of truth |
| §7.1 degradation guarantee | "single-rooted after strip" (false when :legal-* edge is sole connector) | "remaining edges are core AMR; each connected component single-rooted" | Corrected guarantee |
| Per-tag necessity + example | absent | added | Required for spec validity (v1.1 principle 4) |

**Totals**: v1.0 had 7 / 16 / 11 (+1 optional); v1.1 has **7 / 12 / 11** (+1 optional). Four relations and one optional attribute removed; one validity violation fixed; one false guarantee corrected.

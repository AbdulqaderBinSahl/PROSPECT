# Human Judge Evaluation Rubric
## PROSPECT Pipeline — Step 2 Corpus Evaluation

**Evaluators:** two domain experts (raters H1 and H2)  
**Papers to evaluate:** ~101 papers (score sheet provided separately)  
**Estimated time:** 3–5 minutes per paper (depends)

---

## Application Context

You are evaluating papers for inclusion in a curated knowledge corpus that will power an **LLM-based advisory system for polypropylene compound formulation**.

The system must answer questions such as:
- How does blending ratio (virgin PP : PCR-PP) affect impact strength and melt flow rate?
- What additives (impact modifiers, compatibilisers, stabilisers, fillers) are needed to reach target mechanical properties?
- What processing parameters are required for impact-toughened PP formulations?

**Target blend:** Virgin impact copolymer PP + virgin homopolymer PP + post-consumer recycled PP (PCR-PP) for automotive applications (target properties: impact strength, melt flow rate, tensile strength).

**Most useful papers** provide quantitative experimental data — tables or plots of composition versus measured mechanical properties — rather than purely theoretical discussions.

---

## Scoring Rubric

Assign each paper a score from **1 to 4**:

---

### Score 4 — Highly Valuable
**Include with high confidence.**

The paper directly addresses PP/recyclate blending, PCR-PP formulation, or additive effects in PP compounds, AND provides quantitative experimental data (e.g., composition–property tables, mechanical test results, processing parameter data).

**Typical indicators:**
- Reports MFI, impact strength, or tensile strength vs. blend ratio or additive loading
- Uses actual PCR-PP or post-consumer recyclate
- Provides formulation recommendations or optimisation data
- Studies impact modifiers (rubber, elastomers), compatibilisers, or stabilisers in PP

**Example from calibration:** — virgin PP blended with PCR-PP at varying ratios + stabilizer, Table 2 reports MFI/tensile/modulus, plus response-surface equations for property prediction.

---

### Score 3 — Useful
**Probably include.**

The paper is about PP or a closely related polymer system and provides transferable quantitative data, even if it does not directly use PCR-PP or the exact target blend.

**Typical indicators:**
- PP toughening, blending, or compounding data using virgin PP only
- Impact modifier or filler effects in PP (rubber, CaCO3, talc, glass fibre)
- Closely related polyolefin systems (PE/PP blends, polyolefin alloys)
- Processing studies (injection moulding, extrusion) for PP compounds

**Example from calibration:** — PP + SBS rubber + nano-CaCO3, quantitative impact/tensile data. No PCR-PP, but the additive synergy data is transferable to formulation decisions.

---

### Score 2 — Marginal
**Probably exclude.**

The paper has some connection to PP or polymer compounding but lacks the quantitative formulation data the system needs, or is too far removed from the target application.

**Typical indicators:**
- PP studied but only for a very different application (packaging, fibres, medical devices)
- Only theoretical or modelling work, no experimental property data
- Focus on degradation or recycling process chemistry rather than formulation properties
- Other polymers (PET, nylon, PLA) with no PP connection

**Example from calibration:**  — PP/EPDM toughening with reactive compatibiliser. Has quantitative impact data, but no PCR-PP, no MFR, and no recycled content. Borderline — scored 2 by one judge, 3 by two others.

---

### Score 1 — Not Relevant
**Definitely exclude.**

The paper has no useful content for the advisory system. Wrong polymer, wrong application domain, or no applicable data.

**Typical indicators:**
- Completely different polymer (PVC, polystyrene, epoxy, rubber)
- Polymer is a minor component in a non-polymer application
- Environmental/combustion/incineration studies with no formulation relevance
- Review paper with no original data (the system needs primary data)

**Example from calibration:** Paper108 — dioxin formation during PVC/PE combustion at 1073 K. No PP, no mechanical properties, no formulation relevance.

---

## Confidence Rating

After each score, record your confidence:

| Confidence | Meaning |
|---|---|
| **High** | You are certain the score is correct; you read enough to be sure |
| **Medium** | You are fairly confident but the paper is borderline or you skimmed some sections |
| **Low** | You are uncertain; limited reading time or genuinely ambiguous paper |

---

## What to Read (in order, stop when confident)

1. **Title and abstract** — usually enough to assign a 1 or a 4
2. **Introduction** (last paragraph) — confirms the research question
3. **Results/Tables** — check for composition–property data tables
4. **Conclusion** — confirms findings
5. **Methods** — only if you need to verify what polymer system was used

You do **not** need to read the full paper for every entry. If it is clearly irrelevant after the abstract, assign a 1 and move on.

---

## Scoring Sheet

Record your scores in the provided CSV file (`score_sheet.csv`):

| Column | What to enter |
|---|---|
| `paper_id` | Pre-filled — do not change |
| `score` | Integer: 1, 2, 3, or 4 |
| `confidence` | Text: low, medium, or high |
| `rationale` | 1–2 sentences citing a specific finding or reason for your score |

**Rationale requirement:** Do not write generic phrases like "this is a relevant paper." Reference at least one specific thing from the paper — a finding, a dataset, a material, or a gap.

**Example rationale (score 3):** *"Reports tensile strength and impact data for PP/talc composites at 5–20 wt% loading, but uses virgin PP only with no recycled content."*

---

## Important Notes

- **Work independently.** Do not discuss scores with the other evaluator until both have submitted.
- **Blinded evaluation.** The score sheet contains paper IDs only — no filtering condition labels, no LLM scores.
- **Save frequently.** The CSV can be filled in incrementally; you do not need to complete it in one session.
- **Flag uncertainty.** If a paper is genuinely ambiguous, use confidence = low and note your uncertainty in the rationale. These will be reviewed jointly.
---

## Submission

Return your completed `score_sheet.csv` to the study coordinator, who saves it as
`filter_validity/human_scores/<your rater ID>.csv` (e.g. `H1.csv`).

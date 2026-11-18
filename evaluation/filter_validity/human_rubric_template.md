# Rater instructions: paper usefulness

**Your rater ID:** {rater_id}
**Papers to rate:** {n_papers} (listed in `score_sheet.csv`; the PDFs are in this folder)
**Time:** about 3–5 minutes per paper

---

## Application

You are rating papers for a curated knowledge corpus behind this application:

> {application_description}

Replace this paragraph with two or three concrete questions the application must answer, and what a "most useful" paper looks like (for example: papers with quantitative experimental data, not purely theoretical discussion).

---

## Score each paper from 1 to 4

Use the same scale the LLM raters use.

### 4 — Highly valuable (include with high confidence)
The paper directly addresses the application and provides the kind of data it needs.
*Typical indicators:* replace with 3–4 indicators for your application.

### 3 — Useful (probably include)
A closely related system or method with transferable data, even if not the exact target.
*Typical indicators:* replace with 3–4 indicators.

### 2 — Marginal (probably exclude)
Some connection to the topic, but lacks the data the application needs or is too far from it.
*Typical indicators:* replace with 3–4 indicators.

### 1 — Not relevant (exclude)
Wrong material or field, wrong application, or no applicable data. A review with no original data also scores 1 if the application needs primary data.

A worked example for each score, taken from your calibration round, helps raters agree.

---

## Confidence

| Confidence | Meaning |
|---|---|
| **high** | You are sure; you read enough to decide |
| **medium** | Fairly sure, but the paper is borderline or you skimmed parts |
| **low** | Unsure: limited time or a genuinely ambiguous paper |

## What to read (stop when you are confident)

1. Title and abstract — often enough for a 1 or a 4
2. Last paragraph of the introduction — the research question
3. Results, tables and figures — is the needed data there?
4. Conclusion
5. Methods — only to check what was studied

## Filling in the score sheet

| Column | What to enter |
|---|---|
| `paper_id` | Pre-filled — do not change |
| `score` | 1, 2, 3 or 4 |
| `confidence` | low, medium or high |
| `rationale` | 1–2 sentences citing something specific from the paper (a finding, a dataset, a material, a gap) — not "this is a relevant paper" |

## Rules

- **Work independently.** Do not discuss papers with other raters until everyone has submitted.
- **The packet is blinded.** It has no filter decisions, condition names, LLM scores or strata. Please do not look them up.
- **Save often.** You can fill the sheet over several sessions.

When finished, return `score_sheet.csv` to the study coordinator, who saves it as `human_scores/{rater_id}.csv`.

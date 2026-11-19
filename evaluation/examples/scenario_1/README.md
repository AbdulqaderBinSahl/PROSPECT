# Scenario 1 — the paper's evaluation inputs

Application: an LLM advisory system for formulating polypropylene compounds that blend virgin and post-consumer recycled PP for automotive parts (212 candidate papers after Step 1).

| File | What |
|---|---|
| `settings.yaml` | The paper's evaluation settings: models, prompt values, sample sizes, seeds, contrasts |
| `corpus_papers.csv` | The 212 papers of the corpus: paper ID (the PDF filename), DOI, year, journal and title, from the Scopus metadata. Save each PDF as `<paper_id>.pdf` to rebuild the corpus |
| `application_description.txt` | The application text every rater saw (Step 2 panel) |
| `sample_locked.json` | The IDs of the 101 papers rated by the panel (seed 42, locked before rating); which conditions kept each paper is not included |
| `human_rubric.md` | Instructions given to the two human raters (anonymised as H1 and H2) |
| `queries/layer_1.json` | 24 queries: the operational config's representative queries |
| `queries/layer_2.json` | 24 paraphrases, one per layer 1 query (robustness only) |
| `queries/layer_3.json` | 10 held-out queries written by a domain expert |
| `references/Q*.json` | 34 reference answers (layers 1 and 3), drafted from the top 20 chunks of the unfiltered corpus |

Not included: the PDFs and anything derived from their full text (extracted texts, indexes, retrieved chunks), the filter decisions and outputs, the RAG answers and all scores.

To rerun with these inputs, place a PROSPECT run folder with the five conditions at `runs/scenario_1` (or edit `run_dir`) and follow [the evaluation README](../../README.md), passing `--settings evaluation/examples/scenario_1/settings.yaml`. Steps that would recreate a fixed input (drawing the sample, building or paraphrasing queries, drafting references) can be skipped: the settings point to the files here.

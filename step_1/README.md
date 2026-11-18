# Step 1 — Corpus discovery

```bash
python step_1/run_step1.py --run-dir runs/my_topic [--skip-triage | --skip-scopus | --triage-only]
```

Step 1 turns a plain-language description of your RAG application into a set of candidate papers. Steps 1–3 are interactive: you answer questions and approve what the LLM proposes.

| Phase | What happens | Module |
|---|---|---|
| 1. Intent elicitation | The LLM asks up to three rounds of clarifying questions, then drafts an **intent brief**: domain, application context, knowledge axes (with priorities), scope boundaries, content anti-patterns, depth expectation and time window. You confirm or correct it. | `intent_elicitor.py` |
| 2. Search strategy | The LLM groups search terms into 3–5 concept groups and combines them into Scopus Boolean queries. You can approve, edit (by instruction or by hand), remove or add queries. | `strategy_generator.py` |
| 3. Query validation | Each query's Scopus hit count is checked against safety caps derived from your target corpus size. The LLM judges borderline counts and refines queries that are too broad or too narrow (up to three rounds). You approve the final set. | `query_validator.py` |
| 4. Retrieval | Metadata for every hit (up to 2,000 per query) is fetched and de-duplicated across queries by DOI, or by Scopus EID when there is no DOI. Missing abstracts are fetched from the Abstract Retrieval API. | `scopus_retriever.py` |
| 5. Abstract triage | Each title and abstract is classified PASS or SKIP. The prompt defaults to PASS: a paper is skipped only when the LLM is highly confident it is outside the domain, irrelevant to every knowledge axis, and within a scope boundary. | `triage.py` |

Outputs (in `<run-dir>/step1/`):

| File | Contents |
|---|---|
| `intent_brief.json` | The confirmed brief. Step 2 builds its criteria from it. |
| `corpus.json` | Brief, search strategy, validation results and the metadata of every retrieved paper |
| `corpus_triaged.json` | Same structure, PASS papers only |
| `triage_report.json` | Verdict and reason for every paper |
| `pdf_checklist.csv` | One row per paper: expected PDF filename, DOI link, title, year, journal |
| `run_log.md` | The conversation, queries, hit counts and settings, for reporting |

## Modes

- `--skip-triage` keeps every retrieved paper.
- `--skip-scopus` stops after phase 1 and writes only `intent_brief.json` and `run_log.md`. No Scopus key is needed. Put your own PDFs in `<run-dir>/papers/` and run Step 2.
- `--triage-only` runs phase 5 on an existing `corpus.json` (for example after `--skip-triage`).

Step 1 never overwrites an existing `step1/` folder: start a new `--run-dir` for a new search.

## Settings

LLM temperature is 0.3 for phases 1–3 and 0.1 for triage. Other limits (clarification rounds, validation rounds, results per query, request delay) are in `rag_corpus_builder/config.py`.

Tools:
- `python step_1/tools/check_scopus_quota.py` reports your remaining Scopus quota.
- `python step_1/tools/backfill_abstracts.py <corpus.json>` fetches abstracts still missing from a corpus.

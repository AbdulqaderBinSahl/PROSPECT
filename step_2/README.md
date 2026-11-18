# Step 2 — Application-conditioned filtering

```bash
python step_2/run_step2.py --run-dir runs/my_topic [--condition A3w|A3|A4|A2|A1] [--resume] [--prepare-only]
```

## Inputs

- **Intent brief.** Taken from Step 1 output in `<run-dir>/step1/` (`corpus_triaged.json`, else `corpus.json`, else `intent_brief.json`), or from a file passed with `--input` (see `templates/intent_brief.example.json`). It needs at least `domain` and `application_context`.
- **PDFs.** Taken from `<run-dir>/papers/`, or from the folder passed with `--papers-dir`. They need a text layer: PROSPECT does not run OCR. Before filtering, Step 2 lists any PDF averaging under 50 extractable words per page (usually a scanned image) and repeats the list at the end. Run OCR on those PDFs first, for example with OCRmyPDF, or remove them. A PDF with no text at all is discarded by every condition without an LLM call.

The first call records both in `<run-dir>/run_inputs.json`, together with a hash of the input file. Later calls reuse them and refuse to run if they change, so every condition in a run folder filters the same PDFs from the same brief.

## Configuration pre-step (once per run folder)

One LLM call turns the intent brief into `operational_config.json`:
- the application description;
- a domain taxonomy;
- value criteria, each with a 0–5 rubric;
- representative queries the corpus must serve;
- anti-patterns;
- the strong-chunk thresholds used by the evidence gate.

Run `--prepare-only` to generate it and stop, so you can review or edit it. Every condition reads this one file.

## Conditions

Every condition writes one decision per paper and copies each PDF to `routed_pdfs/keep` or `routed_pdfs/discard`. The PDFs in `papers/` are never moved.

**A3w (default), A3 and A4** — `rag_corpus_filter/pipeline.py`

1. Text is extracted per page with PyMuPDF and split into word chunks (1,400 words, 200 overlap, tails under 300 merged into the previous chunk).
2. **Agent 1 — Paper Profiler** characterises the paper from its domain, Scopus metadata, abstract and last pages. It reports paper type, methodology, scope, data types and initial domain fit.
3. *A3w only:* Agent 1 also assigns each chunk an application-conditioned weight between 0 and 2, based on the chunk's section type and the application.
4. **Agent 2 — Relevance Assessor** scores every chunk for domain relevance and on each value criterion (0–5), flags anti-patterns and quotes evidence.
5. **Agent 3 — Utility Simulator** scores every chunk against each representative query. It runs in parallel with Agent 2 and is skipped in A4.
6. A **deterministic layer** computes:
   - the composite score (weighted by the A3w chunk weights);
   - per-criterion averages;
   - the number of strong chunks;
   - query coverage;
   - the relevance–utility gap.
7. **Agent 4 — Aggregator Judge** reads the profile, the per-chunk summaries and those metrics, and decides keep or discard with a written argument for and against. The Judge never sees the chunk weights, only the weighted metrics.
8. An **evidence gate** flags (but does not override) any keep that has fewer strong chunks than the config requires.

**A2** — `baselines/a2_all_roles.py`

The same chunks are each assessed in one prompt that plays the profiler, relevance and utility roles together. One aggregation call then decides. Each chunk prompt carries exactly the paper context Agent 1 receives in A3w, A3 and A4.

**A1** — `baselines/a1_whole_paper.py`

The full paper text and the full operational config go into a single call.

## Outputs (`<run-dir>/<condition>/`)

| File | Contents |
|---|---|
| `paper_decisions/*.json` | Decision, score (1–10) and reasoning per paper. A3w, A3 and A4 also store strengths and weaknesses, the best case for each side, agent disagreements, top evidence quotes, the Agent 1 profile, the deterministic metrics, and the chunk weights (A3w). |
| `routed_pdfs/keep/`, `routed_pdfs/discard/` | Copies of the PDFs |
| `chunks_*.jsonl` | Per-chunk Agent 2 / Agent 3 assessments (A3w, A3, A4) |
| `chunk_assessments/` | Per-chunk assessments (A2) |
| `filter_results_*.xlsx` | Paper, chunk, query-coverage and metrics sheets (A3w, A3, A4) |
| `operational_config.json` | Copy of the run's shared config |
| `run_manifest.json` | Condition, inputs, model, settings, command line |
| `run_metrics.json` | Same format for every condition: papers processed, failed (by name), kept and discarded; total and average LLM calls and time; model and platform; one row per paper. Updated after every paper. |
| `bad_outputs/` | Raw replies that failed JSON parsing or validation, kept for debugging |

## Interrupted runs

Decisions and metrics are saved after every paper. Re-run the same command with `--resume` to skip papers already decided. Without `--resume`, Step 2 refuses to write into a condition folder that already has decisions.

A paper whose LLM calls keep failing (for example, a small model that repeatedly returns incomplete JSON) gets no decision. The run continues with the next paper and, at the end, lists every paper without a decision. Run the same command with `--resume` to retry only those.

## Prompts

The agent prompts are in `rag_corpus_filter/prompts/`. The A1 and A2 prompts are inside their modules in `baselines/`.

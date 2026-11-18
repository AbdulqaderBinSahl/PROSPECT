# PROSPECT

Application-conditioned curation of retrieval-augmented generation (RAG) corpora from scientific literature.

You describe what your RAG system is for. PROSPECT turns that description into a literature search, triages the results by abstract, and then reads each paper's full text with a team of LLM agents to decide whether it belongs in your corpus. The result is a folder of kept PDFs, with a written justification for every keep and discard decision.

This repository contains the method, and the code the paper used to evaluate it ([evaluation/](evaluation/README.md)). It contains no data, corpora or results.

```
Step 1 — Corpus discovery                      You                    Step 2 — Filtering
───────────────────────────                    ───                    ──────────────────
LLM conversation → intent brief                                       configuration pre-step (once per run)
intent brief → Scopus Boolean queries          download the PDFs        criteria + rubrics, queries, anti-patterns
Scopus retrieval → paper metadata      ──►     listed in        ──►   per-paper filtering (A3w by default)
abstract triage → PASS / SKIP                  pdf_checklist.csv      keep / discard + reasoning
```

## Install

Python 3.12 is tested.

```bash
git clone <this repository> PROSPECT
cd PROSPECT
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

## Configure

Copy `.env.example` to `.env` and fill it in. You need:

| Setting | Needed for | Notes |
|---|---|---|
| `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` | Steps 1 and 2 | Any OpenAI-compatible chat endpoint. **Local:** LM Studio, Ollama or vLLM on your machine (no key needed). **Cloud:** OpenAI or any provider with an OpenAI-compatible endpoint (key and model ID required). |
| `SCOPUS_API_KEY` | Step 1 retrieval | Free key from [dev.elsevier.com](https://dev.elsevier.com/). Not needed if you skip Scopus. |
| `SCOPUS_API_INSTITUTION_KEY` | optional | Institutional token, if you have one. |

Every setting can also be passed on the command line (`--base-url`, `--model`, `--api-key`), which overrides `.env`.

The paper's runs used **gpt-oss-120b served by LM Studio**, temperature 0.1.

## Quick start

All commands run from the repository root. Each project lives in one **run folder**, here `runs/my_topic`.

```bash
# Step 1: describe your application, search Scopus, triage abstracts
python step_1/run_step1.py --run-dir runs/my_topic

# Download the PDFs listed in runs/my_topic/step1/pdf_checklist.csv
# into runs/my_topic/papers/ (use the filenames in the checklist)

# Step 2: filter the PDFs (condition A3w by default)
python step_2/run_step2.py --run-dir runs/my_topic
```

Your curated corpus is in `runs/my_topic/A3w/routed_pdfs/keep/`.

PROSPECT does not download full texts: access to most papers depends on your institution's subscriptions. The checklist lists each paper's DOI link and the filename to save it under. Step 2 matches a PDF to its Scopus metadata through the DOI in the filename. A PDF without a match still gets filtered, but its first pages stand in for the abstract.

## Choose how much of Step 1 to run

| You have | Run |
|---|---|
| A Scopus key, and want the full pipeline | `python step_1/run_step1.py --run-dir runs/x` |
| A Scopus key, and want to keep every retrieved paper | `python step_1/run_step1.py --run-dir runs/x --skip-triage` |
| No Scopus key, or your own PDFs, but want the guided conversation to write the application description | `python step_1/run_step1.py --run-dir runs/x --skip-scopus`, then put your PDFs in `runs/x/papers/` |
| Your own PDFs and your own application description | Skip Step 1. Copy `templates/intent_brief.example.json`, edit it, then run `python step_2/run_step2.py --run-dir runs/x --input my_brief.json --papers-dir path/to/pdfs` |

Triage can also be run later on an untriaged corpus: `python step_1/run_step1.py --run-dir runs/x --triage-only`.

In every case, Step 2 builds its filtering criteria from the same **intent brief**: domain, application context, knowledge axes, scope boundaries and content anti-patterns. In Step 1 this brief comes out of the LLM conversation and you confirm it. If you skip Step 1, you write it yourself.

## Step 2 conditions

PROSPECT's default filter is **A3w**. The other four conditions are the architectures compared in the paper's ablation. They are included so the comparison can be repeated.

| Condition | What it does | LLM calls per paper (n = chunks) |
|---|---|---|
| **A3w** (default) | Four specialist agents per chunk: Paper Profiler, Relevance Assessor, Utility Simulator, Aggregator Judge. The Profiler also assigns application-conditioned weights (0–2) to the chunks, which weight the deterministic composite score the Judge sees. | 2n + 3 |
| A3 | As A3w, with uniform chunk weights | 2n + 2 |
| A4 | As A3, without the Utility Simulator | n + 2 |
| A2 | Per chunk, all agent roles in one prompt, then one aggregation call | n + 1 |
| A1 | Whole paper in one call | 1 |

Chunks are 1,400 words with 200 words of overlap (tails under 300 words are merged), so a typical journal article gives 3–6 chunks. Retries for malformed replies add calls. The configuration pre-step adds one call per run folder.

```bash
python step_2/run_step2.py --run-dir runs/my_topic --condition A1
python step_2/run_step2.py --run-dir runs/my_topic --condition A3 --resume   # continue an interrupted run
python step_2/run_step2.py --run-dir runs/my_topic --prepare-only            # generate the config, review it, stop
```

**Same inputs for every condition.** The first Step 2 call in a run folder fixes the input file and the PDF folder in `run_inputs.json`, and generates `operational_config.json`. Every later condition reads both. If you pass a different input or PDF folder, or edit the input file afterwards, Step 2 refuses to run. A2, A3, A3w and A4 also profile each paper from the same context: the domain, the Scopus metadata and abstract, and the last pages. A1 reads the whole paper instead. So conditions in one run folder differ only in their architecture.

You can review or edit `operational_config.json` (from `--prepare-only`) before running any condition. Every condition will use your edited version.

## Run folder

```
runs/my_topic/
├── step1/
│   ├── intent_brief.json         confirmed application description
│   ├── corpus.json               Scopus metadata of all retrieved papers
│   ├── corpus_triaged.json       papers that passed abstract triage
│   ├── triage_report.json        PASS / SKIP and reason per paper
│   ├── pdf_checklist.csv         PDFs to download and their expected filenames
│   └── run_log.md                conversation, queries, counts
├── papers/                       the PDFs you downloaded
├── run_inputs.json               input file + PDF folder shared by all conditions
├── operational_config.json       criteria, rubrics, queries, anti-patterns
└── A3w/                          one folder per condition you run
    ├── routed_pdfs/keep/         ← the curated corpus
    ├── routed_pdfs/discard/
    ├── paper_decisions/          one JSON per paper: decision, score, reasoning
    ├── chunks_*.jsonl            per-chunk agent assessments (A3w, A3, A4)
    ├── filter_results_*.xlsx     papers, chunks and query coverage (A3w, A3, A4)
    ├── run_manifest.json         how the run was invoked
    └── run_metrics.json          timing and LLM calls per paper
```

`runs/` is git-ignored. PDFs are usually copyrighted, so do not commit or redistribute them.

## Practical notes

- **Local models and context length.** A1 sends a whole paper in one prompt. Give your local server a context window large enough for your longest paper; the paper's runs used 120k tokens.
- **Long local runs.** Some local models degrade after many hours of continuous inference. `step_2/tools/llm_watchdog.py` supervises a Step 2 run on LM Studio: it reloads the model when replies degenerate and resumes the run. See its docstring.
- **Cloud APIs.** The pipeline sets `temperature` on every call (0.3 for the Step 1 conversation and query design, 0.1 elsewhere), so the model you choose must accept a temperature parameter. Check your provider's pricing against the call counts above before filtering a large corpus.
- **Per-agent models.** For A3w, A3 and A4 you can give each agent its own model, e.g. `--agent4-model <id>`.
- **Scopus quota.** `python step_1/tools/check_scopus_quota.py` shows your remaining quota. `step_1/tools/backfill_abstracts.py` fetches abstracts missing from a corpus.

More detail: [step_1/README.md](step_1/README.md), [step_2/README.md](step_2/README.md).

## Evaluation

[`evaluation/`](evaluation/README.md) is the code behind the paper's evaluation, for anyone who wants to repeat or extend it on their own run. It is separate from the method and has its own requirements (`evaluation/requirements.txt`).

- **Filter validity:** a panel of LLM judges and human raters scores a stratified sample of papers. The number and kind of raters are settings.
- **RAG evaluation:** each filtered corpus answers a fixed query set, scored with RAGAS (Lens A) and with gold relevance labels (Lens B).
- **The paper's own settings and fixed inputs** (sample, queries, reference answers) are in `evaluation/examples/scenario_1/`.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The tests use a fake LLM: they never call a model or the Scopus API. They cover every Step 1 mode and all five Step 2 conditions end to end, as well as the run-folder input lock. The evaluation tests (statistics, the rater panel with different numbers of raters, Part 2 statistics) run when `evaluation/requirements.txt` is installed and are skipped otherwise.

## Relation to the paper

This repository is the reference implementation of the method described in the paper, cleaned up for reuse after the paper's experiments.

## Citation

The paper describing PROSPECT is in preparation:
*PROSPECT: Application-Conditioned Curation of Retrieval-Augmented Generation Corpora with Adaptive Chunk Scoring.* Citation details will be added on publication.

## License

MIT — see [LICENSE](LICENSE).

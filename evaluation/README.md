# PROSPECT evaluation

How the paper evaluated PROSPECT, as runnable code. Nothing here is needed to *use* PROSPECT; it is for replicating or extending the evaluation on your own run.

There are two parts:

| Part | Question | How |
|---|---|---|
| **1. Filter validity** (`filter_validity/`) | Do the conditions' keep/discard decisions match expert judgement? | A stratified sample of papers is scored 1–4 by a panel of LLM judges and human raters (the Expert-LLM Reference Panel); a majority rule gives a reference label; each condition is scored against it. |
| **2. RAG evaluation** (`retrieval_rag/`) | Does a filtered corpus serve the application as well as the full one? | Each corpus is indexed and answers a fixed query set. **Lens A** scores the answers with RAGAS; **Lens B** checks, with gold relevance labels, which corpora keep the papers that answer each query. |

No data is included except the paper's fixed evaluation inputs (see [Reproducing the paper](#reproducing-the-paper)). The paper's PDFs, extracted texts, indexes, RAG outputs and scores are not distributed.

## Setup

```bash
pip install -r requirements.txt -r evaluation/requirements.txt
```

- **API keys** go in the repository's `.env`, read by name: `OPENAI_API_KEY` (an OpenAI judge, and the RAGAS judge), `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`. You only need the ones your settings use. A local OpenAI-compatible server needs no key (`provider: local`).
- **Part 2 also needs** a Chroma server and Ollama with the embedding model:
  ```bash
  chroma run --path runs/chroma --port 8000      # in its own terminal
  ollama pull embeddinggemma
  ```
- **Settings.** Copy [`settings.yaml`](settings.yaml), fill in every `TODO`, and pass it to each script with `--settings`. It names the PROSPECT run folder, where outputs go, which model plays which role, the human raters, sample sizes and seeds. Run scripts from the repository root.
- **Check the setup** before spending anything:
  ```bash
  python evaluation/check_setup.py --settings runs/my_topic/eval_settings.yaml --services
  ```

The evaluation reads a PROSPECT run folder: one sub-folder per condition with `paper_decisions/` and `routed_pdfs/keep/`, the `papers/` folder and `operational_config.json`. Run all five conditions on the same run folder first (see the main README).

## Part 1 — Filter validity

```bash
S=runs/my_topic/eval_settings.yaml
python evaluation/filter_validity/build_strata.py        --settings $S   # which conditions kept each paper
python evaluation/filter_validity/extract_texts.py       --settings $S   # paper texts for the LLM judges
python evaluation/filter_validity/draw_sample.py         --settings $S   # stratified sample (preview)
python evaluation/filter_validity/draw_sample.py         --settings $S --lock
python evaluation/filter_validity/run_calibration.py     --settings $S   # 5 papers, all judges; review
python evaluation/filter_validity/run_llm_judges.py      --settings $S   # every judge x every paper
python evaluation/filter_validity/prepare_human_packets.py --settings $S # one blinded packet per rater
#   ... raters return their sheets as <workspace>/filter_validity/human_scores/<rater>.csv
python evaluation/filter_validity/build_reference_labels.py --settings $S
python evaluation/filter_validity/compute_metrics.py     --settings $S
```

**Sample.** Papers are grouped by how many conditions kept them. Papers kept by all, or by exactly one condition, are all taken; papers kept by 2, 3 or 4 are capped per fingerprint; papers kept by none are sampled one per TF-IDF topic cluster. Disagreements are therefore over-represented. Lock the sample before anyone rates.

**Raters.** Everything about the panel is a setting:

```yaml
filter_validity:
  llm_judges:                     # any number, any provider; use different model families
    - {id: llm-a, provider: openai, model: gpt-5.5}
    - {id: llm-b, provider: google, model: gemini-pro-latest, temperature: 0.0}
  human_raters: [H1, H2]          # anonymous IDs; any number, or [] for an LLM-only panel
  consensus: {keep_score: 3, min_keep_votes: majority}
```

All raters use one rubric: a 1–4 usefulness score, a confidence, and a rationale that cites the paper. The LLM prompt is [`prompts/llm_judge.txt`](filter_validity/prompts/llm_judge.txt). The human instructions start from [`human_rubric_template.md`](filter_validity/human_rubric_template.md): adapt it to your application, ideally with worked examples from the calibration round. Human packets contain only paper IDs and PDFs: no condition names, decisions, LLM scores or strata. Fix the consensus rule before rating starts.

**Outputs** (`<workspace>/filter_validity/metrics/`):

| File | Contents |
|---|---|
| `condition_metrics.csv` | Per condition: TP/FP/FN/TN, precision, recall, F1, MCC, each with a 95% bootstrap CI (10,000 resamples of papers) |
| `agreement.csv`, `rater_pairs.csv` | Fleiss' κ (binary) for the LLMs, all raters and the humans; pairwise Cohen's κ, quadratic-weighted κ on the 1–4 scores, Spearman ρ; the same on the high-confidence subset |
| `alternative_references.csv` | Each condition's F1 and rank when the reference is an LLM majority, all LLMs agreeing, all humans agreeing, or each rater alone |
| `disagreements.csv`, `report.md` | Every decision that differs from the reference, with rationales; a readable summary |

## Part 2 — RAG evaluation

```bash
python evaluation/retrieval_rag/build_corpora.py       --settings $S   # unfiltered + each condition's kept papers
python evaluation/retrieval_rag/index_corpora.py       --settings $S   # one Chroma collection each + build manifest
python evaluation/retrieval_rag/build_queries.py       --settings $S   # layer 1: the operational config's queries
python evaluation/retrieval_rag/write_held_out_queries.py --settings $S # layer 3: ~10 held-out queries (you write them)
python evaluation/retrieval_rag/generate_paraphrases.py --settings $S  # layer 2: paraphrases, reviewed by you
python evaluation/retrieval_rag/calibrate_k.py         --settings $S   # choose how many chunks the drafter sees
python evaluation/retrieval_rag/draft_references.py    --settings $S   # one reference answer per layer 1/3 query
python evaluation/retrieval_rag/run_rag_grid.py        --settings $S   # every query x every corpus
python evaluation/retrieval_rag/score_ragas.py         --settings $S   # Lens A scores
python evaluation/retrieval_rag/score_self_consistency.py --settings $S # independent judge on 25%
python evaluation/retrieval_rag/answer_quality_stats.py --settings $S  # Lens A statistics
python evaluation/retrieval_rag/label_gold.py          --settings $S   # Lens B: label every (query, paper)
python evaluation/retrieval_rag/build_gold_spot_check.py --settings $S # blinded human spot check
python evaluation/retrieval_rag/label_gold.py          --settings $S --mode spot-check
python evaluation/retrieval_rag/gold_spot_check_agreement.py --settings $S
python evaluation/retrieval_rag/gold_metrics.py        --settings $S   # Lens B statistics
```

**Fixed design.** Every corpus is indexed the same way (500-token chunks with 50 overlap, the same embedding model, cosine similarity), answers the same queries with the same generator and prompt, and is scored by the same judges. The only difference between corpora is which papers the filter kept. Fix the queries, references and index before generating any answers.

**Queries.** Layer 1 is the operational config's representative queries, verbatim. Layer 3 holds about 10 queries written by a person, within scope but not restating layer 1; it tests generalisation. Layer 2 paraphrases layer 1; it tests robustness only and is never pooled with layers 1 and 3.

**Lens A — answer quality.** RAGAS context recall and answer correctness on every answer; faithfulness, context precision and answer relevancy on layer 1. The reference answers are drafted from the unfiltered corpus, which caps the filtered corpora's context recall, so read Lens A as retention and non-inferiority, not superiority. Use three different model families for the drafter, generator and judge, so no model grades its own writing. An independent judge from the third family re-scores 25% of the answers; agreement is the Spearman ρ between the two judges' per-corpus means.

**Lens B — corpus quality.** A labeller model marks each (query, paper) pair 0, 1 or 2 from the title and the start of the text, without seeing any answer or filter result. From the gold sets: per corpus precision (Wilson CI), recall and F1; per query recall. A blinded human spot check and a second model validate the labels (Cohen's κ).

**Statistics.** The design has four planned contrasts, each isolating one mechanism: A1→A2 chunking, A2→A3 decomposition, A3→A3w adaptive weights, A3→A4 the Utility Simulator (A4 − A3 = removing it). Each is a paired Wilcoxon test over queries, Holm-corrected over the four, with a rank-biserial effect size and a bootstrap CI on the mean difference. They are reported two-sided; set `alternative` per contrast to add a directional test.

**Outputs** are in `<workspace>/retrieval_rag/results/` (`answer_quality/`, `self_consistency/`, `gold/`, `gold_spot_check/`), each with CSVs and a `report.md`.

## Reproducing the paper

[`examples/scenario_1/`](examples/scenario_1/) holds the paper's settings and its fixed inputs:

| File | What |
|---|---|
| `settings.yaml` | Every model, prompt value, sample size and seed used in the paper |
| `sample_locked.json` | The 101-paper Step 2 sample (seed 42) |
| `application_description.txt` | The text all raters saw |
| `human_rubric.md` | The human raters' instructions (raters anonymised as H1, H2) |
| `queries/layer_1.json`, `layer_2.json`, `layer_3.json` | 24 + 24 + 10 queries |
| `references/*.json` | The 34 reference answers (layers 1 and 3) |

With the paper's run folder in place (`run_dir`), these settings reproduce the paper's sample, strata, reference labels and every reported statistic from the saved scores. This was checked against the paper's outputs: differences are rounding only. New model calls will differ somewhat: hosted models change behind aliases such as `gemini-pro-latest`, and some models run only at their default temperature.

## Cost and time (paper scale)

| Step | Calls | Notes |
|---|---|---|
| LLM judges | 3 × 101 papers | about $10–15 with the paper's models; full paper texts |
| RAG grid | 58 queries × 6 corpora | Claude Opus, 2,500 output tokens max |
| RAGAS | 348 rows | GPT-4o; several calls per metric |
| Gold labels | 34 queries × 212 papers = 7,208 | Gemini Flash; the largest number of calls but cheap |
| Human raters | 101 papers each | about 3–5 minutes per paper |

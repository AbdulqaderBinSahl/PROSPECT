"""
Abstract triage for a Step 1 corpus.

Sends each paper's title + abstract to the LLM to classify as PASS or SKIP.
Papers where there is any reasonable doubt are kept (PASS). Only obvious
non-relevant papers are removed.

Called from ``run_step1.py`` (after Scopus retrieval, or with --triage-only).

Outputs (in the Step 1 output folder):
    - corpus_triaged.json   — same structure, only PASS papers
    - triage_report.json    — full classification log for every paper
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM generation settings for triage
# ---------------------------------------------------------------------------
LLM_TEMPERATURE = 0.1  # low temp for consistent classification
LLM_MAX_TOKENS = 256   # only need a short response


# ---------------------------------------------------------------------------
# Triage prompt builder
# ---------------------------------------------------------------------------

def build_triage_prompt(intent_brief: dict) -> str:
    """Build the system prompt from the corpus intent brief."""

    domain = intent_brief.get("domain", "")
    app_context = intent_brief.get("application_context", "")
    scope_boundaries = intent_brief.get("scope_boundaries", [])
    anti_patterns = intent_brief.get("content_anti_patterns", [])
    axes = intent_brief.get("knowledge_axes", [])

    axes_text = "\n".join(f"  - {ax['name']}: {ax['description']}" for ax in axes)
    boundaries_text = "\n".join(f"  - {b}" for b in scope_boundaries)
    anti_text = "\n".join(f"  - {a}" for a in anti_patterns)

    return f"""You are a scientific literature triage assistant. Your DEFAULT answer is PASS.

DOMAIN: {domain}
APPLICATION: {app_context}

KNOWLEDGE AXES the corpus must cover:
{axes_text}

SCOPE BOUNDARIES (papers matching these should be SKIPPED):
{boundaries_text}

CONTENT ANTI-PATTERNS (papers matching ALL of these should be SKIPPED):
{anti_text}

YOUR TASK:
Given a paper's title and abstract, classify it as PASS or SKIP.
Your default classification is PASS. You must PASS unless you are highly confident the paper is irrelevant.

ALWAYS PASS if the paper:
- Addresses any of the knowledge axes listed above
- Reports experimental data (properties, measurements, formulations) relevant to the domain
- Discusses materials, processes, or methods directly related to the application context
- Could provide useful background or supporting knowledge for the application
- Is ambiguous or you are unsure

ONLY SKIP if you are HIGHLY CONFIDENT the paper meets ALL of these conditions:
1. The paper falls clearly outside the domain described above, AND
2. The paper has ZERO relevance to any of the knowledge axes, AND
3. The paper matches one or more scope boundaries listed above

CRITICAL: A false SKIP is 10x worse than a false PASS. If there is even 1% doubt, output PASS.

RESPOND with exactly one line:
PASS
or
SKIP | <one-sentence reason>

Nothing else."""


def build_user_prompt(title: str, abstract: str | None) -> str:
    """Build the per-paper user prompt."""
    abs_text = abstract if abstract else "[No abstract available]"
    return f"Title: {title}\n\nAbstract: {abs_text}"


def classify_paper(
    client: OpenAI,
    model: str,
    system_prompt: str,
    title: str,
    abstract: str | None,
) -> tuple[str, str]:
    """Classify a single paper. Returns (verdict, reason)."""
    user_msg = build_user_prompt(title, abstract)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("LLM call failed for '%s': %s — defaulting to PASS", title[:50], exc)
        return "PASS", f"LLM error: {exc}"

    # Parse response
    line = raw.split("\n")[0].strip()

    if line.upper().startswith("PASS"):
        return "PASS", ""
    elif line.upper().startswith("SKIP"):
        reason = line.split("|", 1)[1].strip() if "|" in line else "No reason given"
        return "SKIP", reason
    else:
        # Unparseable response — default to PASS (safe side)
        logger.warning("Unparseable LLM response for '%s': %s — defaulting to PASS", title[:50], raw[:100])
        return "PASS", f"Unparseable response: {raw[:100]}"


# ---------------------------------------------------------------------------
# Triage run
# ---------------------------------------------------------------------------

def run_triage(
    corpus_path: Path,
    client: OpenAI,
    model: str,
    out_dir: Path,
    delay: float = 0.0,
) -> Path:
    """Triage every paper in ``corpus_path`` and write the filtered corpus.

    Returns the path to ``corpus_triaged.json``.
    """
    corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    intent_brief = corpus.get("intent_brief", {})
    papers = corpus.get("papers", [])
    if not papers:
        raise ValueError(f"No papers to triage in {corpus_path}")

    system_prompt = build_triage_prompt(intent_brief)

    print("\n" + "=" * 60)
    print(f"  TRIAGE: classifying {len(papers)} papers")
    print("=" * 60 + "\n")

    results: list[dict] = []
    pass_count = 0
    skip_count = 0
    no_abstract = 0
    start_time = time.time()

    for i, paper in enumerate(papers, 1):
        title = paper.get("title", "")
        abstract = paper.get("abstract")
        doi = paper.get("doi", "")

        if not abstract:
            no_abstract += 1

        verdict, reason = classify_paper(client, model, system_prompt, title, abstract)

        results.append({
            "index": i,
            "doi": doi,
            "title": title,
            "verdict": verdict,
            "reason": reason,
        })

        if verdict == "PASS":
            pass_count += 1
        else:
            skip_count += 1

        # Progress
        short_title = title[:55] + "..." if len(title) > 55 else title
        if verdict == "SKIP":
            print(f"  [x] {i:>4}/{len(papers)}  SKIP  {short_title}")
            print(f"         Reason: {reason}")
        elif i % 25 == 0 or i == len(papers):
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            print(f"  ... {i:>4}/{len(papers)}  ({pass_count} pass, {skip_count} skip)  "
                  f"[{rate:.1f} papers/sec]")

        if delay > 0:
            time.sleep(delay)

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("  TRIAGE COMPLETE")
    print("=" * 60)
    print(f"  Total:        {len(papers)}")
    print(f"  PASS:         {pass_count}")
    print(f"  SKIP:         {skip_count}")
    print(f"  No abstract:  {no_abstract}")
    print(f"  Time:         {elapsed:.1f}s")
    print(f"  Retention:    {pass_count/len(papers)*100:.1f}%")

    # Build filtered corpus (same structure, only PASS papers)
    pass_indices = {r["index"] - 1 for r in results if r["verdict"] == "PASS"}
    filtered_papers = [p for i, p in enumerate(papers) if i in pass_indices]

    filtered_corpus = dict(corpus)
    filtered_corpus["papers"] = filtered_papers

    # Update summary statistics
    if "summary" in filtered_corpus:
        filtered_corpus["summary"]["total_unique_papers"] = len(filtered_papers)

    # Add triage metadata
    filtered_corpus["triage"] = {
        "performed_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "original_count": len(papers),
        "filtered_count": len(filtered_papers),
        "skipped_count": skip_count,
        "retention_pct": round(pass_count / len(papers) * 100, 1) if papers else 0,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    filtered_path = out_dir / "corpus_triaged.json"
    report_path = out_dir / "triage_report.json"

    filtered_path.write_text(
        json.dumps(filtered_corpus, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  Triaged corpus: {filtered_path}")

    report = {
        "source": str(corpus_path),
        "performed_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "total_papers": len(papers),
        "pass_count": pass_count,
        "skip_count": skip_count,
        "no_abstract_count": no_abstract,
        "elapsed_seconds": round(elapsed, 1),
        "classifications": results,
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Triage report:  {report_path}\n")
    return filtered_path

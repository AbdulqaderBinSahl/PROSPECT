"""The rater panel: LLM judges, human raters, the judge prompt and score files.

Raters are defined in the settings file:

    filter_validity:
      llm_judges:   [{id: llm-a, provider: openai, model: gpt-5.5}, ...]   # any number
      human_raters: [H1, H2]                                               # any number, or []

Every rater scores a paper 1-4 (4 = highly valuable) with a confidence
(low / medium / high) and a short rationale. LLM scores go to one JSONL file;
each human returns a CSV score sheet saved as ``human_scores/<rater id>.csv``.
"""

from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from evaluation.shared.llm import LLMClient, ModelSpec, parse_json_object
from evaluation.shared.settings import Settings, read_jsonl

SCORES = (1, 2, 3, 4)
CONFIDENCE = ("low", "medium", "high")

CALIBRATION_HEADER = (
    "# CALIBRATION ROUND — {n} papers only\n"
    "# These papers are used to check inter-judge alignment before the main evaluation run.\n"
    "# Score and rationale will be compared across all judges (LLM and human) after completion.\n\n"
)


# ── paths ───────────────────────────────────────────────────────────────────


def fv_path(s: Settings, *parts: str) -> Path:
    return s.out("filter_validity", *parts)


def texts_dir(s: Settings) -> Path:
    d = s.workspace / "filter_validity" / "texts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def sample_file(s: Settings) -> Path:
    """The locked sample: a file named in the settings, else the workspace one."""
    return s.path("filter_validity.sample_file", None) or fv_path(s, "sample_locked.json")


def llm_scores_file(s: Settings) -> Path:
    return fv_path(s, "llm_scores.jsonl")


def human_scores_file(s: Settings, rater: str) -> Path:
    return fv_path(s, "human_scores", f"{rater}.csv")


# ── raters ──────────────────────────────────────────────────────────────────


def llm_judge_specs(s: Settings) -> List[Dict]:
    judges = s.get("filter_validity.llm_judges")
    ids = [j["id"] for j in judges]
    if len(set(ids)) != len(ids):
        raise SystemExit("\nERROR: llm_judges ids must be unique.\n")
    return judges


def human_raters(s: Settings) -> List[str]:
    return [str(r) for r in (s.get("filter_validity.human_raters", []) or [])]


def all_raters(s: Settings) -> List[str]:
    return [j["id"] for j in llm_judge_specs(s)] + human_raters(s)


def keep_score(s: Settings) -> int:
    return int(s.get("filter_validity.consensus.keep_score", 3))


def min_keep_votes(s: Settings) -> int:
    """Votes needed for a "keep" reference label ("majority" = more than half the panel)."""
    rule = s.get("filter_validity.consensus.min_keep_votes", "majority")
    n = len(all_raters(s))
    return n // 2 + 1 if rule == "majority" else int(rule)


# ── the judge ───────────────────────────────────────────────────────────────


@dataclass
class JudgeResponse:
    paper_id: str
    judge_name: str
    model_name: str
    score: int
    confidence: str
    rationale: str
    raw_response: str
    timestamp: str
    cost_usd: float


def render_prompt(s: Settings, paper_text: str, calibration_n: int = 0) -> str:
    """The judge prompt for one paper (with the calibration header if ``calibration_n``)."""
    from jinja2 import Template

    cfg = s.get("filter_validity.judge_prompt")
    template = s.resolve(cfg["template"]).read_text(encoding="utf-8")
    prompt = Template(template).render(
        application_description=s.application_description(),
        paper_text=paper_text,
        example_questions=cfg.get("example_questions", ""),
        core_area=cfg["core_area"],
    )
    return (CALIBRATION_HEADER.format(n=calibration_n) + prompt) if calibration_n else prompt


def parse_judgement(raw: str, paper_id: str) -> Dict:
    """Validate a judge reply: integer score 1-4, confidence, non-empty rationale."""
    data = parse_json_object(raw)
    score = data.get("score")
    if not isinstance(score, int) or score not in SCORES:
        raise ValueError(f"invalid score {score!r} for {paper_id}; expected an integer 1-4")
    confidence = str(data.get("confidence", "")).lower()
    if confidence not in CONFIDENCE:
        raise ValueError(f"invalid confidence {confidence!r} for {paper_id}")
    if not data.get("rationale"):
        raise ValueError(f"missing rationale for {paper_id}")
    return {"score": score, "confidence": confidence, "rationale": data["rationale"]}


class Judge:
    """One LLM judge from the settings."""

    def __init__(self, spec: Dict) -> None:
        self.judge_id = spec["id"]
        self.client = LLMClient(ModelSpec.from_settings(spec))

    def judge(self, paper_id: str, prompt: str) -> JudgeResponse:
        result = self.client.complete(prompt)
        parsed = parse_judgement(result.text, paper_id)
        return JudgeResponse(
            paper_id=paper_id, judge_name=self.judge_id, model_name=self.client.model,
            raw_response=result.text, timestamp=datetime.now(timezone.utc).isoformat(),
            cost_usd=result.cost_usd, **parsed,
        )

    @staticmethod
    def record(response: JudgeResponse) -> Dict:
        return asdict(response)


# ── texts and scores ────────────────────────────────────────────────────────


def paper_text(s: Settings, paper_id: str) -> str:
    p = texts_dir(s) / f"{paper_id}.txt"
    if not p.exists():
        raise FileNotFoundError(f"No extracted text for {paper_id}; run extract_texts.py first.")
    return p.read_text(encoding="utf-8")


def load_llm_scores(s: Settings) -> Dict[str, Dict[str, Dict]]:
    """{paper_id: {judge_id: {score, confidence, rationale}}}; later lines win."""
    out: Dict[str, Dict[str, Dict]] = {}
    for rec in read_jsonl(llm_scores_file(s)):
        out.setdefault(rec["paper_id"], {})[rec["judge_name"]] = {
            "score": int(rec["score"]), "confidence": rec.get("confidence", "medium"),
            "rationale": rec.get("rationale", ""),
        }
    return out


def load_human_scores(s: Settings, rater: str) -> Dict[str, Dict]:
    """{paper_id: {score, confidence, rationale}} from one returned score sheet."""
    path = human_scores_file(s, rater)
    if not path.exists():
        return {}
    out: Dict[str, Dict] = {}
    with path.open(encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            pid = (row.get("paper_id") or "").strip()
            raw = (row.get("score") or "").strip()
            if not pid or not raw:
                continue
            try:
                score = int(raw)
            except ValueError:
                print(f"  Warning: {rater}: invalid score {raw!r} for {pid}, skipped")
                continue
            out[pid] = {
                "score": score,
                "confidence": (row.get("confidence") or "").strip().lower() or "medium",
                "rationale": (row.get("rationale") or "").strip(),
            }
    return out


def all_scores(s: Settings) -> Dict[str, Dict[str, Dict]]:
    """{paper_id: {rater_id: {...}}} for every LLM judge and every returned human sheet."""
    scores = load_llm_scores(s)
    known = {j["id"] for j in llm_judge_specs(s)}
    for pid in list(scores):
        scores[pid] = {k: v for k, v in scores[pid].items() if k in known}
    for rater in human_raters(s):
        for pid, entry in load_human_scores(s, rater).items():
            scores.setdefault(pid, {})[rater] = entry
    return scores


def natural_key(text: str) -> List:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]

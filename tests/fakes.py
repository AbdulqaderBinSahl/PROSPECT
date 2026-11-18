"""A fake LLM that returns schema-valid answers for every PROSPECT call.

Papers whose filename contains ``Paper1`` are kept, all others discarded, so
tests can check routing. Every call is recorded for inspection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from rag_corpus_filter.models import OperationalConfig, ValueCriterion

QUERIES = [
    "How does elastomer content change impact strength?",
    "What melt flow rates are reported for recycled blends?",
]
CRITERIA = ["quantitative_density", "formulation_actionability"]


def fake_operational_config() -> OperationalConfig:
    return OperationalConfig(
        application_description="Fake application description.",
        domain_taxonomy=["impact_strength", "melt_flow"],
        representative_queries=QUERIES,
        value_criteria=[
            ValueCriterion(name=n, description=n, scale_anchors={0: "none", 5: "full"}) for n in CRITERIA
        ],
        anti_patterns=["Purely qualitative market commentary"],
        strong_chunk_min_criteria=1,
        strong_chunk_score_threshold=3,
        min_strong_chunks_for_keep=1,
    )


def _keep(label: str) -> bool:
    return "Paper1_" in label


def _payload(model_name: str, fields: set, human_prompt: str, label: str) -> Dict[str, Any]:
    """Schema-valid JSON for each response model."""
    if model_name == "OperationalConfig":
        return fake_operational_config().model_dump()
    if model_name == "PaperProfile":
        return {
            "paper_type": "experimental", "methodology_summary": "m", "scope": "s",
            "data_types_present": ["mechanical_properties"], "domain_fit_initial": "strong",
            "profiler_notes": "n",
        }
    if model_name == "ChunkWeightsResponse":
        ids = sorted(set(re.findall(r"\b(c\d{4})\b", human_prompt)))
        return {"chunk_weights": [
            {"chunk_id": cid, "section_type": "results", "weight": 1.5, "reasoning": "r"} for cid in ids
        ]}
    if model_name == "ChunkRelevanceQuality":
        return {
            "domain_relevance_0_to_5": 4, "assigned_labels": ["impact_strength"], "anti_pattern_flags": [],
            "criteria_scores": {c: 4 for c in CRITERIA}, "figure_table_dependency": "none",
            "standalone_comprehensibility": "mostly_standalone", "evidence_snippets": [],
            "contains_decision_rules": False, "contains_mechanisms_or_causes": True,
            "contains_quantitative_data": True, "keywords": ["k"], "assessor_notes": "n",
        }
    if model_name == "ChunkUtilityAssessment":
        return {
            "query_assessments": [{"query": q, "utility_score_0_to_5": 4, "reasoning": "r"} for q in QUERIES],
            "overall_utility_0_to_5": 4, "utility_type": "direct_answer",
            "best_served_queries": QUERIES[:1], "simulator_notes": "n",
        }
    if model_name == "PaperDecision" and "main_reasoning" in fields:  # A3 family judge
        return {
            "decision": "keep" if _keep(label) else "discard", "overall_score_1_to_10": 7,
            "main_reasoning": "r", "strengths": [], "weaknesses": [],
            "best_case_for_keeping": "k", "best_case_for_discarding": "d", "why_decision_wins": "w",
            "agent_disagreements": [], "recommended_tags": [], "top_evidence": [],
        }
    if model_name in ("PaperDecision", "WholePaperDecision"):  # A2 aggregation / A1
        return {
            "decision": "keep" if _keep(label) else "discard", "overall_score_1_to_10": 7,
            "reasoning": "r", "strengths": [], "weaknesses": [],
        }
    if model_name == "ChunkAllRolesAssessment":
        return {
            "paper_type": "experimental", "data_types_observed": ["impact data"], "domain_fit": "strong",
            "scope_notes": "s", "domain_relevance_0_to_5": 4, "criteria_scores": {c: 4 for c in CRITERIA},
            "anti_pattern_flags": [], "contains_quantitative_data": True,
            "query_assessments": [{"query": q, "utility_score_0_to_5": 4, "reasoning": "r"} for q in QUERIES],
            "overall_utility_0_to_5": 4, "best_served_queries": QUERIES[:1], "assessor_notes": "n",
        }
    raise AssertionError(f"Fake LLM has no answer for {model_name}")


@dataclass
class FakeLLM:
    """Records calls; stands in for both LLM call paths (agents and baselines)."""
    calls: List[Dict[str, str]] = field(default_factory=list)

    def _answer(self, human_prompt, response_model, context_label):
        self.calls.append({"label": context_label, "prompt": human_prompt, "model": response_model.__name__})
        data = _payload(response_model.__name__, set(response_model.model_fields), human_prompt, context_label)
        return response_model.model_validate(data)

    # signature of rag_corpus_filter.llm_utils.call_llm_structured
    def structured(self, llm, system_prompt, human_prompt, response_model, config,
                   context_label="", paper_metrics=None):
        return self._answer(human_prompt, response_model, context_label)

    # signature of baselines.shared.call_llm_json
    def baseline(self, system_prompt, human_prompt, response_model, endpoint, model, api_key="",
                 temperature=0.1, max_tokens=4096, max_retries=3, context_label=""):
        return self._answer(human_prompt, response_model, context_label)

    def labels(self, contains: str) -> List[str]:
        return [c["label"] for c in self.calls if contains in c["label"]]

    def prompt_for(self, label_contains: str) -> str:
        return next(c["prompt"] for c in self.calls if label_contains in c["label"])


def install(monkeypatch) -> FakeLLM:
    """Route every LLM call in Step 2 to a FakeLLM."""
    fake = FakeLLM()
    for module in (
        "rag_corpus_filter.config_prestep",
        "rag_corpus_filter.paper_profiler",
        "rag_corpus_filter.relevance_assessor",
        "rag_corpus_filter.utility_simulator",
        "rag_corpus_filter.aggregator_judge",
    ):
        monkeypatch.setattr(f"{module}.call_llm_structured", fake.structured)
    monkeypatch.setattr("baselines.shared.call_llm_json", fake.baseline)
    return fake

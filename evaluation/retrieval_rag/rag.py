"""Shared pieces of the RAG evaluation: corpora, queries, retrieval and generation.

Corpora: "unfiltered" (every PDF that was filtered) plus one per condition
(its kept PDFs). Each corpus is indexed as its own Chroma collection with the
same chunking and embedding, so the only difference between collections is
which papers the filter kept.

Queries come in three layers:
  layer 1  the operational config's representative queries, verbatim
  layer 2  one or more paraphrases of each layer 1 query (robustness only)
  layer 3  held-out queries written by a person, not covered by layer 1
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from evaluation.shared.settings import Settings

UNFILTERED = "unfiltered"
LAYERS = (1, 2, 3)


# ── paths ───────────────────────────────────────────────────────────────────


def rr_path(s: Settings, *parts: str) -> Path:
    return s.out("retrieval_rag", *parts)


def corpora_names(s: Settings) -> List[str]:
    return [UNFILTERED] + s.conditions


def corpus_file(s: Settings, corpus: str) -> Path:
    return rr_path(s, "corpora", f"{corpus}_corpus.json")


def load_corpus(s: Settings, corpus: str) -> Dict[str, Dict]:
    """{paper_id: {"title", "full_text"}} for one corpus."""
    p = corpus_file(s, corpus)
    if not p.exists():
        raise SystemExit(f"\nERROR: {p} not found. Run build_corpora.py first.\n")
    return json.loads(p.read_text(encoding="utf-8"))


def queries_dir(s: Settings) -> Path:
    """A folder with layer_1.json, layer_2.json, layer_3.json (setting, else the workspace)."""
    d = s.path("retrieval_rag.queries.dir", None) or s.workspace / "retrieval_rag" / "queries"
    d.mkdir(parents=True, exist_ok=True)
    return d


def references_dir(s: Settings) -> Path:
    d = s.path("retrieval_rag.references.dir", None) or s.workspace / "retrieval_rag" / "references"
    d.mkdir(parents=True, exist_ok=True)
    return d


def rag_output_file(s: Settings, layer: int, query_id: str, corpus: str) -> Path:
    return rr_path(s, "rag_outputs", f"layer_{layer}", query_id, f"{corpus}.json")


def ragas_csv(s: Settings) -> Path:
    return rr_path(s, "scores", "ragas_scores.csv")


def results_dir(s: Settings, *parts: str) -> Path:
    d = s.workspace.joinpath("retrieval_rag", "results", *parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── queries and references ──────────────────────────────────────────────────


def load_layer(s: Settings, layer: int, required: bool = True) -> List[Dict]:
    p = queries_dir(s) / f"layer_{layer}.json"
    if not p.exists():
        if required:
            raise SystemExit(f"\nERROR: {p} not found (see the query steps in the README).\n")
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def load_queries(s: Settings, layers=(1, 3)) -> List[Dict]:
    out = []
    for layer in layers:
        for q in load_layer(s, layer, required=layer == 1):
            out.append({**q, "layer": q.get("layer", layer)})
    return out


def paraphrase_parents(s: Settings) -> Dict[str, str]:
    """{layer 2 query_id: its layer 1 query_id}."""
    return {q["query_id"]: q["source_query_id"] for q in load_layer(s, 2, required=False)
            if "source_query_id" in q}


def load_references(s: Settings) -> Dict[str, str]:
    """{query_id: reference answer} for layer 1 and 3 queries."""
    refs = {}
    for p in sorted(references_dir(s).glob("*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        refs[data.get("query_id", p.stem)] = data.get("draft", data.get("text", ""))
    return refs


# ── retrieval ───────────────────────────────────────────────────────────────


def collection_name(s: Settings, corpus: str) -> str:
    c = s.get("retrieval_rag.chroma")
    return f"{c.get('collection_prefix', '')}{corpus}{c.get('collection_suffix', '_chroma')}"


class Retriever:
    """Embeds a query and returns the top-k chunks from one corpus's collection."""

    def __init__(self, s: Settings) -> None:
        from evaluation.retrieval_rag.retrieval import store
        from evaluation.retrieval_rag.retrieval.embedder import get_embedder

        chroma = s.get("retrieval_rag.chroma")
        emb = s.get("retrieval_rag.embedding")
        self._s = s
        self._store = store
        self._client = store.get_client(host=chroma.get("host", "localhost"), port=int(chroma.get("port", 8000)))
        store.check_server(self._client)
        self._embedder = get_embedder(model=emb["model"], batch_size=1, backend=emb.get("backend", "ollama"))
        self._collections: Dict[str, object] = {}

    def retrieve(self, corpus: str, query: str, k: int) -> List[Dict]:
        if corpus not in self._collections:
            self._collections[corpus] = self._client.get_collection(name=collection_name(self._s, corpus))
        emb = self._embedder.embed_query(query)
        return self._store.query_collection(self._collections[corpus], emb, top_k=k)["chunks"]


def short_id(paper_id: str) -> str:
    """"Paper56_10_1007_..." -> "Paper56" (the ID cited in answers)."""
    return paper_id.split("_")[0]


def format_generator_context(chunks: List[Dict]) -> str:
    """Numbered evidence blocks plus a source list, as given to the RAG generator."""
    from evaluation.retrieval_rag.retrieval.chunker import detect_section

    seen: Dict[str, str] = {}
    for c in chunks:
        seen.setdefault(short_id(c["paper_id"]), c["paper_title"])
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[{i}] [{short_id(c['paper_id'])}] \"{c['paper_title']}\"\n"
            f"    internal chunk: {c.get('chunk_index', '?')}  |  section: {detect_section(c['chunk_text'])}"
            f"  |  score: {c['similarity_score']:.3f}\n\n"
            f"{c['chunk_text']}"
        )
    sources = "\n".join(f"  [Source {i + 1}] [{sid}] {title}" for i, (sid, title) in enumerate(seen.items()))
    return "\n\n---\n\n".join(parts) + f"\n\n---\n\nSources:\n{sources}"


def format_reference_context(chunks: List[Dict]) -> str:
    """Numbered evidence blocks, as given to the reference drafter and the auditor."""
    return "\n\n---\n\n".join(
        f"[{i}] [{short_id(c.get('paper_id', ''))}] \"{c.get('paper_title', 'Unknown')}\" "
        f"(similarity: {c.get('similarity_score', 0.0):.3f})\n{c.get('chunk_text', '')}"
        for i, c in enumerate(chunks, 1)
    )


def generator_prompt(s: Settings, query: str, context: str) -> str:
    g = s.get("retrieval_rag.generator")
    template = s.resolve(g["prompt"]).read_text(encoding="utf-8")
    return template.format(role=g["role"].strip(), key_quantities=g["key_quantities"].strip(),
                           query=query, context=context)

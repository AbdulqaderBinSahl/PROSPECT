"""Step 2 — Chunk, embed and index every corpus in its own Chroma collection.

Chunks are ``chunking.chunk_size`` tokens with ``chunking.chunk_overlap``
tokens of overlap (tiktoken cl100k_base, recursive split on paragraphs,
lines, sentences, words). Embeddings come from ``embedding.model`` (Ollama by
default); collections use cosine distance. Papers already in a collection are
skipped, so an interrupted run resumes.

Then writes ``retrieval_rag/build_manifest.json``: all build settings, a
hash of the generator prompt, and paper/chunk counts per collection. Treat
the index as fixed from here on.

Needs a running Chroma server:  chroma run --path runs/chroma --port 8000

    python evaluation/retrieval_rag/index_corpora.py --settings <file>
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.retrieval_rag.rag import (  # noqa: E402
    collection_name,
    corpora_names,
    corpus_file,
    rr_path,
)
from evaluation.shared.settings import REPO_ROOT, add_settings_arg, load_settings, write_json  # noqa: E402


def index_corpus(s, corpus: str, client, store) -> dict:
    from tqdm import tqdm

    from evaluation.retrieval_rag.retrieval.chunker import chunk_paper
    from evaluation.retrieval_rag.retrieval.embedder import get_embedder
    from evaluation.retrieval_rag.retrieval.paper_loader import load_papers

    ch, emb = s.get("retrieval_rag.chunking"), s.get("retrieval_rag.embedding")
    papers = load_papers(str(corpus_file(s, corpus)))
    collection = store.get_or_create_collection(client, collection_name(s, corpus))
    done = store.get_indexed_paper_ids(collection)
    pending = {pid: p for pid, p in papers.items() if pid not in done}
    if not pending:
        return {"corpus": corpus, "status": "already indexed"}
    embedder = get_embedder(model=emb["model"], batch_size=int(emb.get("batch_size", 64)),
                            backend=emb.get("backend", "ollama"))
    failed = []
    for pid, paper in tqdm(pending.items(), desc=corpus, unit="paper"):
        try:
            chunks = chunk_paper(paper_id=pid, title=paper.title, full_text=paper.full_text,
                                 corpus_condition=corpus, chunk_size=int(ch["chunk_size"]),
                                 chunk_overlap=int(ch["chunk_overlap"]))
            if not chunks:
                continue
            store.add_chunks(
                collection, [c.chunk_id for c in chunks], embedder.embed_texts([c.text for c in chunks]),
                [c.text for c in chunks],
                [{"paper_id": c.paper_id, "title": c.title, "chunk_index": c.chunk_index,
                  "source_section": c.source_section, "corpus_condition": c.corpus_condition,
                  "original_text": c.original_text} for c in chunks],
            )
        except Exception as exc:  # noqa: BLE001 — report and continue
            tqdm.write(f"  [{corpus}] failed on {pid}: {exc}")
            failed.append(pid)
    return {"corpus": corpus, "status": "done", "failed": failed}


def manifest(s, client) -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=str(REPO_ROOT), check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    prompt = s.resolve(s.get("retrieval_rag.generator.prompt")).read_text(encoding="utf-8")
    out = {
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "chunking": s.get("retrieval_rag.chunking"),
        "embedding": s.get("retrieval_rag.embedding"),
        "retrieval_top_k": s.get("retrieval_rag.retrieval.top_k"),
        "generator": s.get("retrieval_rag.generator.model"),
        "generator_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "chroma_space": "cosine",
        "collections": {},
    }
    for corpus in corpora_names(s):
        name = collection_name(s, corpus)
        try:
            col = client.get_collection(name=name)
            meta = col.get(limit=100_000, include=["metadatas"])["metadatas"]
            out["collections"][corpus] = {
                "collection": name, "chunk_count": col.count(),
                "paper_count": len({m.get("paper_id") for m in meta}),
                "section_distribution": dict(Counter(m.get("source_section", "other") for m in meta)),
            }
        except Exception as exc:  # noqa: BLE001
            out["collections"][corpus] = {"collection": name, "error": str(exc)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_settings_arg(parser)
    s = load_settings(parser.parse_args().settings)

    from evaluation.retrieval_rag.retrieval import store

    chroma = s.get("retrieval_rag.chroma")
    client = store.get_client(host=chroma.get("host", "localhost"), port=int(chroma.get("port", 8000)))
    store.check_server(client)
    for corpus in corpora_names(s):
        if not corpus_file(s, corpus).exists():
            print(f"  {corpus}: no corpus file — run build_corpora.py first")
            continue
        print("  ", index_corpus(s, corpus, client, store))

    out = rr_path(s, "build_manifest.json")
    write_json(out, manifest(s, client))
    print(f"Build manifest: {out}")
    print("Next: python evaluation/retrieval_rag/build_queries.py")


if __name__ == "__main__":
    main()

"""Combine keyword and semantic rankings using reciprocal rank fusion."""

import argparse
from pathlib import Path

from filinglens.search import KeywordSearch, load_chunks
from filinglens.semantic import SemanticSearch


def fuse_rankings(rankings, top_k=5, constant=60):
    """Combine ranks without treating different score scales as comparable."""
    if top_k < 1 or constant < 1:
        raise ValueError("top_k and constant must be positive.")

    scores = {}
    passages = {}

    for ranking in rankings:
        seen = set()
        for rank, passage in enumerate(ranking, start=1):
            passage_id = passage["id"]
            if passage_id in seen:
                continue
            seen.add(passage_id)
            passages[passage_id] = passage
            scores[passage_id] = scores.get(passage_id, 0.0) + 1.0 / (constant + rank)

    ordered = sorted(scores, key=lambda key: (-scores[key], key))
    return [{**passages[key], "score": scores[key]} for key in ordered[:top_k]]


class HybridSearch:
    def __init__(self, chunks):
        self.semantic = SemanticSearch(chunks)
        # Both retrieval methods use identical passages and IDs.
        self.keyword = KeywordSearch(self.semantic.chunks)

    def search(self, query, top_k=5):
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        if not query.strip():
            return []

        candidate_count = max(30, top_k)
        keyword_results = self.keyword.search(query, candidate_count)
        semantic_results = self.semantic.search(query, candidate_count)
        return fuse_rankings([keyword_results, semantic_results], top_k=top_k)


def print_results(label, results):
    print(f"\n=== {label} ===")
    for rank, result in enumerate(results, start=1):
        print(
            f"\n[{rank}] {result['source']} | PDF page {result['page']}"
            f" | Score {result['score']:.4f}"
        )
        print(result["text"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be positive.")

    chunks = load_chunks(Path("data/processed"))
    if not chunks:
        parser.error("Ingest a PDF first.")

    engine = HybridSearch(chunks)
    print(f"\nQuestion: {args.query}")
    print_results("BM25", engine.keyword.search(args.query, args.top_k))
    print_results("Semantic", engine.semantic.search(args.query, args.top_k))
    print_results("Hybrid", engine.search(args.query, args.top_k))


if __name__ == "__main__":
    main()

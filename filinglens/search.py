"""Search extracted PDF passages with a BM25 keyword baseline."""

import argparse
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Apply the same normalization to documents and queries."""
    return re.findall(r"[a-z0-9]+", text.lower())


class KeywordSearch:
    def __init__(self, chunks: list[dict]):
        self.chunks = []
        self.tokens = []

        for chunk in chunks:
            tokens = tokenize(chunk["text"])
            if tokens:
                self.chunks.append(chunk)
                self.tokens.append(tokens)

        if not self.chunks:
            raise ValueError("No searchable passages. Ingest a PDF first.")

        self.index = BM25Okapi(self.tokens)
        self.token_sets = [set(tokens) for tokens in self.tokens]

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.index.get_scores(query_tokens)
        query_set = set(query_tokens)

        # Exclude passages with no matching words, even if all scores are zero.
        candidates = [
            i
            for i, tokens in enumerate(self.token_sets)
            if query_set.intersection(tokens)
        ]
        ranked = sorted(candidates, key=lambda i: (-float(scores[i]), i))

        return [{**self.chunks[i], "score": float(scores[i])} for i in ranked[:top_k]]


def load_chunks(directory: Path) -> list[dict]:
    chunks = []
    seen = set()

    for path in sorted(directory.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        for chunk in document["chunks"]:
            if chunk["id"] not in seen:
                seen.add(chunk["id"])
                chunks.append(chunk)

    return chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")

    chunks = load_chunks(args.index_dir)
    if not chunks:
        parser.error("No indexed passages. Run ingestion first.")

    engine = KeywordSearch(chunks)
    results = engine.search(args.query, top_k=args.top_k)

    print(f"\nIndexed passages: {len(engine.chunks)}")
    print(f"Query: {args.query}")

    if not results:
        print("No passages matched the query words.")

    for rank, result in enumerate(results, start=1):
        print(
            f"\n[{rank}] {result['source']} | PDF page {result['page']}"
            f" | BM25 score {result['score']:.3f}"
        )
        print(result["text"])
        print("-" * 70)


if __name__ == "__main__":
    main()

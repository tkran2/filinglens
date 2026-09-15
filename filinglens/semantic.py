"""Local semantic search with cached, normalized passage embeddings."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from filinglens.search import load_chunks

MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"


class SemanticSearch:
    def __init__(self, chunks: list[dict]):
        if not chunks:
            raise ValueError("No passages found. Ingest a PDF first.")

        self.model = SentenceTransformer(MODEL_NAME, device="cpu")

        # Keep passages near the model's training length.
        # Split by words, checking actual tokenizer length each time.
        token_limit = min(240, self.model.max_seq_length)

        def split_to_fit(passage):
            tokens = self.model.tokenizer(passage, truncation=False, verbose=False)[
                "input_ids"
            ]
            if len(tokens) <= token_limit:
                return [passage]

            words = passage.split()
            if len(words) < 2:
                raise ValueError("A single word exceeds the token limit.")

            middle = len(words) // 2
            return split_to_fit(" ".join(words[:middle])) + split_to_fit(
                " ".join(words[middle:])
            )

        fitted_chunks = []
        for chunk in chunks:
            parts = split_to_fit(chunk["text"])
            for position, part in enumerate(parts):
                fitted_chunks.append(
                    {
                        **chunk,
                        "id": f"{chunk['id']}:s{position}",
                        "parent_id": chunk["id"],
                        "text": part,
                    }
                )

        print(
            f"Prepared {len(fitted_chunks)} embedding passages "
            f"from {len(chunks)} original passages."
        )
        chunks = fitted_chunks
        self.chunks = chunks

        # Cache identity includes the text, ordering, model and token limit.
        identity = {
            "model": MODEL_NAME,
            "max_seq_length": self.model.max_seq_length,
            "passages": [(c["id"], c["text"]) for c in chunks],
        }
        fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()

        cache_dir = Path("data/processed/embeddings")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{fingerprint}.npy"

        if cache_path.exists():
            print("Loading cached passage embeddings...")
            self.vectors = np.load(cache_path, allow_pickle=False)
        else:
            texts = [chunk["text"] for chunk in chunks]

            # Report truncation explicitly rather than silently hiding it.
            tokenized = self.model.tokenizer(texts, truncation=False, padding=False)
            oversized = sum(
                len(ids) > self.model.max_seq_length for ids in tokenized["input_ids"]
            )
            print(f"Passages exceeding model token limit: {oversized}")

            print(f"Embedding {len(texts)} passages...")
            self.vectors = self.model.encode(
                texts,
                batch_size=16,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            )
            temporary = cache_path.with_suffix(".tmp")
            with temporary.open("wb") as file:
                np.save(file, self.vectors, allow_pickle=False)
            temporary.replace(cache_path)

        if self.vectors.shape[0] != len(chunks):
            raise ValueError("Cached embeddings do not match passage count.")

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if not query.strip():
            return []

        query_vector = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # For unit-length vectors, dot product equals cosine similarity.
        scores = self.vectors @ query_vector
        ranked = np.argsort(-scores, kind="stable")[:top_k]

        return [{**self.chunks[int(i)], "score": float(scores[i])} for i in ranked]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1.")

    chunks = load_chunks(Path("data/processed"))
    if not chunks:
        parser.error("No passages found. Ingest a PDF first.")

    engine = SemanticSearch(chunks)
    results = engine.search(args.query, args.top_k)

    print(f"\nQuery: {args.query}")
    for rank, result in enumerate(results, start=1):
        print(
            f"\n[{rank}] {result['source']} | PDF page {result['page']}"
            f" | Cosine similarity {result['score']:.3f}"
        )
        print(result["text"])
        print("-" * 70)


if __name__ == "__main__":
    main()

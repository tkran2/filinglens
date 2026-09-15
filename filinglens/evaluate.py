"""Development evaluation using approximate page-level relevance labels."""

import hashlib
import json
from pathlib import Path

from filinglens.hybrid import HybridSearch
from filinglens.search import load_chunks


def first_relevant_rank(results, case):
    for rank, result in enumerate(results, start=1):
        if (
            result["source"] == case["source"]
            and result["page"] in case["relevant_pages"]
        ):
            return rank
    return None


def main():
    dataset_path = Path("evaluation/apple_dev.json")
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))
    chunks = load_chunks(Path("data/processed"))

    if not cases or not chunks:
        raise ValueError("Evaluation cases and indexed passages are required.")

    available = {(c["source"], c["page"]) for c in chunks}
    for case in cases:
        for page in case["relevant_pages"]:
            if (case["source"], page) not in available:
                raise ValueError(f"Missing labeled page for {case['id']}.")

    engine = HybridSearch(chunks)
    methods = {
        "BM25": engine.keyword.search,
        "Semantic": engine.semantic.search,
        "Hybrid": engine.search,
    }
    report = {
        "label_type": "approximate page-level development labels",
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "document_ids": sorted({c["document_id"] for c in chunks}),
        "passage_count": len(engine.semantic.chunks),
        "results": {},
    }

    print("\nEight-question DEVELOPMENT benchmark")
    print("All methods search the same passages.")
    print("Page matches are approximate, not verified answer correctness.\n")

    for name, search in methods.items():
        records = []
        hits_at_3 = 0
        hits_at_5 = 0
        reciprocal_rank_sum = 0.0

        for case in cases:
            results = search(case["question"], top_k=5)
            rank = first_relevant_rank(results, case)

            if rank is not None:
                hits_at_5 += 1
                hits_at_3 += int(rank <= 3)
                reciprocal_rank_sum += 1.0 / rank

            records.append(
                {
                    "case": case,
                    "first_relevant_rank": rank,
                    "retrieved": [
                        {
                            "id": result["id"],
                            "source": result["source"],
                            "page": result["page"],
                            "score": result["score"],
                            "text": result["text"],
                        }
                        for result in results
                    ],
                }
            )

        count = len(cases)
        metrics = {
            "page_hit_at_3": hits_at_3 / count,
            "page_hit_at_5": hits_at_5 / count,
            "page_mrr_at_5": reciprocal_rank_sum / count,
        }
        report["results"][name] = {
            "metrics": metrics,
            "queries": records,
        }

        print(
            f"{name:10} | Page Hit@3: {hits_at_3}/{count}"
            f" | Page Hit@5: {hits_at_5}/{count}"
            f" | Page MRR@5: {metrics['page_mrr_at_5']:.3f}"
        )
        for record in records:
            rank = record["first_relevant_rank"]
            print(
                f"  {record['case']['id']:24} "
                f"first labeled page: {rank if rank else 'not in top 5'}"
            )
        print()

    destination = Path("data/processed/evaluation")
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "apple_dev_results.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Detailed results saved to {output}")


if __name__ == "__main__":
    main()

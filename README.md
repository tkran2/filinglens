# FilingLens

[![Tests](https://github.com/tkran2/filinglens/actions/workflows/tests.yml/badge.svg)](https://github.com/tkran2/filinglens/actions/workflows/tests.yml)

A financial-document research prototype that answers questions about
Apple's 2024 annual report and displays the retrieved evidence alongside
page-level citations.

## Features

- PDF ingestion with document hashes and page provenance
- BM25 keyword search, local semantic search, and reciprocal rank fusion
- Token-aware splitting for embedding inputs
- Cached passage embeddings
- Structured answer generation with citation-ID validation
- Unsupported-question abstention
- Streamlit interface with source links and downloadable evidence
- Retrieval evaluation and automated tests

## Local setup

Requires uv, Python 3.12, and an OpenAI API key for answer generation.
API usage is separately billed.

    uv sync --locked
    cp .env.example .env

Set OPENAI_API_KEY in .env. Never commit this file.

Download the starting document:

    mkdir -p data/raw
    curl -fL https://www.annualreports.com/HostedData/AnnualReportArchive/a/NASDAQ_AAPL_2024.pdf -o data/raw/apple-2024.pdf
    uv run python -m filinglens.ingest data/raw/apple-2024.pdf

Start the interface:

    uv run streamlit run app.py --server.address 127.0.0.1

The embedding model downloads on first use.

## Tests and evaluation

    uv run python -m pytest -q
    uv run python -m filinglens.evaluate

Initial results on eight development questions:

| Retrieval method | Page Hit@3 | Page Hit@5 | Page MRR@5 |
|---|---:|---:|---:|
| BM25 | 8/8 | 8/8 | 1.000 |
| Semantic | 7/8 | 8/8 | 0.838 |
| Hybrid | 7/8 | 8/8 | 0.775 |

All three methods searched the same 616 passages. BM25 is the initial
answer-generation default based on these development results.

These are approximate page-level retrieval measurements on a small,
manually constructed development set. They are not held-out results
or measurements of answer accuracy.

## Known limitations

- Evaluation currently covers one company and one annual report.
- Valid citation IDs do not guarantee factual support.
- Generated answers can still add unsupported timing or causal claims.
- Abstention is prompted behavior, not a calibrated confidence threshold.
- PDF extraction and passage boundaries may lose table or sentence context.
- Scanned documents require OCR, which is not implemented.
- Citation links depend on the browser's PDF viewer.
- The interface is currently intended for local use.

## Data source

Apple Inc., 2024 Form 10-K:
https://www.annualreports.com/HostedData/AnnualReportArchive/a/NASDAQ_AAPL_2024.pdf

Downloaded documents, generated indexes, and API keys are excluded from Git.

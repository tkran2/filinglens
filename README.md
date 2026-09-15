# FilingLens

**[Open the live demo](https://filinglens-m7yp8jnffuwwzny7jtpmsl.streamlit.app/)**

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

The hosted serving path uses committed passages. Retrieval experiments download the embedding model on first use.

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

## Answer verification

The generator drafts structured claims with citation IDs. Code checks
that each claim has citations and that those IDs exist. A second model
pass reviews each claim against only its cited passages and removes
claims it judges unsupported.

In one manually inspected deployed example, the verifier retained the
holiday-demand explanation and removed an unsupported product-launch
timing claim: two drafted claims became one retained claim.

This is an observed example, not a measured hallucination-reduction rate.
Both passes use the same model family and can share errors. Citation
validation and model review do not guarantee factual correctness.

## Hosted architecture

The deployed interface uses prebuilt passages in
assets/apple-2024-passages.json and BM25 retrieval. It does not load
PyTorch or the embedding model. Semantic retrieval and hybrid search
remain available locally for experiments and evaluation.

The answer flow is:

1. Retrieve five passages.
2. Generate structured claims with citations.
3. Validate citation IDs.
4. Review claims against their cited evidence.
5. Display retained claims alongside source passages.

Supported answers normally require two API requests. Token counts
include both passes, and reported generation time includes verification.

## Public demo limits

The interface admits up to five questions per minute and 100 per rolling
24 hours, shared across visitors within one server process. Failed API
attempts also consume an admission slot.

Limits are thread-safe but process-local. They reset on restart and are
not a durable billing cap or distributed rate limiter.

## Cloud deployment

Deploy deploy/streamlit_app.py on Streamlit Community Cloud with
Python 3.12. Its adjacent requirements.txt contains the serving
dependencies. Configure OPENAI_API_KEY in private app secrets.

For a local launch with the committed passages:

    uv sync --locked
    cp .env.example .env

Set the key in .env, then run:

    uv run streamlit run app.py --server.address 127.0.0.1

PDF ingestion and embedding-model downloads are only needed when
rebuilding passages or running retrieval experiments.

## Validation scope

The automated suite currently contains 22 tests covering passage
splitting, keyword retrieval, rank fusion, citation validation, and
request admission limits.

These tests do not establish generated-answer accuracy or the verifier's
semantic correctness. The retrieval benchmark contains eight development
questions with approximate page labels, not an independent held-out set.

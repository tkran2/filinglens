"""Extract PDF passages while preserving source-page citations."""

import argparse
import hashlib
import json
from pathlib import Path

from pypdf import PdfReader


def split_text(text: str, size: int = 250, overlap: int = 40) -> list[str]:
    """Split into overlapping word windows without dropping the final words."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size.")

    words = text.split()
    passages = []
    start = 0

    while start < len(words):
        end = min(start + size, len(words))
        passages.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap

    return passages


def ingest_pdf(path: Path, output_dir: Path) -> dict:
    """Write one JSON index per PDF, replacing it safely on repeated runs."""
    document_id = hashlib.sha256(path.read_bytes()).hexdigest()
    reader = PdfReader(path)
    chunks = []
    empty_pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        passages = split_text(text)

        if not passages:
            empty_pages.append(page_number)

        for position, passage in enumerate(passages):
            chunks.append(
                {
                    "id": f"{document_id}:p{page_number}:c{position}",
                    "document_id": document_id,
                    "source": path.name,
                    "page": page_number,
                    "text": passage,
                }
            )

    if not chunks:
        raise ValueError(
            f"No text extracted from {path.name}. "
            "Use a PDF with selectable text; scans need OCR."
        )

    result = {
        "schema_version": 1,
        "document_id": document_id,
        "source": path.name,
        "page_count": len(reader.pages),
        "empty_pages": empty_pages,
        "chunking": {"size_words": 250, "overlap_words": 40},
        "chunks": chunks,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{document_id}.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    if not args.pdf.is_file():
        parser.error(f"File not found: {args.pdf}")

    result = ingest_pdf(args.pdf, args.output_dir)
    print(f"Document: {result['source']}")
    print(f"PDF pages: {result['page_count']}")
    print(f"Searchable passages: {len(result['chunks'])}")
    print(f"Pages without extracted text: {result['empty_pages']}")
    print("\nFirst passage:")
    print(result["chunks"][0]["text"][:700])


if __name__ == "__main__":
    main()

from filinglens.search import KeywordSearch


def make_engine():
    return KeywordSearch(
        [
            {
                "id": "a",
                "source": "example.pdf",
                "page": 2,
                "text": "Cybersecurity incidents may disrupt operations.",
            },
            {
                "id": "b",
                "source": "example.pdf",
                "page": 3,
                "text": "Dividends are paid to shareholders.",
            },
            {
                "id": "c",
                "source": "example.pdf",
                "page": 4,
                "text": "Manufacturing depends on suppliers.",
            },
        ]
    )


def test_search_preserves_source_citation():
    results = make_engine().search("CYBERSECURITY?")
    assert len(results) == 1
    assert results[0]["id"] == "a"
    assert results[0]["source"] == "example.pdf"
    assert results[0]["page"] == 2


def test_unmatched_query_returns_no_results():
    assert make_engine().search("penguins") == []


def test_empty_query_returns_no_results():
    assert make_engine().search("?!") == []

from itertools import pairwise

import pytest

from filinglens.ingest import split_text


def test_empty_text():
    assert split_text(" \n\t ") == []


def test_short_text():
    assert split_text("Annual revenue increased.") == ["Annual revenue increased."]


def test_overlap_and_final_words_are_preserved():
    words = [f"word{i}" for i in range(623)]
    chunks = [chunk.split() for chunk in split_text(" ".join(words))]

    recovered = chunks[0].copy()
    for previous, current in pairwise(chunks):
        assert previous[-40:] == current[:40]
        recovered.extend(current[40:])

    assert recovered == words
    assert all(len(chunk) <= 250 for chunk in chunks)


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, -1), (10, 10), (10, 11)])
def test_invalid_settings(size, overlap):
    with pytest.raises(ValueError):
        split_text("Some text", size=size, overlap=overlap)

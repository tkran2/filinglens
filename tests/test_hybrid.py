import pytest

from filinglens.hybrid import fuse_rankings


def test_agreement_between_methods_boosts_passage():
    first = [{"id": "a"}, {"id": "shared"}]
    second = [{"id": "b"}, {"id": "shared"}]
    results = fuse_rankings([first, second])
    assert results[0]["id"] == "shared"
    assert len(results) == 3


def test_duplicate_in_one_ranking_is_not_counted_twice():
    results = fuse_rankings([[{"id": "a"}, {"id": "a"}]])
    assert len(results) == 1
    assert results[0]["score"] == pytest.approx(1 / 61)


def test_empty_rankings():
    assert fuse_rankings([[], []]) == []

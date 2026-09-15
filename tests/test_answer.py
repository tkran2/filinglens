import pytest

from filinglens.answer import Answer, Claim, validate_answer


def test_valid_citation():
    answer = Answer(
        supported=True,
        claims=[Claim(text="A supported statement.", citations=[1])],
    )
    validate_answer(answer, 2)


@pytest.mark.parametrize("citations", [[], [0], [3]])
def test_missing_or_unknown_citation_is_rejected(citations):
    answer = Answer(
        supported=True,
        claims=[Claim(text="A statement.", citations=citations)],
    )
    with pytest.raises(ValueError):
        validate_answer(answer, 2)


def test_abstention():
    validate_answer(Answer(supported=False, claims=[]), 2)


def test_abstention_cannot_include_claims():
    answer = Answer(
        supported=False,
        claims=[Claim(text="A statement.", citations=[1])],
    )
    with pytest.raises(ValueError):
        validate_answer(answer, 2)

import pytest

from lm_retry import LMOutputError, _looks_empty, call_with_retry


class _Pred:
    def __init__(self, **fields):
        self.__dict__.update(fields)


def test_looks_empty_treats_none_blank_and_empty_containers_as_empty():
    assert _looks_empty(None)
    assert _looks_empty("")
    assert _looks_empty("   \n")
    assert _looks_empty([])
    assert _looks_empty({})
    assert not _looks_empty("x")
    assert not _looks_empty(["x"])
    assert not _looks_empty(False)  # a valid bool output is not "empty"
    assert not _looks_empty(0)


def test_returns_first_good_result_without_retrying():
    calls = []

    def program(**kwargs):
        calls.append(kwargs)
        return _Pred(prose="hello")

    result = call_with_retry(program, fields="prose", label="x", a=1)
    assert result.prose == "hello"
    assert calls == [{"a": 1}]


def test_retries_on_empty_field_then_succeeds():
    attempts = []

    def program(**_):
        attempts.append(1)
        return _Pred(prose="" if len(attempts) < 2 else "good")

    result = call_with_retry(
        program, fields="prose", label="x", backoff_seconds=0
    )
    assert result.prose == "good"
    assert len(attempts) == 2


def test_retries_on_exception_then_succeeds():
    attempts = []

    def program(**_):
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("boom")
        return _Pred(prose="recovered")

    result = call_with_retry(
        program, fields="prose", label="x", backoff_seconds=0
    )
    assert result.prose == "recovered"
    assert len(attempts) == 3


def test_raises_lm_output_error_after_exhausting_attempts_on_empty():
    def program(**_):
        return _Pred(prose=None)

    with pytest.raises(LMOutputError):
        call_with_retry(
            program, fields="prose", label="draft", attempts=2, backoff_seconds=0
        )


def test_raises_lm_output_error_after_exhausting_attempts_on_exception():
    def program(**_):
        raise ValueError("nope")

    with pytest.raises(LMOutputError):
        call_with_retry(program, label="step", attempts=2, backoff_seconds=0)


def test_no_fields_means_retry_on_exceptions_only():
    # An empty/None field is fine when no fields are required.
    result = call_with_retry(lambda **_: _Pred(prose=None), label="x")
    assert result.prose is None

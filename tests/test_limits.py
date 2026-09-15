import pytest

from filinglens.limits import DemoLimiter


def test_minute_limit_and_expiry():
    now = [0.0]
    limiter = DemoLimiter(per_minute=1, clock=lambda: now[0])
    limiter.acquire()
    with pytest.raises(ValueError, match="busy"):
        limiter.acquire()
    now[0] = 60.0
    limiter.acquire()


def test_day_limit_and_expiry():
    now = [0.0]
    limiter = DemoLimiter(per_minute=10, per_day=1, clock=lambda: now[0])
    limiter.acquire()
    now[0] = 120.0
    with pytest.raises(ValueError, match="daily"):
        limiter.acquire()
    now[0] = 86400.0
    limiter.acquire()


def test_rejected_request_does_not_consume_capacity():
    limiter = DemoLimiter(per_minute=1)
    limiter.acquire()
    with pytest.raises(ValueError):
        limiter.acquire()
    assert len(limiter.requests) == 1

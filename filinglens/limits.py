"""Thread-safe, process-local admission limits for the public demo."""

import time
from collections import deque
from threading import Lock


class DemoLimiter:
    def __init__(self, per_minute=5, per_day=100, clock=time.monotonic):
        self.per_minute = per_minute
        self.per_day = per_day
        self.clock = clock
        self.requests = deque()
        self.lock = Lock()

    def acquire(self):
        with self.lock:
            now = self.clock()
            while self.requests and now - self.requests[0] >= 86400:
                self.requests.popleft()

            if len(self.requests) >= self.per_day:
                raise ValueError(
                    "This demo has reached its daily question limit. "
                    "Please try again later."
                )

            recent = sum(now - timestamp < 60 for timestamp in self.requests)
            if recent >= self.per_minute:
                raise ValueError(
                    "The demo is busy. Please wait a minute before trying again."
                )

            # Reserve capacity before the API request, including failed attempts.
            self.requests.append(now)

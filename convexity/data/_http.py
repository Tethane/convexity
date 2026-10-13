"""Bounded, read-only JSON HTTP requests with transient-error retries."""
import json
import math
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .core import DataError


def positive(value: float, name: str, *, zero: bool = False) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (value == 0 and not zero):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}.")


class JSONClient:
    def __init__(self, base_url: str, *, timeout: float = 30, retries: int = 2,
                 request_interval: float = 0.2):
        positive(timeout, "timeout")
        positive(request_interval, "request_interval", zero=True)
        if type(retries) is not int or not 0 <= retries <= 10:
            raise ValueError("retries must be an integer between 0 and 10.")
        if not base_url.startswith("https://"):
            raise ValueError("base_url must use HTTPS.")
        self.base_url = base_url.rstrip("/")
        self.timeout, self.retries, self.request_interval = timeout, retries, request_interval

    def get(self, path: str, params: dict):
        request = Request(f"{self.base_url}{path}?{urlencode(params)}",
                          headers={"User-Agent": "convexity/0.1", "Accept": "application/json"})
        for attempt in range(self.retries + 1):
            time.sleep(self.request_interval)
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except HTTPError as exc:
                if (exc.code == 429 or 500 <= exc.code < 600) and attempt < self.retries:
                    delay = exc.headers.get("Retry-After", "")
                    # Do not retry sooner than a long provider-requested cooldown.
                    if delay.isdigit() and int(delay) > 60:
                        raise DataError(f"HTTP {exc.code}; retry after {delay} seconds.") from exc
                    time.sleep(int(delay) if delay.isdigit() else 2 ** attempt)
                    continue
                detail = exc.read(1024).decode(errors="replace")
                raise DataError(f"HTTP {exc.code} from {self.base_url}: {detail}") from exc
            except (URLError, TimeoutError) as exc:
                if attempt == self.retries:
                    raise DataError(f"Request to {self.base_url} failed: {exc}") from exc
                time.sleep(2 ** attempt)
            except (ValueError, UnicodeError) as exc:
                raise DataError(f"Invalid JSON from {self.base_url}.") from exc

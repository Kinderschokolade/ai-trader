"""Polite HTTP client for sec.gov / data.sec.gov (User-Agent + rate limit + retries)."""
from __future__ import annotations

import time

import requests


class SecClient:
    def __init__(self, user_agent: str, max_rps: float = 8, retries: int = 5, timeout: int = 120):
        if not user_agent or "@" not in user_agent:
            raise ValueError(
                "Set [sec] user_agent in config/global.toml, e.g. 'Your Name ai-trader your@email.org'. "
                "The SEC blocks requests without a contact e-mail."
            )
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.min_interval = 1.0 / max_rps
        self.retries = retries
        self.timeout = timeout
        self._last = 0.0

    def get(self, url: str, **kw) -> requests.Response:
        resp = None
        for attempt in range(self.retries):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                resp = self.session.get(url, timeout=self.timeout, **kw)
            except requests.RequestException as exc:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)
                continue
            if resp.status_code in (200, 404):
                return resp
            if resp.status_code in (403, 429, 500, 502, 503, 504):
                # 403 from the SEC usually means "too fast" or a missing User-Agent.
                time.sleep(2 ** attempt * 2)
                continue
            resp.raise_for_status()
        assert resp is not None
        if resp.status_code == 403:
            raise RuntimeError(
                f"SEC returned 403 for {url}. Check [sec] user_agent and lower max_requests_per_second."
            )
        resp.raise_for_status()
        return resp


def client_from_settings(settings) -> SecClient:
    return SecClient(
        settings.get("sec.user_agent", ""),
        max_rps=float(settings.get("sec.max_requests_per_second", 8)),
    )

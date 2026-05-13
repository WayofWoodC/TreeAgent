from __future__ import annotations

import time
from dataclasses import dataclass

import requests

from pe_pipeline.exceptions import FetchError


@dataclass(slots=True)
class HttpClient:
    timeout: int = 60
    retries: int = 3
    pause_seconds: float = 0.25

    def get_json(self, url: str, params: dict | None = None, headers: dict | None = None) -> object:
        last_error: Exception | None = None
        for _ in range(self.retries):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_error = exc
                time.sleep(self.pause_seconds)
        raise FetchError(f"Failed request: {url}") from last_error

    def get_text(self, url: str, params: dict | None = None, headers: dict | None = None) -> str:
        last_error: Exception | None = None
        for _ in range(self.retries):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                return response.text
            except Exception as exc:
                last_error = exc
                time.sleep(self.pause_seconds)
        raise FetchError(f"Failed request: {url}") from last_error

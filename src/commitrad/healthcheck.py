"""Container healthcheck: exit 0 if the dashboard or the API answers on localhost."""

from __future__ import annotations

import sys
import urllib.request

URLS = ("http://127.0.0.1:8501/_stcore/health", "http://127.0.0.1:8000/health")


def healthy(urls: tuple[str, ...] = URLS, timeout: float = 3.0) -> bool:
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                if resp.status == 200:
                    return True
        except OSError:
            continue
    return False


if __name__ == "__main__":
    sys.exit(0 if healthy() else 1)

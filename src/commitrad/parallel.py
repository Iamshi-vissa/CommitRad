"""Run a per-repo function over many repos in parallel, preserving input order."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TypeVar

T = TypeVar("T")


def map_repos(fn: Callable[[str], T], sources: list[str], workers: int) -> list[T]:
    """Apply ``fn`` to each source, in worker processes when ``workers > 1``.

    ``fn`` must be picklable (a module-level function or ``functools.partial`` of one).
    """
    if workers <= 1 or len(sources) <= 1:
        return [fn(s) for s in sources]
    results: dict[str, T] = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(sources))) as pool:
        futures = {pool.submit(fn, s): s for s in sources}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    return [results[s] for s in sources]

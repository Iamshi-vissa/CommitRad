"""Repo list parsing, slugging, and a local clone cache."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

_URL_RE = re.compile(r"^(?:https?://|git@)(?P<host>[^/:]+)[/:](?P<path>.+?)(?:\.git)?/?$")
_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def is_local(source: str) -> bool:
    """True when ``source`` is an existing directory rather than a remote URL."""
    return Path(source).expanduser().is_dir()


def repo_slug(source: str) -> str:
    """Stable, filesystem-safe name for a repo, e.g. ``jd/tenacity`` -> ``jd__tenacity``."""
    if is_local(source):
        return _SAFE_RE.sub("_", Path(source).expanduser().resolve().name)
    if source.startswith("file://"):
        return _SAFE_RE.sub("_", Path(source.removeprefix("file://")).name)
    match = _URL_RE.match(source.strip())
    if not match:
        raise ValueError(f"Not a repo URL or local directory: {source!r}")
    parts = [p for p in match.group("path").split("/") if p]
    return "__".join(_SAFE_RE.sub("_", p) for p in parts)


def read_repo_list(path: Path) -> list[str]:
    """Read one repo per line, ignoring blanks, ``#`` comments and duplicates."""
    seen: dict[str, None] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            seen.setdefault(line, None)
    return list(seen)


def ensure_clone(
    source: str,
    clone_dir: Path,
    timeout: float | None = None,
    git_config: list[str] | None = None,
) -> Path:
    """Return a local checkout for ``source``, cloning into ``clone_dir`` if needed.

    Local directories are used in place. Clones go to a temp dir first and are
    renamed into place, so an interrupted clone never looks complete.
    """
    if is_local(source):
        return Path(source).expanduser().resolve()
    target = clone_dir / repo_slug(source)
    if (target / ".git").is_dir():
        return target
    clone_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=clone_dir))
    try:
        subprocess.run(
            ["git", *(git_config or []), "clone", "--quiet", source, str(tmp)],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        tmp.rename(target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return target

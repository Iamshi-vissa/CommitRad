"""Pre-push risk check: score the commits you are about to push, with reasons.

``commitrad check`` mines the full local history (so author experience, file history and
past bug rates are right), scores only the commits in ``<upstream>..HEAD`` and prints them
riskiest first. ``commitrad hook install`` runs it from git's pre-push hook: warn-only by
default, or blocking above a risk percentile you choose.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from commitrad.explain import describe_value, top_reasons
from commitrad.features import FEATURE_COLUMNS, build_features
from commitrad.mining import mine_table
from commitrad.model import load_model
from commitrad.pipeline import score_commits
from commitrad.sentiment import get_scorer
from commitrad.szz import label_table

HOOK_MARKER = "# installed by commitrad"


class CheckError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckedReason:
    text: str
    shap: float


@dataclass(frozen=True)
class CheckedCommit:
    hash: str
    subject: str
    risk: float
    percentile: float
    reasons: list[CheckedReason]


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise CheckError(exc.stderr.strip() or f"git {' '.join(args)} failed") from exc


def resolve_base(repo: Path, base: str | None) -> str:
    """The ref to compare against: ``base``, else the upstream, else origin/main|master."""
    if base:
        _git(repo, "rev-parse", "--verify", base)
        return base
    for candidate in ("@{upstream}", "origin/HEAD", "origin/main", "origin/master"):
        try:
            _git(repo, "rev-parse", "--verify", "--quiet", candidate)
        except CheckError:
            continue
        return _git(repo, "rev-parse", "--abbrev-ref", candidate) or candidate
    raise CheckError("No upstream branch found; pass --base <ref> (e.g. origin/main)")


def unpushed(repo: Path, base: str) -> list[str]:
    return _git(repo, "rev-list", "--no-merges", f"{base}..HEAD").split()


def check_repo(
    repo: Path, model_dir: Path, base: str | None = None
) -> tuple[str, list[CheckedCommit]]:
    """Score the not-yet-pushed commits of ``repo``; riskiest first."""
    repo = Path(_git(repo, "rev-parse", "--show-toplevel"))
    ref = resolve_base(repo, base)
    targets = set(unpushed(repo, ref))
    if not targets:
        return ref, []
    model, meta = load_model(model_dir)
    labeled = label_table(mine_table(repo), repo).to_pandas()
    features = build_features(labeled, get_scorer("lexicon"), repo.name)
    rows = features[features["hash"].isin(targets)]
    if rows.empty:
        return ref, []
    scored = score_commits(rows, model, meta["risk_quantiles"])
    shap = model.shap_values(scored)
    out = []
    for i, r in scored.iterrows():
        reasons = top_reasons(shap.loc[i], scored.loc[i, FEATURE_COLUMNS])
        msg = (r["message"] or "").strip()
        out.append(
            CheckedCommit(
                hash=r["hash"],
                subject=msg.splitlines()[0] if msg else "(no message)",
                risk=float(r["risk"]),
                percentile=float(r["risk_pct"]),
                reasons=[
                    CheckedReason(describe_value(x["feature"], x["value"]), x["shap"])
                    for x in reasons
                ],
            )
        )
    return ref, sorted(out, key=lambda c: c.risk, reverse=True)


def format_report(ref: str, commits: list[CheckedCommit], fail_above: float | None) -> str:
    if not commits:
        return f"commitrad check · nothing to push beyond {ref}"
    lines = [f"commitrad check · {len(commits)} unpushed commit(s) vs {ref}", ""]
    lines.append(f"  {'RISK':>5}  {'PCTL':>5}  COMMIT   SUBJECT")
    for c in commits:
        flag = (
            "  <- above threshold" if fail_above is not None and c.percentile >= fail_above else ""
        )
        lines.append(
            f"  {c.risk:5.2f}  {c.percentile * 100:5.1f}  {c.hash[:7]}  {c.subject[:60]}{flag}"
        )
        for r in c.reasons:
            sign = "+" if r.shap > 0 else "-"
            lines.append(f"  {'':5}  {'':5}  {sign} {r.text} ({r.shap:+.2f})")
    lines += [
        "",
        "Risk percentile vs the model's reference commits. A prompt to review, not a verdict.",
    ]
    return "\n".join(lines)


def should_block(commits: list[CheckedCommit], fail_above: float | None) -> bool:
    return fail_above is not None and any(c.percentile >= fail_above for c in commits)


def hook_script(model_dir: Path, block_above: float | None) -> str:
    python = Path(sys.executable).as_posix()
    block = f" --fail-above {block_above}" if block_above is not None else ""
    return (
        "#!/bin/sh\n"
        f"{HOOK_MARKER}: scores unpushed commits before every push.\n"
        "# Remove this file (or run `commitrad hook uninstall`) to disable it.\n"
        f'"{python}" -m commitrad.cli check --repo "$(git rev-parse --show-toplevel)" '
        f'--models "{model_dir.resolve().as_posix()}"{block}\n'
    )


def install_hook(
    repo: Path, model_dir: Path, block_above: float | None = None, force: bool = False
) -> Path:
    hooks = Path(_git(repo, "rev-parse", "--git-path", "hooks"))
    hooks = (
        hooks if hooks.is_absolute() else Path(_git(repo, "rev-parse", "--show-toplevel")) / hooks
    )
    hook = hooks / "pre-push"
    if hook.exists() and HOOK_MARKER not in hook.read_text(errors="replace") and not force:
        raise CheckError(f"{hook} already exists and was not written by commitrad; use --force")
    hooks.mkdir(parents=True, exist_ok=True)
    hook.write_text(hook_script(model_dir, block_above))
    hook.chmod(0o755)
    return hook


def uninstall_hook(repo: Path) -> Path | None:
    hooks = Path(_git(repo, "rev-parse", "--git-path", "hooks"))
    hooks = (
        hooks if hooks.is_absolute() else Path(_git(repo, "rev-parse", "--show-toplevel")) / hooks
    )
    hook = hooks / "pre-push"
    if hook.exists() and HOOK_MARKER in hook.read_text(errors="replace"):
        hook.unlink()
        return hook
    return None

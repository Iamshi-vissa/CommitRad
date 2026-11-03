"""Design layer for the dashboard: tokens, CSS and pure HTML builders (unit-testable).

Every builder escapes the text it is given, so repo names, commit messages and author names
from arbitrary repositories can never inject markup.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from commitrad import __version__

LOGO = Path(__file__).with_name("assets") / "logo.svg"
PROJECT_URL = "https://github.com/Iamshi-vissa/CommitRad"

# Design tokens (dark only). Charts and SVG read the same values.
BG = "#0a0c10"
SURFACE = "#0f1318"
RAISED = "#151a21"
BORDER = "#222831"
BORDER_HI = "#3a4350"
TEXT = "#e6e9ee"
MUTED = "#8b95a3"
FAINT = "#5d6673"
ACCENT = "#58a6ff"
AUTHOR = "#4493f8"  # validated pair with REPO for colour-blind separation
REPO = "#db6d28"
UP = "#ff7b72"  # raises risk (diff "+")
DOWN = "#56d364"  # lowers risk (diff "-")
RAMP = ("#0c2d6b", "#0d419d", "#1158c7", "#1f6feb", "#388bfd", "#58a6ff", "#79c0ff", "#a5d6ff")
MINUS = "\u2212"  # typographic minus for diff lines
MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
SANS = "Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"

CSS = f"""
<style>
:root {{
  --bg:{BG}; --surface:{SURFACE}; --raised:{RAISED}; --border:{BORDER}; --border-hi:{BORDER_HI};
  --text:{TEXT}; --muted:{MUTED}; --faint:{FAINT}; --accent:{ACCENT}; --up:{UP}; --down:{DOWN};
  --mono:{MONO}; --sans:{SANS};
  --ease:cubic-bezier(.2,.7,.2,1); --fast:160ms; --slow:420ms;
}}
html {{ scroll-behavior:smooth; }}
[data-testid="stHeader"] {{ display:none; }}
[data-testid="stApp"] {{ background:var(--bg); }}
[data-testid="stMainBlockContainer"] {{ max-width:1180px; padding-top:1.1rem; padding-bottom:1rem; }}
::selection {{ background:rgba(88,166,255,.28); }}

/* --- streamlit widgets, restyled --- */
[data-testid="stWidgetLabel"] p {{ font-family:var(--mono); font-size:11.5px; color:var(--muted);
  text-transform:uppercase; letter-spacing:.08em; }}
[data-testid="stTextInputRootElement"], [data-baseweb="select"] > div {{
  background:var(--surface); border-color:var(--border);
  transition:border-color var(--fast) var(--ease), box-shadow var(--fast) var(--ease); }}
[data-testid="stTextInputRootElement"]:hover, [data-baseweb="select"] > div:hover {{
  border-color:var(--border-hi); }}
[data-testid="stTextInputRootElement"]:focus-within {{ border-color:var(--accent);
  box-shadow:0 0 0 3px rgba(88,166,255,.16); }}
[data-testid="stTextInputRootElement"] input {{ font-family:var(--mono); font-size:14px; }}
[data-baseweb="select"] {{ font-family:var(--mono); font-size:13.5px; }}
[data-testid="stBaseButton-secondary"] {{ font-family:var(--mono); font-size:12.5px;
  background:var(--surface); border-color:var(--border); color:var(--text);
  transition:border-color var(--fast) var(--ease), background var(--fast) var(--ease),
    transform var(--fast) var(--ease), color var(--fast) var(--ease); }}
[data-testid="stBaseButton-secondary"]:hover {{ border-color:var(--border-hi);
  background:var(--raised); color:#fff; transform:translateY(-1px); }}
[data-testid="stBaseButton-secondary"]:active {{ transform:translateY(0); }}
[data-testid="stBaseButton-secondary"] p {{ font-family:var(--mono); font-size:12.5px; }}
[data-testid="stExpander"] details {{ border-color:var(--border); background:var(--surface);
  transition:border-color var(--fast) var(--ease); }}
[data-testid="stExpander"] details:hover {{ border-color:var(--border-hi); }}
[data-testid="stExpander"] summary p {{ font-family:var(--mono); font-size:12.5px; }}
[data-testid="stSlider"] [role="slider"] {{ transition:box-shadow var(--fast) var(--ease); }}
[data-testid="stSlider"] [role="slider"]:hover {{ box-shadow:0 0 0 6px rgba(88,166,255,.18); }}
[class*="st-key-chart-"] {{ background:var(--surface); border-color:var(--border) !important;
  border-radius:10px !important; transition:border-color var(--fast) var(--ease); }}
[class*="st-key-chart-"]:hover {{ border-color:var(--border-hi) !important; }}
#vg-tooltip-element {{ background:var(--raised)!important; color:var(--text)!important;
  border:1px solid var(--border-hi)!important; border-radius:8px!important;
  font-family:var(--mono)!important; font-size:12px!important; padding:8px 10px!important;
  box-shadow:0 10px 30px rgba(0,0,0,.5)!important; }}
#vg-tooltip-element td.key {{ color:var(--muted)!important; }}
.st-key-examples {{ gap:8px; align-items:center; }}
.st-key-examples [data-testid="stBaseButton-secondary"] {{ min-height:30px; padding:2px 12px; }}

/* --- components --- */
.cr a {{ color:var(--accent); text-decoration:none; background-image:linear-gradient(currentColor,currentColor);
  background-size:0 1px; background-repeat:no-repeat; background-position:0 100%;
  transition:background-size var(--fast) var(--ease), color var(--fast) var(--ease); }}
.cr a:hover {{ background-size:100% 1px; color:#8cc4ff; }}
.cr-nav {{ display:flex; align-items:center; gap:16px; padding:6px 0 18px;
  border-bottom:1px solid var(--border); }}
.cr-nav img {{ width:68px; height:68px; transition:transform var(--slow) var(--ease); }}
.cr-nav img:hover {{ transform:rotate(-8deg) scale(1.06); }}
.cr-word {{ font-family:var(--mono); font-weight:700; font-size:60px; line-height:1; letter-spacing:-.01em; color:var(--text); }}
.cr-tag {{ font-family:var(--mono); font-size:13px; align-self:flex-start; margin-top:6px; color:var(--muted); border:1px solid var(--border);
  border-radius:6px; padding:1px 7px; }}
.cr-nav .sp {{ flex:1; }}
.cr-nav .lk {{ font-family:var(--mono); font-size:12px; color:var(--muted); }}
.cr-hero {{ padding:34px 0 18px; position:relative; }}
.cr-hero::before {{ content:""; position:absolute; inset:0 -40px; z-index:-1; opacity:.55;
  background-image:radial-gradient(rgba(139,149,163,.16) 1px, transparent 1px); background-size:22px 22px;
  -webkit-mask-image:linear-gradient(to bottom, #000 0%, transparent 92%);
          mask-image:linear-gradient(to bottom, #000 0%, transparent 92%); }}
.cr-hero h1 {{ font-family:var(--sans); font-size:38px; line-height:1.12; font-weight:650;
  letter-spacing:-.025em; margin:0; color:var(--text); }}
.cr-hero h1 span {{ color:var(--muted); }}
.cr-hero p {{ font-size:15.5px; line-height:1.6; color:var(--muted); max-width:62ch; margin:14px 0 0; }}
.cr-prompt {{ font-family:var(--mono); font-size:12.5px; color:var(--muted); margin:6px 0 -6px; }}
.cr-prompt b {{ color:var(--accent); font-weight:500; }}
.cr-prompt i {{ color:var(--faint); font-style:normal; }}
.cr-note {{ display:flex; gap:12px; align-items:flex-start; border:1px solid var(--border);
  border-left:2px solid var(--accent); border-radius:8px; padding:10px 14px; margin:14px 0 4px;
  background:var(--surface); font-size:13px; line-height:1.55; color:var(--muted); }}
.cr-note b {{ font-family:var(--mono); font-size:10.5px; color:var(--accent); letter-spacing:.1em;
  padding-top:2px; font-weight:500; }}
.cr-empty {{ border:1px dashed var(--border-hi); border-radius:10px; padding:18px 20px;
  font-size:14px; color:var(--muted); line-height:1.6; }}
.cr-empty code {{ font-family:var(--mono); color:var(--text); background:var(--raised);
  border:1px solid var(--border); border-radius:5px; padding:1px 6px; font-size:12.5px; }}
.cr-repo {{ display:flex; flex-wrap:wrap; align-items:center; gap:10px 14px; padding:22px 0 6px; }}
.cr-repo .nm {{ font-family:var(--mono); font-size:22px; color:var(--muted); }}
.cr-repo .nm strong {{ color:var(--text); font-weight:700; }}
.cr-chip {{ font-family:var(--mono); font-size:11.5px; color:var(--muted); border:1px solid var(--border);
  border-radius:6px; padding:2px 8px; transition:border-color var(--fast) var(--ease), color var(--fast) var(--ease); }}
.cr-chip:hover {{ border-color:var(--border-hi); color:var(--text); }}
.cr-chip .dot {{ display:inline-block; width:6px; height:6px; border-radius:50%; margin-right:6px;
  vertical-align:1px; }}
.cr-grid {{ display:grid; gap:12px; margin:10px 0 6px; }}
.cr-kpi {{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:14px 16px;
  transition:border-color var(--fast) var(--ease), transform var(--fast) var(--ease), background var(--fast) var(--ease); }}
.cr-kpi:hover {{ border-color:var(--border-hi); background:var(--raised); transform:translateY(-2px); }}
.cr-kpi .lb {{ font-family:var(--mono); font-size:10.5px; letter-spacing:.1em; text-transform:uppercase;
  color:var(--muted); }}
.cr-kpi .v {{ font-size:28px; font-weight:600; letter-spacing:-.02em; margin-top:4px; color:var(--text);
  font-variant-numeric:tabular-nums; }}
.cr-kpi .s {{ font-family:var(--mono); font-size:11.5px; color:var(--faint); margin-top:2px; }}
.cr-sec {{ margin:38px 0 4px; }}
.cr-sec .hd {{ display:flex; align-items:center; gap:16px; font-family:var(--mono); font-size:24px;
  letter-spacing:.08em; text-transform:uppercase; color:var(--text); font-weight:500; }}
.cr-sec .hd em {{ color:var(--accent); font-style:normal; }}
.cr-sec .hd i {{ flex:1; height:1px; background:var(--border); }}
.cr-sec p {{ font-size:13.5px; color:var(--muted); margin:8px 0 0; max-width:80ch; line-height:1.55; }}
.cr-legend {{ display:flex; gap:16px; font-family:var(--mono); font-size:12px; color:var(--muted);
  margin:2px 0 6px; }}
.cr-legend span::before {{ content:""; display:inline-block; width:8px; height:8px; border-radius:50%;
  margin-right:7px; background:var(--c); }}
.cr-radar {{ width:100%; max-width:560px; display:block; margin:auto; }}
.cr-radar .shape {{ transition:fill-opacity var(--fast) var(--ease); }}
.cr-radar .shape:hover {{ fill-opacity:.34; }}
.cr-radar circle {{ transition:r var(--fast) var(--ease); cursor:default; }}
.cr-radar circle:hover {{ r:8px; }}
.cr-panel {{ background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:14px;
  transition:border-color var(--fast) var(--ease); }}
.cr-panel:hover {{ border-color:var(--border-hi); }}
.cr-foot {{ display:flex; flex-wrap:wrap; gap:8px 18px; justify-content:space-between; margin-top:40px;
  padding-top:14px; border-top:1px solid var(--border); font-family:var(--mono); font-size:11.5px;
  color:var(--faint); }}

/* --- carousel (pure CSS: radio inputs drive a sliding track) --- */
.cr-car input {{ position:absolute; opacity:0; pointer-events:none; }}
.cr-car .view {{ overflow:hidden; border-radius:12px; }}
.cr-car .track {{ display:flex; transition:transform var(--slow) cubic-bezier(.22,.61,.36,1); }}
.cr-card {{ flex:0 0 100%; box-sizing:border-box; background:var(--surface); border:1px solid var(--border);
  border-radius:12px; padding:20px 22px; transition:border-color var(--fast) var(--ease); }}
.cr-card:hover {{ border-color:var(--border-hi); }}
.cr-card .top {{ display:flex; justify-content:space-between; gap:12px; font-family:var(--mono);
  font-size:11.5px; color:var(--muted); letter-spacing:.06em; text-transform:uppercase; }}
.cr-card .top b {{ color:{REPO}; font-weight:500; }}
.cr-card h3 {{ font-family:var(--sans); font-size:19px; font-weight:600; letter-spacing:-.01em;
  margin:10px 0 6px; color:var(--text); line-height:1.35; }}
.cr-card .meta {{ font-family:var(--mono); font-size:12.5px; color:var(--muted); }}
.cr-card .meta .sha {{ color:var(--accent); background:rgba(88,166,255,.08); border:1px solid rgba(88,166,255,.25);
  border-radius:5px; padding:1px 6px; }}
.cr-meter {{ margin:16px 0 14px; }}
.cr-meter .row {{ display:flex; justify-content:space-between; font-family:var(--mono); font-size:11px;
  color:var(--faint); letter-spacing:.06em; text-transform:uppercase; }}
.cr-meter .row span:last-child {{ color:var(--text); }}
.cr-meter .bar {{ height:6px; border-radius:3px; background:#1a2029; margin-top:6px; overflow:hidden; }}
.cr-meter .bar i {{ display:block; height:100%; border-radius:3px;
  background:linear-gradient(90deg, #9e6a03, {REPO}); }}
.cr-diff {{ font-family:var(--mono); font-size:13px; border:1px solid var(--border); border-radius:8px;
  background:#0b0f14; overflow:hidden; }}
.cr-diff .hunk {{ color:#a371f7; background:rgba(163,113,247,.07); padding:6px 12px; font-size:12px; }}
.cr-diff .ln {{ display:flex; gap:12px; padding:5px 12px; transition:background var(--fast) var(--ease); }}
.cr-diff .ln .sg {{ width:10px; flex:none; }}
.cr-diff .ln .tx {{ flex:1; }}
.cr-diff .ln .w {{ color:var(--faint); }}
.cr-diff .up {{ background:rgba(248,81,73,.07); color:#ffb4ae; }}
.cr-diff .up .sg {{ color:var(--up); }}
.cr-diff .up:hover {{ background:rgba(248,81,73,.14); }}
.cr-diff .dn {{ background:rgba(63,185,80,.07); color:#aff5b4; }}
.cr-diff .dn .sg {{ color:var(--down); }}
.cr-diff .dn:hover {{ background:rgba(63,185,80,.14); }}
.cr-car .nav {{ display:flex; align-items:center; justify-content:space-between; gap:12px; margin-top:12px; }}
.cr-car .dots {{ display:flex; gap:6px; }}
.cr-car label {{ font-family:var(--mono); font-size:12px; color:var(--muted); border:1px solid var(--border);
  border-radius:7px; padding:5px 10px; cursor:pointer; user-select:none; background:var(--surface);
  transition:border-color var(--fast) var(--ease), color var(--fast) var(--ease),
    background var(--fast) var(--ease), transform var(--fast) var(--ease); }}
.cr-car label:hover {{ border-color:var(--border-hi); color:var(--text); transform:translateY(-1px); }}
.cr-car .arrows {{ display:flex; gap:6px; }}
.cr-car .arrows label {{ display:none; }}
.cr-car .count span {{ display:none; font-family:var(--mono); font-size:12px; color:var(--faint); }}

.cr-profile .av {{ width:44px; height:44px; border-radius:50%; border:1px solid var(--border-hi); }}
.cr-profile .nm strong {{ color:var(--text); font-family:var(--sans); font-weight:600; }}
.cr-table {{ overflow-x:auto; border:1px solid var(--border); border-radius:10px; background:var(--surface); margin-top:12px; }}
.cr-table table {{ width:100%; border-collapse:collapse; font-family:var(--mono); font-size:12.5px; }}
.cr-table th {{ text-align:left; font-weight:500; font-size:10.5px; letter-spacing:.1em; text-transform:uppercase;
  color:var(--muted); padding:11px 14px; border-bottom:1px solid var(--border); white-space:nowrap; }}
.cr-table td {{ padding:11px 14px; border-bottom:1px solid #1a2029; color:var(--text); white-space:nowrap; }}
.cr-table tr:last-child td {{ border-bottom:none; }}
.cr-table tr {{ transition:background var(--fast) var(--ease); }}
.cr-table tr:hover td {{ background:var(--raised); }}
.cr-table .mt {{ color:var(--faint); }}
.cr-table th .sb {{ font-size:9.5px; letter-spacing:.06em; color:var(--faint); text-transform:none; }}
.cr-table .tg {{ font-size:10.5px; color:var(--muted); border:1px solid var(--border); border-radius:5px; padding:0 5px; margin-left:8px; }}
.cr-table .lg {{ font-size:11px; color:var(--faint); margin-left:8px; }}
.cr-table .dl {{ font-size:11px; margin-left:6px; }}
.cr-table .dl.hi {{ color:{UP}; }} .cr-table .dl.lo {{ color:{DOWN}; }} .cr-table .dl.eq {{ color:var(--faint); }}
.cr-table tr.bad td {{ color:var(--faint); }}
.cr-code {{ border:1px solid var(--border); border-radius:10px; background:var(--surface); overflow:hidden; margin-top:12px;
  transition:border-color var(--fast) var(--ease); }}
.cr-code:hover {{ border-color:var(--border-hi); }}
.cr-code .hd {{ font-family:var(--mono); font-size:11px; letter-spacing:.1em; text-transform:uppercase; color:var(--muted);
  padding:10px 14px; border-bottom:1px solid var(--border); }}
.cr-code pre {{ margin:0; padding:14px; font-family:var(--mono); font-size:13px; line-height:1.7; color:var(--text);
  background:#0b0f14; white-space:pre-wrap; }}
.cr-code p {{ margin:0; padding:10px 14px; font-size:13px; color:var(--muted); border-top:1px solid var(--border); }}
[data-testid="stButtonGroup"] button {{ font-family:var(--mono); font-size:12.5px; transition:all var(--fast) var(--ease); }}
[data-testid="stFormSubmitButton"] button {{ font-family:var(--mono); transition:all var(--fast) var(--ease); }}
[data-testid="stFormSubmitButton"] button:hover {{ transform:translateY(-1px); }}
[data-testid="stForm"] {{ border-color:var(--border); background:var(--surface); }}

@media (max-width:760px) {{
  .cr-hero h1 {{ font-size:28px; }}
  .cr-grid {{ grid-template-columns:repeat(2, minmax(0,1fr)) !important; }}
  .cr-card {{ padding:16px; }}
  .cr-nav .lk.opt {{ display:none; }}
  .cr-word {{ font-size:34px; }}
  .cr-nav img {{ width:40px; height:40px; }}
  .cr-sec .hd {{ font-size:18px; gap:10px; }}
}}
</style>
"""


def logo_data_uri() -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(LOGO.read_bytes()).decode()


def nav_html(model_note: str) -> str:
    return (
        f'<div class="cr cr-nav"><img src="{logo_data_uri()}" alt="CommitRad logo"/>'
        f'<span class="cr-word">commitrad</span><span class="cr-tag">v{escape(__version__)}</span>'
        f'<span class="sp"></span><span class="lk opt">{escape(model_note)}</span>'
        f'<a class="lk" href="{PROJECT_URL}" target="_blank" rel="noopener">source ↗</a></div>'
    )


def page_hero_html(title: str, subtitle: str, body: str) -> str:
    return (
        f'<div class="cr cr-hero"><h1>{escape(title)}<br/><span>{escape(subtitle)}</span></h1>'
        f"<p>{escape(body)}</p></div>"
    )


def note_html(text: str) -> str:
    return f'<div class="cr cr-note"><b>NOTE</b><span>{escape(text)}</span></div>'


def prompt_html(command: str, arg: str) -> str:
    return f'<div class="cr-prompt"><b>$</b> {escape(command)} <i>&lt;{escape(arg)}&gt;</i></div>'


def anchor_html(anchor_id: str) -> str:
    return f'<div id="{escape(anchor_id)}" style="scroll-margin-top:12px"></div>'


def profile_html(
    login: str,
    name: str | None,
    avatar: str | None,
    url: str,
    chips: Sequence[tuple[str, str | None]],
) -> str:
    img = f'<img class="av" src="{escape(avatar)}" alt=""/>' if avatar else ""
    who = f"<strong>{escape(name)}</strong> " if name else ""
    chip_html = "".join(
        f'<span class="cr-chip">{f"<span class=dot style=background:{c}></span>" if c else ""}{escape(t)}</span>'
        for t, c in chips
    )
    return (
        f'<div class="cr cr-repo cr-profile">{img}<span class="nm">{who}'
        f'<a href="{escape(url)}" target="_blank" rel="noopener">@{escape(login)} ↗</a></span>{chip_html}</div>'
    )


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.0%}"


def _delta(mine: float | None, repo: float | None) -> str:
    if mine is None or repo is None:
        return ""
    d = mine - repo
    cls = "hi" if d > 0.02 else "lo" if d < -0.02 else "eq"
    return f' <span class="dl {cls}">{d * 100:+.0f} pts</span>'


def repo_table_html(rows: Sequence[dict[str, Any]]) -> str:
    """The account view's per-repo table (one row per analyzed repo or fork)."""
    sub = "<br/><span class=sb>mine vs repo</span>"
    head = (
        "<tr><th>repository</th><th>commits<br/><span class=sb>mine / all</span></th>"
        f"<th>later fixed{sub}</th><th>via pr<br/><span class=sb>mine</span></th>"
        f"<th>off-hours<br/><span class=sb>mine</span></th><th>risk pctl{sub}</th>"
        "<th>fork drift<br/><span class=sb>vs upstream</span></th></tr>"
    )
    body = []
    for r in rows:
        tags = '<span class="tg">fork</span>' if r.get("fork") else ""
        tags += '<span class="tg">private</span>' if r.get("private") else ""
        lang = f'<span class="lg">{escape(r["language"])}</span>' if r.get("language") else ""
        name = f'<a href="{escape(r["url"])}" target="_blank" rel="noopener">{escape(r["full_name"])}</a>{tags}{lang}'
        if r.get("status") != "ok":
            body.append(
                f'<tr class="bad"><td>{name}</td><td colspan="6">skipped: {escape(str(r.get("error") or ""))[:140]}</td></tr>'
            )
            continue
        drift = "—"
        if r.get("parent"):
            ahead, behind = r.get("ahead_by"), r.get("behind_by")
            drift = (
                f'<span title="vs {escape(r["parent"])}">↑{ahead if ahead is not None else "?"} '
                f"↓{behind if behind is not None else '?'}</span>"
            )
        risk = "—" if r.get("my_risk_pct") is None else f"{r['my_risk_pct'] * 100:.0f}"
        repo_risk = "—" if r.get("repo_risk_pct") is None else f"{r['repo_risk_pct'] * 100:.0f}"
        body.append(
            f"<tr><td>{name}</td><td>{r['my_commits']:,} <span class=mt>/ {r['repo_commits']:,}</span></td>"
            f"<td>{_pct(r.get('my_bug_rate'))} <span class=mt>vs {_pct(r.get('repo_bug_rate'))}</span>"
            f"{_delta(r.get('my_bug_rate'), r.get('repo_bug_rate'))}</td>"
            f"<td>{_pct(r.get('my_via_pr'))}</td><td>{_pct(r.get('my_off_hours'))}</td>"
            f"<td>{risk} <span class=mt>vs {repo_risk}</span></td><td>{drift}</td></tr>"
        )
    return f'<div class="cr cr-table"><table>{head}{"".join(body)}</table></div>'


def code_card_html(title: str, lines: Sequence[str], note: str = "") -> str:
    code = "\n".join(escape(x) for x in lines)
    extra = f"<p>{escape(note)}</p>" if note else ""
    return f'<div class="cr cr-code"><div class="hd">{escape(title)}</div><pre>{code}</pre>{extra}</div>'


def hero_html(disclaimer: str) -> str:
    return (
        '<div class="cr cr-hero"><h1>Commit risk and team strain,<br/>'
        "<span>read straight from git history.</span></h1>"
        "<p>CommitRad traces past bugs back to the commits that introduced them, learns what "
        "risky changes look like, and ranks every commit in a repository with the reasons "
        "attached.</p></div>"
        f'<div class="cr cr-note"><b>NOTE</b><span>{escape(disclaimer)}</span></div>'
        '<div class="cr-prompt"><b>$</b> commitrad analyze <i>&lt;repo-url&gt;</i></div>'
    )


def empty_html() -> str:
    return (
        '<div class="cr cr-empty">Paste any public <code>github.com</code>, '
        "<code>gitlab.com</code> or <code>codeberg.org</code> repository above. A live run "
        "clones it and scores every commit in 20-60 seconds; pallets/click, psf/requests and "
        "pallets/flask are pre-computed and open instantly.</div>"
    )


def repo_html(
    owner: str, name: str, url: str | None, chips: Sequence[tuple[str, str | None]]
) -> str:
    """Repository title + chips; each chip is ``(text, dot_colour_or_None)``."""
    label = (
        f"{escape(owner)}/<strong>{escape(name)}</strong>"
        if owner
        else f"<strong>{escape(name)}</strong>"
    )
    title = (
        f'<a href="{escape(url)}" target="_blank" rel="noopener">{label} ↗</a>' if url else label
    )
    chip_html = "".join(
        f'<span class="cr-chip">{f"<span class=dot style=background:{c}></span>" if c else ""}'
        f"{escape(t)}</span>"
        for t, c in chips
    )
    return f'<div class="cr cr-repo"><span class="nm">{title}</span>{chip_html}</div>'


@dataclass(frozen=True)
class Tile:
    label: str
    value: str
    sub: str = ""


def tiles_html(tiles: Sequence[Tile]) -> str:
    cells = "".join(
        f'<div class="cr-kpi"><div class="lb">{escape(t.label)}</div><div class="v">{escape(t.value)}</div>'
        + (f'<div class="s">{escape(t.sub)}</div>' if t.sub else "")
        + "</div>"
        for t in tiles
    )
    return (
        f'<div class="cr cr-grid" style="grid-template-columns:repeat({len(tiles)},minmax(0,1fr))">'
        f"{cells}</div>"
    )


def section_html(index: int, title: str, description: str = "") -> str:
    desc = f"<p>{escape(description)}</p>" if description else ""
    return (
        f'<div class="cr cr-sec"><div class="hd"><em>{index:02d}</em>{escape(title)}<i></i></div>'
        f"{desc}</div>"
    )


def legend_html(items: Sequence[tuple[str, str]]) -> str:
    spans = "".join(f'<span style="--c:{c}">{escape(t)}</span>' for t, c in items)
    return f'<div class="cr-legend">{spans}</div>'


def footer_html(model_note: str = "lightgbm ensemble · szz labels") -> str:
    return (
        f'<div class="cr cr-foot"><span>commitrad v{escape(__version__)} · {escape(model_note)}'
        "</span><span>signals, not judgments</span></div>"
    )


@dataclass(frozen=True)
class Reason:
    text: str
    shap: float


@dataclass(frozen=True)
class CommitCard:
    rank: int
    subject: str
    author: str
    date: str
    sha: str
    url: str | None
    risk: float
    percentile: float
    band: str
    reasons: Sequence[Reason]
    context: str = ""  # e.g. the repo, in the cross-repo account view


def _card_html(c: CommitCard) -> str:
    sha = escape(c.sha[:7])
    sha_html = (
        f'<a class="sha" href="{escape(c.url)}" target="_blank" rel="noopener">{sha}</a>'
        if c.url
        else f'<span class="sha">{sha}</span>'
    )
    lines = "".join(
        f'<div class="ln {"up" if r.shap > 0 else "dn"}"><span class="sg">{"+" if r.shap > 0 else MINUS}</span>'
        f'<span class="tx">{escape(r.text)}</span><span class="w">{r.shap:+.2f}</span></div>'
        for r in c.reasons
    )
    pct = max(0.0, min(1.0, c.percentile))
    return (
        f'<article class="cr-card"><div class="top"><span>risk rank #{c.rank}'
        f"{(' · ' + escape(c.context)) if c.context else ''}</span>"
        f"<span><b>{escape(c.band)}</b></span></div>"
        f"<h3>{escape(c.subject or '(no message)')}</h3>"
        f'<div class="meta">{escape(c.author)} · {escape(c.date)} · {sha_html}</div>'
        f'<div class="cr-meter"><div class="row"><span>risk percentile</span>'
        f"<span>{pct * 100:.1f} · score {c.risk:.2f}</span></div>"
        f'<div class="bar"><i style="width:{pct * 100:.1f}%"></i></div></div>'
        f'<div class="cr-diff"><div class="hunk">@@ why the model flagged it @@</div>{lines}</div>'
        "</article>"
    )


def carousel_html(cards: Sequence[CommitCard], uid: str = "risky") -> str:
    """A sliding carousel driven by hidden radio inputs: smooth, no rerun, no JavaScript."""
    n = len(cards)
    if not n:
        return ""
    ids = [f"{uid}-{i}" for i in range(1, n + 1)]
    radios = "".join(
        f'<input type="radio" name="{uid}" id="{rid}"{" checked" if i == 0 else ""}/>'
        for i, rid in enumerate(ids)
    )
    track = "".join(_card_html(c) for c in cards)
    dots = "".join(
        f'<label for="{rid}" class="d{i}">{i:02d}</label>' for i, rid in enumerate(ids, 1)
    )
    arrows = "".join(
        f'<label for="{ids[(i - 2) % n]}" class="p{i}" aria-label="Previous commit">← prev</label>'
        f'<label for="{ids[i % n]}" class="n{i}" aria-label="Next commit">next →</label>'
        for i in range(1, n + 1)
    )
    count = "".join(f'<span class="c{i}">{i:02d} / {n:02d}</span>' for i in range(1, n + 1))
    rules = "".join(
        f"#{rid}:checked~.view .track{{transform:translateX(-{(i - 1) * 100}%)}}"
        f"#{rid}:checked~.nav .d{i}{{color:var(--text);border-color:var(--accent);"
        "background:rgba(88,166,255,.10)}"
        f"#{rid}:checked~.nav .p{i},#{rid}:checked~.nav .n{i}{{display:inline-block}}"
        f"#{rid}:checked~.nav .c{i}{{display:inline}}"
        for i, rid in enumerate(ids, 1)
    )
    return (
        f'<style>{rules}</style><div class="cr cr-car">{radios}'
        f'<div class="view"><div class="track">{track}</div></div>'
        f'<div class="nav"><div class="dots">{dots}</div><div class="count">{count}</div>'
        f'<div class="arrows">{arrows}</div></div></div>'
    )

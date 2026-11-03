from __future__ import annotations

from commitrad import ui


def card(rank: int, subject: str = "Fix parser", shap: float = 0.5) -> ui.CommitCard:
    return ui.CommitCard(
        rank=rank, subject=subject, author="Ann", date="2024-01-02", sha="abcdef1234",
        url="https://github.com/o/r/commit/abcdef1234", risk=0.5, percentile=0.97,
        band="in the top 3%", reasons=[ui.Reason("adds 80 lines", shap), ui.Reason("x", -0.1)],
    )  # fmt: skip


def test_builders_escape_untrusted_text() -> None:
    evil = '<img src=x onerror="alert(1)">'
    html = ui.carousel_html([card(1, subject=evil)])
    assert "<img src=x" not in html and "&lt;img src=x" in html
    repo = ui.repo_html(evil, evil, None, [(evil, None)])
    assert "<img" not in repo
    assert "<img" not in ui.tiles_html([ui.Tile(evil, evil, evil)])
    assert "<img" not in ui.section_html(1, evil, evil)
    assert "<img" not in ui.legend_html([(evil, "#fff")])


def test_carousel_has_one_slide_per_card_and_wraps() -> None:
    html = ui.carousel_html([card(i) for i in range(1, 4)], uid="t")
    assert html.count('type="radio"') == 3 and html.count(" checked") == 1
    assert html.count('class="cr-card"') == 3
    assert 'for="t-3" class="p1"' in html  # prev from the first slide wraps to the last
    assert 'for="t-1" class="n3"' in html  # next from the last slide wraps to the first
    assert "#t-2:checked~.view .track{transform:translateX(-100%)}" in html
    assert "01 / 03" in html and ui.carousel_html([]) == ""


def test_diff_lines_show_direction_and_weight() -> None:
    html = ui.carousel_html([card(1)])
    assert '<div class="ln up"><span class="sg">+</span>' in html and "+0.50" in html
    assert f'<div class="ln dn"><span class="sg">{ui.MINUS}</span>' in html and "-0.10" in html
    assert "width:97.0%" in html  # risk meter


def test_page_chrome() -> None:
    assert ui.logo_data_uri().startswith("data:image/svg+xml;base64,")
    assert "commitrad" in ui.nav_html("model") and ui.PROJECT_URL in ui.nav_html("model")
    assert "not judgments" in ui.hero_html("Strain is a signal, not judgments.")
    assert ui.footer_html().count("cr-foot") == 1
    assert "<style>" in ui.CSS and "transition" in ui.CSS

"""Dashboard accuracy: the tailnet link row in the :8091 header.

Context: board-overview (the Kanban boards page) carries a `.links` nav row of
tailnet destinations. This is the same row on the gpu-manager dashboard page, so
one can hop between the fleet pages without typing an address.

The page you are on is marked `aria-current="page"` and rendered as a span, not
an anchor: gpu-manager is "here", so it must NOT link back to itself.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FACADE = REPO / "scripts" / "gpu-manager.py"

# Tailnet destinations, in the order they appear in the row.
LABELS = [
    "T3 home",
    "Boards",
    "Fleet",
    "Storage",
    "Pibook",
    "Brewing",
    "gpu-manager",
    "Hermes kanban",
]
HREFS = [
    "https://hermes.example.org:8443/",
    "http://192.0.2.1:8152/",
    "http://192.0.2.1:8153/",
    "http://192.0.2.1:8154/",
    "http://192.0.2.10:8156/",
    "http://brewing.example.org:8098/",
    "https://hermes.example.org/kanban",
]
_ITEM = r'<a [^>]*>([^<]*)</a>|<span class="here"[^>]*>([^<]*)</span>'


def _dashboard_html() -> str:
    """The dashboard page template, read the way test_mesh_host_cards.py reads it."""
    src = FACADE.read_text()
    match = re.search(r'DASHBOARD_HTML = r"""(.*?)"""', src, re.S)
    assert match, "DASHBOARD_HTML not found in scripts/gpu-manager.py"
    return match.group(1)


def _search(pattern: str, text: str, flags: int = re.S) -> re.Match[str]:
    match = re.search(pattern, text, flags)
    assert match, f"not found in the dashboard template: {pattern}"
    return match


def _links_nav() -> str:
    return _search(r'<nav class="links"[^>]*>(.*?)</nav>', _dashboard_html()).group(1)


def test_link_row_carries_all_labels_in_order():
    labels = [
        item.group(1) if item.group(1) is not None else item.group(2)
        for item in re.finditer(_ITEM, _links_nav())
    ]
    assert labels == LABELS


def test_link_row_hrefs_are_exact_and_in_order():
    hrefs = re.findall(r'<a [^>]*href="([^"]*)"', _links_nav())
    assert hrefs == HREFS
    assert len(hrefs) == 7, "exactly seven links; the eighth entry is 'here'"


def test_gpu_manager_is_here_not_a_self_link():
    nav = _links_nav()
    here = _search(r'<span class="here" aria-current="page">([^<]*)</span>', nav).group(1)
    assert here == "gpu-manager"
    # No anchor points back at this dashboard's own URL.
    assert not re.search(r'<a [^>]*href="[^"]*8091[^"]*"', nav)


def test_link_row_styles_are_inlined_in_the_page_style_block():
    style = _search(r"<style>(.*?)</style>", _dashboard_html()).group(1)
    for rule in (
        r"\.links\s*\{[^}]*display:\s*flex",
        r"\.links\s*\{[^}]*flex-wrap:\s*wrap",
        r"\.links \.here\s*\{[^}]*font-weight:\s*600",
        r"\.links \.muted\s*\{[^}]*font-size:\s*12px",
    ):
        assert re.search(rule, style), rule
    # Inlined only: the page must not pull in an external stylesheet.
    html = _dashboard_html()
    assert 'rel="stylesheet"' not in html
    assert 'aria-label="Tailnet pages"' in html


def test_host_chips_still_render_alongside_the_link_row():
    """The link row is additive: the VRAM-tab host chips still render."""
    html = _dashboard_html()
    assert 'id="vram-mesh-nodes"' in html  # mesh overview card container
    js = max(re.findall(r"<script>(.*?)</script>", html, re.S), key=len)
    for name in ("_meshHostChips", "_meshCardView", "_meshEsc"):
        assert re.search(rf"^function {name}\(.*?^}}\n", js, re.S | re.M), name
    body = _search(r"^async function updateVramTab\(\) \{.*?^}\n", js, re.S | re.M).group(0)
    assert "_meshHostChips(data)" in body
    assert "_meshCardView(" in body
    assert "_meshEsc(view.label)" in body
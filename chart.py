"""
mycoSwarm Chart Tool — Phase 26

Generate publication-ready charts for InsiderLLM articles and benchmarks.
Dark theme matching InsiderLLM brand. Designed for CC (Claude Code) automation.

Usage:
    # As module
    from mycoswarm.chart import bar_chart, line_chart, comparison_table, flow_diagram

    # From CLI (future)
    mycoswarm chart bar --title "VRAM by Model" --data vram.json --output chart.png
    mycoswarm chart line --title "tok/s Over 10 Turns" --data bench.json --output chart.png

Dependencies: matplotlib (optional — not required for core mycoswarm)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")  # headless rendering
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ── InsiderLLM Brand Theme ─────────────────────────────────────────────

THEME = {
    # Backgrounds
    "bg_dark": "#0f1117",       # main background
    "bg_card": "#1a1b26",       # chart area
    "bg_grid": "#2a2b3d",       # subtle gridlines
    # Text
    "text_primary": "#e1e2e8",  # titles, labels
    "text_secondary": "#8b8fa3", # axis labels, annotations
    "text_muted": "#565a6e",    # grid labels
    # Accent palette (ordered for series)
    "accent_1": "#7aa2f7",      # blue — primary
    "accent_2": "#f7768e",      # coral/red — secondary
    "accent_3": "#9ece6a",      # green — tertiary
    "accent_4": "#e0af68",      # amber — quaternary
    "accent_5": "#bb9af7",      # purple — fifth
    "accent_6": "#73daca",      # teal — sixth
    # Semantic
    "positive": "#9ece6a",
    "negative": "#f7768e",
    "neutral": "#7aa2f7",
    "highlight": "#ff9e64",
}

ACCENT_CYCLE = [
    THEME["accent_1"], THEME["accent_2"], THEME["accent_3"],
    THEME["accent_4"], THEME["accent_5"], THEME["accent_6"],
]

# Font stack — Miu has these; CC environments may vary
FONT_FAMILY = "DejaVu Sans"
FONT_TITLE_SIZE = 16
FONT_LABEL_SIZE = 11
FONT_TICK_SIZE = 9
FONT_ANNOTATION_SIZE = 8


def _check_matplotlib():
    if not HAS_MATPLOTLIB:
        print("❌ matplotlib required: pip install matplotlib", file=sys.stderr)
        sys.exit(1)


def _apply_theme(fig, ax):
    """Apply InsiderLLM dark theme to figure and axes."""
    fig.patch.set_facecolor(THEME["bg_dark"])
    ax.set_facecolor(THEME["bg_card"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(THEME["bg_grid"])
    ax.spines["bottom"].set_color(THEME["bg_grid"])

    ax.tick_params(colors=THEME["text_muted"], labelsize=FONT_TICK_SIZE)
    ax.xaxis.label.set_color(THEME["text_secondary"])
    ax.yaxis.label.set_color(THEME["text_secondary"])
    ax.title.set_color(THEME["text_primary"])

    ax.grid(True, axis="y", color=THEME["bg_grid"], linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)


def _watermark(fig):
    """Add subtle InsiderLLM watermark."""
    fig.text(
        0.98, 0.02, "insiderllm.com",
        fontsize=7, color=THEME["text_muted"],
        ha="right", va="bottom", alpha=0.5,
        fontfamily=FONT_FAMILY,
    )


def _save(fig, output: str | Path, dpi: int = 180):
    """Save and close figure."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✅ Saved: {output}")


# ── Chart Types ─────────────────────────────────────────────────────────

def bar_chart(
    labels: list[str],
    values: list[float] | list[list[float]],
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    series_names: list[str] | None = None,
    horizontal: bool = False,
    output: str | Path = "chart.png",
    figsize: tuple[float, float] = (10, 6),
    value_labels: bool = True,
    sort: bool = False,
    colors: list[str] | None = None,
):
    """Bar chart — single or grouped series.

    Args:
        labels: Category labels (x-axis for vertical, y-axis for horizontal)
        values: Single list or list-of-lists for grouped bars
        series_names: Legend labels for grouped bars
        horizontal: Horizontal bar chart if True
        value_labels: Show values on bars
        sort: Sort by value (descending for vertical, ascending for horizontal)
    """
    _check_matplotlib()
    import numpy as np

    # Normalize to list of series
    if values and not isinstance(values[0], (list, tuple)):
        all_series = [values]
    else:
        all_series = values

    n_cats = len(labels)
    n_series = len(all_series)

    if sort and n_series == 1:
        paired = sorted(zip(labels, all_series[0]), key=lambda x: x[1], reverse=not horizontal)
        labels = [p[0] for p in paired]
        all_series = [[p[1] for p in paired]]

    fig, ax = plt.subplots(figsize=figsize)
    _apply_theme(fig, ax)

    bar_width = 0.7 / n_series
    x = np.arange(n_cats)
    palette = colors or ACCENT_CYCLE

    for i, series in enumerate(all_series):
        offset = (i - n_series / 2 + 0.5) * bar_width
        color = palette[i % len(palette)]
        label = series_names[i] if series_names and i < len(series_names) else None

        if horizontal:
            bars = ax.barh(x + offset, series, bar_width * 0.9, color=color, label=label, alpha=0.9)
            if value_labels:
                for bar, val in zip(bars, series):
                    ax.text(
                        bar.get_width() + max(series) * 0.02, bar.get_y() + bar.get_height() / 2,
                        f"{val:g}", va="center", fontsize=FONT_ANNOTATION_SIZE,
                        color=THEME["text_secondary"], fontfamily=FONT_FAMILY,
                    )
        else:
            bars = ax.bar(x + offset, series, bar_width * 0.9, color=color, label=label, alpha=0.9)
            if value_labels:
                for bar, val in zip(bars, series):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, bar.get_height() + max(series) * 0.02,
                        f"{val:g}", ha="center", fontsize=FONT_ANNOTATION_SIZE,
                        color=THEME["text_secondary"], fontfamily=FONT_FAMILY,
                    )

    if horizontal:
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=FONT_TICK_SIZE, fontfamily=FONT_FAMILY)
        ax.set_xlabel(xlabel, fontsize=FONT_LABEL_SIZE, fontfamily=FONT_FAMILY)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=FONT_TICK_SIZE, fontfamily=FONT_FAMILY, rotation=30 if n_cats > 6 else 0, ha="right" if n_cats > 6 else "center")
        ax.set_ylabel(ylabel, fontsize=FONT_LABEL_SIZE, fontfamily=FONT_FAMILY)

    ax.set_title(title, fontsize=FONT_TITLE_SIZE, fontfamily=FONT_FAMILY, pad=15, fontweight="bold")

    if series_names:
        legend = ax.legend(fontsize=FONT_ANNOTATION_SIZE, facecolor=THEME["bg_card"],
                          edgecolor=THEME["bg_grid"], labelcolor=THEME["text_secondary"])

    _watermark(fig)
    _save(fig, output)


def line_chart(
    x_values: list,
    y_series: list[list[float]] | list[float],
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    series_names: list[str] | None = None,
    output: str | Path = "chart.png",
    figsize: tuple[float, float] = (10, 6),
    markers: bool = True,
    fill: bool = False,
    colors: list[str] | None = None,
):
    """Line chart — single or multi-series.

    Args:
        x_values: Shared x-axis values (numbers, strings, or dates)
        y_series: Single list or list-of-lists for multiple lines
        markers: Show data point markers
        fill: Fill area under lines (useful for single series)
    """
    _check_matplotlib()

    if y_series and not isinstance(y_series[0], (list, tuple)):
        all_series = [y_series]
    else:
        all_series = y_series

    fig, ax = plt.subplots(figsize=figsize)
    _apply_theme(fig, ax)

    palette = colors or ACCENT_CYCLE
    marker_style = "o" if markers else None

    for i, series in enumerate(all_series):
        color = palette[i % len(palette)]
        label = series_names[i] if series_names and i < len(series_names) else None
        ax.plot(x_values, series, color=color, label=label, linewidth=2,
                marker=marker_style, markersize=5, alpha=0.9)
        if fill and len(all_series) == 1:
            ax.fill_between(x_values, series, alpha=0.15, color=color)

    ax.set_xlabel(xlabel, fontsize=FONT_LABEL_SIZE, fontfamily=FONT_FAMILY)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL_SIZE, fontfamily=FONT_FAMILY)
    ax.set_title(title, fontsize=FONT_TITLE_SIZE, fontfamily=FONT_FAMILY, pad=15, fontweight="bold")

    if series_names:
        legend = ax.legend(fontsize=FONT_ANNOTATION_SIZE, facecolor=THEME["bg_card"],
                          edgecolor=THEME["bg_grid"], labelcolor=THEME["text_secondary"])

    _watermark(fig)
    _save(fig, output)


def comparison_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    title: str = "",
    output: str | Path = "chart.png",
    figsize: tuple[float, float] | None = None,
    highlight_col: int | None = None,
):
    """Styled comparison table as image — for article feature comparisons.

    Args:
        headers: Column headers
        rows: List of rows, each a list of cell values
        highlight_col: Column index to highlight (0-based)
    """
    _check_matplotlib()

    n_rows = len(rows)
    n_cols = len(headers)
    if figsize is None:
        figsize = (max(8, n_cols * 2.2), max(3, (n_rows + 1) * 0.55 + 1))

    fig, ax = plt.subplots(figsize=figsize)
    _apply_theme(fig, ax)
    ax.axis("off")
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows + 1.5)

    # Header row
    for j, header in enumerate(headers):
        ax.text(
            j + 0.5, n_rows + 0.75, header,
            ha="center", va="center", fontsize=FONT_LABEL_SIZE,
            fontweight="bold", color=THEME["text_primary"], fontfamily=FONT_FAMILY,
        )
    ax.plot([0, n_cols], [n_rows + 0.25, n_rows + 0.25],
            color=THEME["accent_1"], linewidth=2)

    # Data rows
    for i, row in enumerate(rows):
        y = n_rows - i - 0.25
        # Alternating row bg
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0, y - 0.4), n_cols, 0.8,
                         facecolor=THEME["bg_card"], alpha=0.5))
        for j, cell in enumerate(row):
            color = THEME["accent_1"] if j == highlight_col else THEME["text_primary"]
            weight = "bold" if j == highlight_col else "normal"
            ax.text(
                j + 0.5, y, str(cell),
                ha="center", va="center", fontsize=FONT_TICK_SIZE,
                color=color, fontweight=weight, fontfamily=FONT_FAMILY,
            )

    if title:
        ax.set_title(title, fontsize=FONT_TITLE_SIZE, fontfamily=FONT_FAMILY,
                     pad=15, fontweight="bold", color=THEME["text_primary"])

    _watermark(fig)
    _save(fig, output)


def flow_diagram(
    nodes: list[dict],
    edges: list[tuple[int, int]],
    *,
    title: str = "",
    output: str | Path = "chart.png",
    figsize: tuple[float, float] = (12, 6),
    direction: str = "horizontal",
):
    """Simple flow/architecture diagram.

    Args:
        nodes: List of dicts with 'label' (required), 'color' (optional), 'x', 'y' (optional).
               If x/y not provided, auto-layouts in a line.
        edges: List of (from_index, to_index) tuples
        direction: 'horizontal' or 'vertical' for auto-layout
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(THEME["bg_dark"])
    ax.set_facecolor(THEME["bg_dark"])
    ax.axis("off")

    n = len(nodes)

    # Auto-layout if positions not provided
    for i, node in enumerate(nodes):
        if "x" not in node or "y" not in node:
            if direction == "horizontal":
                node["x"] = (i + 0.5) / n
                node["y"] = 0.5
            else:
                node["x"] = 0.5
                node["y"] = 1.0 - (i + 0.5) / n

    # Draw edges first (behind nodes)
    for src_i, dst_i in edges:
        src = nodes[src_i]
        dst = nodes[dst_i]
        ax.annotate(
            "", xy=(dst["x"], dst["y"]), xytext=(src["x"], src["y"]),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="-|>", color=THEME["accent_1"],
                connectionstyle="arc3,rad=0.1", linewidth=1.5, alpha=0.7,
            ),
        )

    # Draw nodes
    for i, node in enumerate(nodes):
        color = node.get("color", ACCENT_CYCLE[i % len(ACCENT_CYCLE)])
        bbox = dict(
            boxstyle="round,pad=0.4", facecolor=color, alpha=0.2,
            edgecolor=color, linewidth=1.5,
        )
        ax.text(
            node["x"], node["y"], node["label"],
            transform=ax.transAxes, ha="center", va="center",
            fontsize=FONT_LABEL_SIZE, fontfamily=FONT_FAMILY,
            color=THEME["text_primary"], fontweight="bold", bbox=bbox,
        )

    if title:
        ax.set_title(title, fontsize=FONT_TITLE_SIZE, fontfamily=FONT_FAMILY,
                     pad=20, fontweight="bold", color=THEME["text_primary"])

    _watermark(fig)
    _save(fig, output)


def before_after(
    labels: list[str],
    before: list[float],
    after: list[float],
    *,
    title: str = "",
    before_label: str = "Before",
    after_label: str = "After",
    ylabel: str = "",
    output: str | Path = "chart.png",
    figsize: tuple[float, float] = (10, 6),
):
    """Before/after comparison — paired bars with improvement arrows."""
    bar_chart(
        labels, [before, after],
        title=title, ylabel=ylabel,
        series_names=[before_label, after_label],
        colors=[THEME["negative"], THEME["positive"]],
        output=output, figsize=figsize,
    )


# ── CLI Entry Point ─────────────────────────────────────────────────────

def chart_from_json(json_path: str | Path, output: str | Path = "chart.png"):
    """Generate chart from a JSON spec file.

    JSON format:
    {
        "type": "bar" | "line" | "table" | "flow" | "before_after",
        "title": "Chart Title",
        "xlabel": "X Axis",
        "ylabel": "Y Axis",
        "data": { ... type-specific ... }
    }
    """
    _check_matplotlib()
    spec = json.loads(Path(json_path).read_text())

    chart_type = spec["type"]
    title = spec.get("title", "")
    data = spec.get("data", {})

    if chart_type == "bar":
        bar_chart(
            labels=data["labels"],
            values=data["values"],
            title=title,
            xlabel=spec.get("xlabel", ""),
            ylabel=spec.get("ylabel", ""),
            series_names=data.get("series_names"),
            horizontal=data.get("horizontal", False),
            sort=data.get("sort", False),
            output=output,
        )
    elif chart_type == "line":
        line_chart(
            x_values=data["x"],
            y_series=data["y"],
            title=title,
            xlabel=spec.get("xlabel", ""),
            ylabel=spec.get("ylabel", ""),
            series_names=data.get("series_names"),
            markers=data.get("markers", True),
            fill=data.get("fill", False),
            output=output,
        )
    elif chart_type == "table":
        comparison_table(
            headers=data["headers"],
            rows=data["rows"],
            title=title,
            highlight_col=data.get("highlight_col"),
            output=output,
        )
    elif chart_type == "flow":
        flow_diagram(
            nodes=data["nodes"],
            edges=data["edges"],
            title=title,
            direction=data.get("direction", "horizontal"),
            output=output,
        )
    elif chart_type == "before_after":
        before_after(
            labels=data["labels"],
            before=data["before"],
            after=data["after"],
            title=title,
            before_label=data.get("before_label", "Before"),
            after_label=data.get("after_label", "After"),
            ylabel=spec.get("ylabel", ""),
            output=output,
        )
    else:
        print(f"❌ Unknown chart type: {chart_type}", file=sys.stderr)
        sys.exit(1)


# ── Standalone runner ───────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python chart.py <spec.json> [output.png]")
        sys.exit(1)
    out = sys.argv[2] if len(sys.argv) > 2 else "chart.png"
    chart_from_json(sys.argv[1], out)

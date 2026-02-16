"""
mycoSwarm Chart Tool — Phase 26 (v3)

Generate publication-ready charts for InsiderLLM articles and benchmarks.
Dark theme matching InsiderLLM brand. Designed for CC (Claude Code) automation.

Chart engines:
- Bar, line, table, before/after: matplotlib
- Flow diagrams: Graphviz (proper layout engine — handles box sizing and arrow routing)

Dependencies: matplotlib (optional), graphviz (optional, for flow diagrams)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# —— InsiderLLM Brand Theme ———————————————————————————————————

THEME = {
    "bg_dark": "#0f1117",
    "bg_card": "#181924",
    "bg_grid": "#252636",
    "text_primary": "#d5d6dc",
    "text_secondary": "#9a9eb2",
    "text_muted": "#5c607a",
    "accent_1": "#6b8fd4",
    "accent_2": "#d4787e",
    "accent_3": "#7fb86a",
    "accent_4": "#c9a055",
    "accent_5": "#9b82c9",
    "accent_6": "#5fb8a8",
    "positive": "#7fb86a",
    "negative": "#d4787e",
    "neutral": "#6b8fd4",
    "highlight": "#d49a5a",
}

ACCENT_CYCLE = [
    THEME["accent_1"], THEME["accent_2"], THEME["accent_3"],
    THEME["accent_4"], THEME["accent_5"], THEME["accent_6"],
]

FONT_FAMILY = "DejaVu Sans"
FONT_TITLE_SIZE = 20
FONT_LABEL_SIZE = 14
FONT_TICK_SIZE = 12
FONT_ANNOTATION_SIZE = 11


def _check_matplotlib():
    if not HAS_MATPLOTLIB:
        print("matplotlib required: pip install matplotlib", file=sys.stderr)
        sys.exit(1)


def _apply_theme(fig, ax):
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
    fig.text(
        0.98, 0.02, "insiderllm.com",
        fontsize=7, color=THEME["text_muted"],
        ha="right", va="bottom", alpha=0.5,
        fontfamily=FONT_FAMILY,
    )


def _save(fig, output: str | Path, dpi: int = 180):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {output}")


# —— Bar Chart ————————————————————————————————————————————————

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
    _check_matplotlib()
    import numpy as np

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
                        bar.get_width() + max(series) * 0.02,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:g}", va="center", fontsize=FONT_ANNOTATION_SIZE,
                        color=THEME["text_secondary"], fontfamily=FONT_FAMILY,
                        fontweight="bold",
                    )
        else:
            bars = ax.bar(x + offset, series, bar_width * 0.9, color=color, label=label, alpha=0.9)
            if value_labels:
                for bar, val in zip(bars, series):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(series) * 0.02,
                        f"{val:g}", ha="center", fontsize=FONT_ANNOTATION_SIZE,
                        color=THEME["text_secondary"], fontfamily=FONT_FAMILY,
                        fontweight="bold",
                    )

    if horizontal:
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=FONT_TICK_SIZE, fontfamily=FONT_FAMILY, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=FONT_LABEL_SIZE, fontfamily=FONT_FAMILY, fontweight="bold")
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=FONT_TICK_SIZE, fontfamily=FONT_FAMILY,
                           rotation=30 if n_cats > 6 else 0,
                           ha="right" if n_cats > 6 else "center",
                           fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=FONT_LABEL_SIZE, fontfamily=FONT_FAMILY, fontweight="bold")

    ax.set_title(title, fontsize=FONT_TITLE_SIZE, fontfamily=FONT_FAMILY, pad=15, fontweight="bold")

    if series_names:
        legend = ax.legend(fontsize=FONT_ANNOTATION_SIZE, facecolor=THEME["bg_card"],
                          edgecolor=THEME["bg_grid"], labelcolor=THEME["text_secondary"])
        for text in legend.get_texts():
            text.set_fontweight("bold")

    _watermark(fig)
    _save(fig, output)


# —— Line Chart ———————————————————————————————————————————————

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

    ax.set_xlabel(xlabel, fontsize=FONT_LABEL_SIZE, fontfamily=FONT_FAMILY, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL_SIZE, fontfamily=FONT_FAMILY, fontweight="bold")
    ax.set_title(title, fontsize=FONT_TITLE_SIZE, fontfamily=FONT_FAMILY, pad=15, fontweight="bold")

    if series_names:
        legend = ax.legend(fontsize=FONT_ANNOTATION_SIZE, facecolor=THEME["bg_card"],
                          edgecolor=THEME["bg_grid"], labelcolor=THEME["text_secondary"])
        for text in legend.get_texts():
            text.set_fontweight("bold")

    _watermark(fig)
    _save(fig, output)


# —— Comparison Table ——————————————————————————————————————————

def comparison_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    title: str = "",
    output: str | Path = "chart.png",
    figsize: tuple[float, float] | None = None,
    highlight_col: int | None = None,
):
    _check_matplotlib()

    n_rows = len(rows)
    n_cols = len(headers)

    # Estimate column widths based on content length
    col_widths = []
    for j in range(n_cols):
        max_len = len(headers[j])
        for row in rows:
            if j < len(row):
                max_len = max(max_len, len(str(row[j])))
        col_widths.append(max_len)
    total_chars = sum(col_widths)
    col_fracs = [w / total_chars for w in col_widths]

    if figsize is None:
        fig_w = max(10, total_chars * 0.18)
        fig_h = max(3, (n_rows + 1) * 0.7 + 1.5)
        figsize = (fig_w, fig_h)

    fig, ax = plt.subplots(figsize=figsize)
    _apply_theme(fig, ax)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_rows + 1.5)

    col_centers = []
    x_pos = 0
    for frac in col_fracs:
        col_centers.append(x_pos + frac / 2)
        x_pos += frac

    for j, header in enumerate(headers):
        ax.text(
            col_centers[j], n_rows + 0.75, header,
            ha="center", va="center", fontsize=FONT_LABEL_SIZE,
            fontweight="bold", color=THEME["text_primary"], fontfamily=FONT_FAMILY,
        )
    ax.plot([0, 1], [n_rows + 0.25, n_rows + 0.25],
            color=THEME["accent_1"], linewidth=2)

    for i, row in enumerate(rows):
        y = n_rows - i - 0.25
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0, y - 0.4), 1, 0.8,
                         facecolor=THEME["bg_card"], alpha=0.5))
        for j, cell in enumerate(row):
            color = THEME["accent_1"] if j == highlight_col else THEME["text_primary"]
            ax.text(
                col_centers[j], y, str(cell),
                ha="center", va="center", fontsize=FONT_TICK_SIZE,
                color=color, fontweight="bold", fontfamily=FONT_FAMILY,
            )

    if title:
        ax.set_title(title, fontsize=FONT_TITLE_SIZE, fontfamily=FONT_FAMILY,
                     pad=15, fontweight="bold", color=THEME["text_primary"])

    _watermark(fig)
    _save(fig, output)


# —— Flow Diagram (Graphviz) ——————————————————————————————————

def _check_graphviz():
    try:
        import graphviz as _gv
        return _gv
    except ImportError:
        print("graphviz required: pip install graphviz (+ system graphviz package)", file=sys.stderr)
        sys.exit(1)


def _blend_color(fg_hex: str, bg_hex: str, alpha: float = 0.25) -> str:
    """Blend foreground onto background at given alpha — fake transparency."""
    r, g, b = int(fg_hex[1:3], 16), int(fg_hex[3:5], 16), int(fg_hex[5:7], 16)
    br, bg_, bb = int(bg_hex[1:3], 16), int(bg_hex[3:5], 16), int(bg_hex[5:7], 16)
    fr = int(r * alpha + br * (1 - alpha))
    fg = int(g * alpha + bg_ * (1 - alpha))
    fb = int(b * alpha + bb * (1 - alpha))
    return f"#{fr:02x}{fg:02x}{fb:02x}"


def flow_diagram(
    nodes: list[dict],
    edges: list[tuple[int, int]],
    *,
    title: str = "",
    output: str | Path = "chart.png",
    direction: str = "horizontal",
    dpi: int = 180,
    **_kwargs,  # absorb unused matplotlib params (figsize, node_spacing)
):
    """Flow/architecture diagram using Graphviz layout engine.

    Args:
        nodes: List of dicts with 'label' (required), 'color' (optional).
        edges: List of (from_index, to_index) tuples.
        direction: 'horizontal' (LR) or 'vertical' (TB).
        dpi: Output resolution.
    """
    gv = _check_graphviz()

    rankdir = "LR" if direction == "horizontal" else "TB"

    dot = gv.Digraph(
        format="png",
        engine="dot",
        graph_attr={
            "bgcolor": THEME["bg_dark"],
            "rankdir": rankdir,
            "dpi": str(dpi),
            "pad": "0.5,0.4",
            "ranksep": "0.8",
            "nodesep": "0.5",
            "margin": "0.2,0.4",
            "label": title + "\n\n " if title else "",
            "labelloc": "t",
            "labeljust": "c",
            "fontname": "DejaVu Sans Bold",
            "fontsize": "20",
            "fontcolor": THEME["text_primary"],
        },
        node_attr={
            "shape": "box",
            "style": "filled,rounded",
            "fontname": "DejaVu Sans Bold",
            "fontsize": "13",
            "fontcolor": THEME["text_primary"],
            "penwidth": "2",
            "margin": "0.3,0.2",
        },
        edge_attr={
            "color": THEME["text_secondary"],
            "penwidth": "2.5",
            "arrowsize": "1.2",
            "arrowhead": "vee",
        },
    )

    # Detect back-edges (cycle arrows) for special styling
    back_edges = set()
    for src_i, dst_i in edges:
        if dst_i < src_i:
            back_edges.add((src_i, dst_i))

    # Add nodes
    for i, node in enumerate(nodes):
        color = node.get("color", ACCENT_CYCLE[i % len(ACCENT_CYCLE)])
        fill = _blend_color(color, THEME["bg_dark"], alpha=0.25)
        dot.node(
            str(i),
            label=node["label"],
            fillcolor=fill,
            color=color,
        )

    # Add edges
    for src_i, dst_i in edges:
        if (src_i, dst_i) in back_edges:
            dot.edge(
                str(src_i), str(dst_i),
                color=THEME["highlight"],
                style="dashed",
                penwidth="3",
                arrowsize="1.3",
                constraint="false",
            )
        else:
            dot.edge(str(src_i), str(dst_i))

    # Render
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stem = str(output.with_suffix(""))
    dot.render(stem, cleanup=True)
    print(f"Saved: {output}")


# —— Before/After —————————————————————————————————————————————

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
    bar_chart(
        labels, [before, after],
        title=title, ylabel=ylabel,
        series_names=[before_label, after_label],
        colors=[THEME["negative"], THEME["positive"]],
        output=output, figsize=figsize,
    )


# —— CLI Entry Point —————————————————————————————————————————

def chart_from_json(json_path: str | Path, output: str | Path = "chart.png"):
    _check_matplotlib()
    spec = json.loads(Path(json_path).read_text())

    chart_type = spec["type"]
    title = spec.get("title", "")
    data = spec.get("data", {})

    if chart_type == "bar":
        bar_chart(labels=data["labels"], values=data["values"], title=title,
                  xlabel=spec.get("xlabel", ""), ylabel=spec.get("ylabel", ""),
                  series_names=data.get("series_names"),
                  horizontal=data.get("horizontal", False),
                  sort=data.get("sort", False), output=output)
    elif chart_type == "line":
        line_chart(x_values=data["x"], y_series=data["y"], title=title,
                   xlabel=spec.get("xlabel", ""), ylabel=spec.get("ylabel", ""),
                   series_names=data.get("series_names"),
                   markers=data.get("markers", True), fill=data.get("fill", False),
                   output=output)
    elif chart_type == "table":
        comparison_table(headers=data["headers"], rows=data["rows"], title=title,
                        highlight_col=data.get("highlight_col"), output=output)
    elif chart_type == "flow":
        flow_diagram(nodes=data["nodes"], edges=data["edges"], title=title,
                    direction=data.get("direction", "horizontal"), output=output)
    elif chart_type == "before_after":
        before_after(labels=data["labels"], before=data["before"],
                    after=data["after"], title=title,
                    before_label=data.get("before_label", "Before"),
                    after_label=data.get("after_label", "After"),
                    ylabel=spec.get("ylabel", ""), output=output)
    else:
        print(f"Unknown chart type: {chart_type}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python chart.py <spec.json> [output.png]")
        sys.exit(1)
    out = sys.argv[2] if len(sys.argv) > 2 else "chart.png"
    chart_from_json(sys.argv[1], out)

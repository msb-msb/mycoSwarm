# Phase 26: Chart Tool — CC Integration Guide

## Setup

```bash
pip install matplotlib
cp chart.py ~/Desktop/mycoSwarm/src/mycoswarm/chart.py
```

## How CC Uses It

### Option A: JSON spec files (preferred for automation)

Create a `.json` file, then run:

```bash
python -m mycoswarm.chart article-assets/poison-cycle-bar.json article-assets/poison-cycle-bar.png
```

### Option B: Python API (for custom/complex charts)

```python
from mycoswarm.chart import bar_chart, line_chart, comparison_table, flow_diagram, before_after
```

## JSON Spec Format

Every spec file has:
```json
{
    "type": "bar | line | table | flow | before_after",
    "title": "Chart Title",
    "xlabel": "optional",
    "ylabel": "optional",
    "data": { ... type-specific ... }
}
```

## Chart Types + Examples

### Bar Chart
```json
{
    "type": "bar",
    "title": "VRAM Usage by Model Size",
    "ylabel": "VRAM (GB)",
    "data": {
        "labels": ["gemma3:1b", "gemma3:4b", "phi4:14b", "gemma3:27b"],
        "values": [1.2, 3.1, 9.8, 17.2],
        "sort": true
    }
}
```

Grouped bars:
```json
{
    "type": "bar",
    "title": "Speed by Node",
    "data": {
        "labels": ["gemma3:1b", "gemma3:4b", "gemma3:27b"],
        "values": [[45, 18, 34], [3, 1.5, 0]],
        "series_names": ["Miu (3090)", "naru (CPU)"]
    }
}
```

Options: `horizontal: bool`, `sort: bool`, `value_labels: bool`

### Line Chart
```json
{
    "type": "line",
    "title": "tok/s Over 10 Turns",
    "xlabel": "Turn",
    "ylabel": "Tokens/sec",
    "data": {
        "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "y": [[34, 33, 34, 33, 32, 33, 34, 33, 34, 33]],
        "series_names": ["gemma3:27b"],
        "markers": true,
        "fill": false
    }
}
```

### Comparison Table
```json
{
    "type": "table",
    "title": "Defense Layers",
    "data": {
        "headers": ["Layer", "Mechanism", "Status"],
        "rows": [
            ["Source Priority", "2x RRF boost for user docs", "Done"],
            ["Confidence Gating", "<0.3 excluded from index", "Done"],
            ["Contradiction Detection", "Cross-reference vs docs", "Done"]
        ],
        "highlight_col": 2
    }
}
```

### Flow Diagram
```json
{
    "type": "flow",
    "title": "Query Pipeline",
    "data": {
        "nodes": [
            {"label": "Intent Gate"},
            {"label": "Memory"},
            {"label": "RAG"},
            {"label": "Inference"}
        ],
        "edges": [[0,1], [1,2], [2,3]],
        "direction": "horizontal"
    }
}
```

Custom positions (for non-linear layouts):
```json
{
    "nodes": [
        {"label": "Query", "x": 0.1, "y": 0.5},
        {"label": "Gate", "x": 0.3, "y": 0.8},
        {"label": "RAG", "x": 0.3, "y": 0.2},
        {"label": "Output", "x": 0.7, "y": 0.5}
    ],
    "edges": [[0,1], [0,2], [1,3], [2,3]]
}
```

### Before/After
```json
{
    "type": "before_after",
    "title": "Immune System Impact",
    "data": {
        "labels": ["Hallucination %", "Citation %", "Grounding"],
        "before": [45, 20, 30],
        "after": [8, 85, 82],
        "before_label": "Pre-Fix",
        "after_label": "Post-Fix"
    }
}
```

## Article-Specific Charts to Generate

### Article 1: "Why Your AI Keeps Lying"
1. `poison-cycle-flow.json` — The feedback loop diagram
2. `defense-layers-table.json` — Immune system layers
3. `before-after-grounding.json` — Pre/post fix metrics
4. `root-causes-bar.json` — 5 root causes ranked by impact

### Article 2: "Distributed Wisdom"
1. `vram-by-model-bar.json` — Model VRAM requirements
2. `toks-comparison-line.json` — 3090 vs CPU inference speed
3. `swarm-nodes-table.json` — Node specs and roles
4. `query-pipeline-flow.json` — Intent → Memory → RAG → Inference flow
5. `memory-types-table.json` — Four memory streams comparison

## InsiderLLM Output Convention

Save all article charts to:
```
~/Desktop/insiderllm/static/images/articles/{article-slug}/
```

Example:
```bash
mkdir -p ~/Desktop/insiderllm/static/images/articles/hallucination-feedback-loop/
python -m mycoswarm.chart specs/poison-cycle.json static/images/articles/hallucination-feedback-loop/poison-cycle.png
```

## Theme

Dark background (#0f1117), muted grid, blue/coral/green accent palette.
Watermark: "insiderllm.com" bottom-right.
All charts render at 180 DPI — good for web, reasonable file size.

## Tip for CC

Use `--output` for emoji-free labels to avoid missing glyph warnings. Use text status
markers like "Done", "Next", "Planned" instead of ✅ 🔨 etc. Emoji render on Miu
(full font set) but not in all environments.

## Adding to pyproject.toml

```toml
[project.optional-dependencies]
chart = ["matplotlib>=3.8.0"]
all = ["pymupdf>=1.24.0", "chromadb>=0.4.0", "rank-bm25>=0.2.2", "matplotlib>=3.8.0"]
```

## Adding CLI command (future)

Add to cli.py:
```python
chart_parser = subparsers.add_parser('chart', help='Generate charts from JSON specs')
chart_parser.add_argument('spec', help='JSON spec file path')
chart_parser.add_argument('-o', '--output', default='chart.png', help='Output PNG path')
chart_parser.set_defaults(func=cmd_chart)
```

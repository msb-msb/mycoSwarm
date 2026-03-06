"""Article category profiles for mycoSwarm pipeline."""

CATEGORIES = {
    "research-driven": {
        "yaml": "pipelines/article-full.yaml",
        "max_rounds": 5,
        "min_depth": 7,
        "context_required": False,
        "writer_tone": "data-driven analysis with citations",
        "writer_model": "qwen3.5:35b-a3b",
    },
    "buying-guide": {
        "yaml": "pipelines/article-full.yaml",
        "max_rounds": 5,
        "min_depth": 7,
        "context_required": False,
        "writer_tone": "opinionated recommendations with comparison tables",
        "writer_model": "qwen3.5:35b-a3b",
    },
    "vs": {
        "yaml": "pipelines/article-full.yaml",
        "max_rounds": 4,
        "min_depth": 6,
        "context_required": False,
        "writer_tone": "head-to-head comparison, balanced and specific",
        "writer_model": "qwen3.5:35b-a3b",
    },
    "how-to": {
        "yaml": "pipelines/article-full.yaml",
        "max_rounds": 3,
        "min_depth": 5,
        "context_required": False,
        "writer_tone": "step-by-step instructional with code blocks",
        "writer_model": "qwen3.5:35b-a3b",
    },
    "news": {
        "yaml": "pipelines/article-full.yaml",
        "max_rounds": 2,
        "min_depth": 4,
        "context_required": False,
        "writer_tone": "timely and factual, lead with what changed",
        "writer_model": "gemma3:27b",
    },
    "model-release": {
        "yaml": "pipelines/article-full.yaml",
        "max_rounds": 2,
        "min_depth": 4,
        "context_required": False,
        "writer_tone": "technical summary of new model capabilities and benchmarks",
        "writer_model": "gemma3:27b",
    },
    "experience-driven": {
        "yaml": "pipelines/article-full.yaml",
        "max_rounds": 3,
        "min_depth": 4,
        "context_required": True,
        "writer_tone": "first-person narrative grounded in personal experience",
        "writer_model": "qwen3.5:35b-a3b",
    },
    "explainer": {
        "yaml": "pipelines/article-short.yaml",
        "max_rounds": 0,
        "min_depth": 0,
        "context_required": False,
        "writer_tone": "concept-first, clear and accessible, no jargon without definition",
        "writer_model": "qwen3.5:35b-a3b",
    },
    "opinion": {
        "yaml": "pipelines/article-short.yaml",
        "max_rounds": 0,
        "min_depth": 0,
        "context_required": True,
        "writer_tone": "direct editorial voice, take a clear stance",
        "writer_model": "qwen3.5:35b-a3b",
    },
}

DEFAULT_CATEGORY = "research-driven"


def get_category(name: str) -> dict:
    """Return category profile, falling back to default."""
    return CATEGORIES.get(name, CATEGORIES[DEFAULT_CATEGORY])


def list_categories() -> list[str]:
    return list(CATEGORIES.keys())

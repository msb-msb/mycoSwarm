"""mycoSwarm document library with RAG (Retrieval-Augmented Generation).

Local document ingestion, chunking, embedding via Ollama, and vector search
via ChromaDB. Documents are stored in ~/mycoswarm-docs/ and indexed into
~/.config/mycoswarm/library/ for retrieval at query time.

Hybrid retrieval: BM25 keyword search + vector similarity, merged via
Reciprocal Rank Fusion (RRF).

Supports: PDF, TXT, MD, HTML, CSV, JSON.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# --- Constants ---

LIBRARY_DIR = Path("~/mycoswarm-docs/").expanduser()
CHROMA_DIR = Path("~/.config/mycoswarm/library").expanduser()
OLLAMA_BASE = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
CHUNK_SIZE = 384       # words (~512 tokens at 1.33 tokens/word)
CHUNK_OVERLAP = 38     # words (~50 tokens)
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".html", ".csv", ".json"}


# --- Text Extraction ---


def extract_file_text(path: Path) -> str:
    """Read file bytes and extract text based on extension."""
    from mycoswarm.worker import _extract_text, _strip_html  # noqa: F401

    raw = path.read_bytes()
    filetype = path.suffix.lstrip(".").lower()
    if not filetype:
        filetype = "txt"
    return _extract_text(raw, filetype)


# --- Text Cleaning ---

# Patterns for boilerplate removal
_PAGE_NUM_LINE_RE = re.compile(r"^\s*\d{1,4}\s*$")
_PAGE_X_OF_Y_RE = re.compile(r"(?i)\bpage\s+\d+\s+of\s+\d+\b")
_COPYRIGHT_RE = re.compile(
    r"(?i)^.*(?:©|copyright|\(c\))\s*\d{4}.*$", re.MULTILINE
)
_CONFIDENTIAL_RE = re.compile(r"(?i)^\s*confidential\s*$")


def clean_text(text: str, doc_type: str = "") -> str:
    """Clean extracted text before chunking.

    - Strips leading/trailing whitespace per line
    - Removes repeated headers/footers (lines appearing 3+ times)
    - Removes standalone page numbers, "Page X of Y", copyright lines,
      "Confidential" watermarks
    - Collapses multiple blank lines to at most two newlines
    - Collapses multiple spaces to one
    - Preserves markdown headings for section extraction
    """
    lines = text.splitlines()

    # --- Strip each line ---
    lines = [line.strip() for line in lines]

    # --- Detect repeated headers/footers ---
    # Count exact occurrences; lines appearing 3+ times are boilerplate.
    # Skip blank lines and markdown headings.
    line_counts: dict[str, int] = {}
    for line in lines:
        if not line or line.startswith("#"):
            continue
        line_counts[line] = line_counts.get(line, 0) + 1
    repeated = {line for line, count in line_counts.items() if count >= 3}

    # --- Filter lines ---
    cleaned: list[str] = []
    for line in lines:
        # Preserve markdown headings unconditionally
        if line.startswith("#"):
            cleaned.append(line)
            continue

        # Remove repeated boilerplate
        if line in repeated:
            continue

        # Remove standalone page numbers
        if _PAGE_NUM_LINE_RE.match(line):
            continue

        # Remove "Page X of Y"
        if _PAGE_X_OF_Y_RE.search(line):
            continue

        # Remove copyright notices
        if _COPYRIGHT_RE.match(line):
            continue

        # Remove "Confidential" watermarks
        if _CONFIDENTIAL_RE.match(line):
            continue

        cleaned.append(line)

    text = "\n".join(cleaned)

    # --- Normalize whitespace ---
    # Collapse 3+ consecutive newlines to 2 (preserving paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces to one
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


# --- Chunking ---


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split text into overlapping word-based chunks.

    Token counts are approximated as words * 1.33.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap

    return chunks


# --- Section Heading Extraction ---

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)", re.MULTILINE)


def _extract_chunk_sections(text: str, chunks: list[str]) -> list[str]:
    """Map each chunk to the nearest markdown heading above it.

    Builds a per-word heading array from the original text, then locates
    each chunk's first words in that array to determine its section.
    Returns a list parallel to chunks with the heading text (or "untitled").
    """
    # Build a per-word heading array by walking lines
    current_heading = "untitled"
    word_headings: list[str] = []
    for line in text.splitlines():
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            current_heading = heading_match.group(2).strip()
        for _ in line.split():
            word_headings.append(current_heading)

    # All words from the original text, for matching chunk starts
    all_words = text.split()

    sections: list[str] = []
    search_from = 0
    for chunk in chunks:
        chunk_words = chunk.split()
        if not chunk_words:
            sections.append("untitled")
            continue

        # Find where this chunk's first words appear in the word list
        prefix = chunk_words[:5]
        found = False
        for idx in range(search_from, len(all_words) - len(prefix) + 1):
            if all_words[idx : idx + len(prefix)] == prefix:
                sections.append(word_headings[idx])
                search_from = idx  # don't go backwards
                found = True
                break
        if not found:
            # Fallback: try from beginning (overlap edge case)
            for idx in range(0, len(all_words) - len(prefix) + 1):
                if all_words[idx : idx + len(prefix)] == prefix:
                    sections.append(word_headings[idx])
                    found = True
                    break
            if not found:
                sections.append("untitled")

    return sections


# --- Embedding ---


def _get_embedding_model(preferred: str | None = None) -> str:
    """Return the best available embedding model.

    Tries preferred model first, then nomic-embed-text, then falls back
    to the first available Ollama model with a warning.
    """
    if preferred:
        return preferred

    try:
        resp = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
        return EMBEDDING_MODEL  # return default, let embed_text handle the error

    # Look for nomic-embed-text variants
    for m in models:
        if "nomic-embed-text" in m:
            return m

    # Fall back to first available model
    if models:
        print(f"  ⚠️  No embedding model found, using {models[0]} (may produce poor results)")
        return models[0]

    return EMBEDDING_MODEL


def embed_text(text: str, model: str = EMBEDDING_MODEL) -> list[float] | None:
    """Generate embedding vector via Ollama /api/embeddings.

    Returns embedding list or None on failure.
    """
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=60.0,
        )
        resp.raise_for_status()
        embedding = resp.json().get("embedding")
        return embedding if embedding else None
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError) as e:
        logger.warning(f"Embedding failed: {e}")
        return None


# --- ChromaDB ---


def _get_collection(model: str = EMBEDDING_MODEL):
    """Get or create the ChromaDB collection for document storage."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name="mycoswarm_docs",
        metadata={"hnsw:space": "cosine"},
    )


def _get_session_collection():
    """Get or create the ChromaDB collection for session memory."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name="session_memory",
        metadata={"hnsw:space": "cosine"},
    )


# --- BM25 Keyword Index ---


class BM25Index:
    """BM25 keyword index over a ChromaDB collection.

    Lazy-builds on first search, caches until invalidate() is called.
    """

    def __init__(self, collection_name: str):
        self._collection_name = collection_name
        self._built = False
        self._index = None  # BM25Okapi | None
        self._doc_ids: list[str] = []
        self._documents: list[str] = []
        self._metadatas: list[dict] = []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercase tokenization."""
        return text.lower().split()

    def _build(self) -> None:
        """Fetch all documents from collection and build BM25 index."""
        self._doc_ids = []
        self._documents = []
        self._metadatas = []
        self._index = None
        self._built = True

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.debug("rank-bm25 not installed, BM25 search disabled")
            return

        try:
            if self._collection_name == "mycoswarm_docs":
                collection = _get_collection()
            else:
                collection = _get_session_collection()

            all_data = collection.get(include=["documents", "metadatas"])
        except Exception:
            return

        if not all_data or not all_data.get("ids"):
            return

        self._doc_ids = all_data["ids"]
        self._documents = all_data["documents"]
        self._metadatas = all_data.get("metadatas") or [{} for _ in self._doc_ids]

        corpus = [self._tokenize(doc) for doc in self._documents]
        if corpus:
            self._index = BM25Okapi(corpus)

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """Return ranked results: [{id, document, metadata, bm25_score}]."""
        if not self._built:
            self._build()

        if self._index is None or not self._doc_ids:
            return []

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        scores = self._index.get_scores(tokenized_query)

        indexed_scores = sorted(
            ((i, float(scores[i])) for i in range(len(scores))),
            key=lambda x: x[1],
            reverse=True,
        )

        results: list[dict] = []
        for idx, score in indexed_scores[:n_results]:
            results.append({
                "id": self._doc_ids[idx],
                "document": self._documents[idx],
                "metadata": self._metadatas[idx],
                "bm25_score": score,
            })
        return results

    def invalidate(self) -> None:
        """Force rebuild on next search."""
        self._built = False
        self._index = None
        self._doc_ids = []
        self._documents = []
        self._metadatas = []


# Module-level BM25 index instances (lazy-built, cached)
_bm25_docs = BM25Index("mycoswarm_docs")
_bm25_sessions = BM25Index("session_memory")


def _rrf_fuse(
    vector_ids: list[str], bm25_ids: list[str], k: int = 60,
) -> dict[str, float]:
    """Reciprocal Rank Fusion of two ranked ID lists.

    score(d) = sum(1 / (k + rank)) for each list containing d.
    Higher score = better.  k=60 is the standard default.
    """
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(vector_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    for rank, doc_id in enumerate(bm25_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def index_session_summary(
    session_id: str,
    summary: str,
    date: str,
    model: str | None = None,
    topic: str | None = None,
) -> bool:
    """Index a session summary into the session_memory ChromaDB collection.

    Returns True if indexed successfully, False on failure.
    """
    model = _get_embedding_model(model)
    embedding = embed_text(summary, model)
    if embedding is None:
        return False

    # Extract topic keywords: first 5 significant words (>3 chars)
    words = summary.split()
    keywords = [w.strip(".,;:!?\"'()") for w in words if len(w) > 3][:10]
    topic_keywords = " ".join(keywords)

    metadata = {
        "session_id": session_id,
        "date": date,
        "topic_keywords": topic_keywords,
        "embedding_model": model,
    }
    if topic is not None:
        metadata["topic"] = topic

    collection = _get_session_collection()
    collection.upsert(
        ids=[session_id],
        documents=[summary],
        embeddings=[embedding],
        metadatas=[metadata],
    )
    _bm25_sessions.invalidate()
    return True


def reindex_sessions(model: str | None = None) -> dict:
    """Drop the session_memory collection and reindex from sessions.jsonl.

    Reads all entries from the JSONL file, splits multi-topic summaries,
    and indexes each topic chunk individually.

    Returns {"sessions": count, "topics": count, "failed": count}.
    """
    import chromadb
    from mycoswarm.memory import load_session_summaries, split_session_topics

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Drop the session_memory collection
    try:
        client.delete_collection("session_memory")
    except (ValueError, Exception):
        pass  # collection didn't exist

    _bm25_sessions.invalidate()

    model = _get_embedding_model(model)
    summaries = load_session_summaries(limit=10000)
    stats = {"sessions": len(summaries), "topics": 0, "failed": 0}

    for entry in summaries:
        name = entry.get("session_name", "")
        summary = entry.get("summary", "")
        ts = entry.get("timestamp", "")
        chat_model = entry.get("model", "")
        date = ts[:10] if ts else ""

        if not summary:
            continue

        topics = split_session_topics(summary, chat_model)
        for i, t in enumerate(topics):
            ok = index_session_summary(
                session_id=f"{name}::topic_{i}",
                summary=t["summary"],
                date=date,
                model=model,
                topic=t["topic"],
            )
            if ok:
                stats["topics"] += 1
            else:
                stats["failed"] += 1

    return stats


def search_sessions(
    query: str, n_results: int = 3, model: str | None = None,
) -> list[dict]:
    """Search session memory for summaries matching a query.

    Returns [{"summary": text, "session_id": id, "date": str, "score": float}].
    """
    model = _get_embedding_model(model)
    query_embedding = embed_text(query, model)
    if query_embedding is None:
        return []

    try:
        collection = _get_session_collection()
    except Exception:
        return []

    count = collection.count()
    if count == 0:
        return []
    n = min(n_results, count)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
    )

    hits: list[dict] = []
    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else 0.0
            hits.append({
                "summary": doc,
                "session_id": meta.get("session_id", ""),
                "date": meta.get("date", ""),
                "topic": meta.get("topic", ""),
                "score": round(distance, 4),
            })

    return hits


# --- LLM Re-Ranking ---


_EMBEDDING_ONLY = ("nomic-embed-text", "mxbai-embed", "all-minilm", "snowflake-arctic-embed")


def _pick_rerank_model() -> str | None:
    """Pick a small, fast model for re-ranking. Returns None if unavailable."""
    try:
        resp = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError):
        return None

    # Prefer small models for speed, skip embedding-only models
    for pattern in ("gemma3:1b", "gemma3:4b", "llama3.2:1b", "llama3.2:3b"):
        for m in models:
            if pattern in m and not any(e in m.lower() for e in _EMBEDDING_ONLY):
                return m

    # Fall back to first non-embedding model
    for m in models:
        if not any(e in m.lower() for e in _EMBEDDING_ONLY):
            return m
    return None


def _score_chunk(query: str, text: str, llm_model: str) -> float:
    """Ask the LLM to rate relevance of text to query. Returns 0-10."""
    prompt = (
        "Rate the relevance of this text to the question on a scale of 0-10.\n"
        f"Question: {query}\n"
        f"Text: {text[:1000]}\n"
        "Reply with ONLY a number 0-10."
    )
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE}/api/generate",
            json={
                "model": llm_model,
                "prompt": prompt,
                "options": {"temperature": 0.0, "num_predict": 8},
                "stream": False,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        # Extract first number from response
        for token in raw.split():
            token = token.strip(".,;:!?()[]")
            try:
                score = float(token)
                return max(0.0, min(10.0, score))
            except ValueError:
                continue
        return 5.0  # parse failure fallback
    except Exception:
        return 5.0


def rerank(
    query: str,
    chunks: list[dict],
    llm_model: str | None = None,
    top_k: int = 5,
    text_key: str = "text",
) -> list[dict]:
    """Re-rank retrieved chunks using LLM relevance scoring.

    Each chunk is scored by a small LLM on 0-10 relevance.
    Returns top_k chunks sorted by descending relevance score.
    Each returned chunk gets a "rerank_score" field.
    """
    if not chunks:
        return []

    if llm_model is None:
        llm_model = _pick_rerank_model()
    if llm_model is None:
        # No LLM available — return as-is, truncated
        return chunks[:top_k]

    scored: list[tuple[float, dict]] = []
    for chunk in chunks:
        text = chunk.get(text_key, chunk.get("summary", ""))
        score = _score_chunk(query, text, llm_model)
        chunk_copy = dict(chunk)
        chunk_copy["rerank_score"] = score
        scored.append((score, chunk_copy))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]


def search_all(
    query: str,
    n_results: int = 5,
    model: str | None = None,
    rerank_model: str | None = None,
    do_rerank: bool = False,
    session_boost: bool = False,
    intent: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    """Search BOTH document library and session memory with hybrid retrieval.

    Uses vector similarity (ChromaDB) + BM25 keyword matching, merged via
    Reciprocal Rank Fusion (RRF).  Embeds the query once.
    When do_rerank=True, passes candidates through LLM re-ranking.
    When session_boost=True, fetches 2x session results (for past-reference queries).
    When intent is provided, adjusts candidate counts based on mode/scope.
    Returns (doc_hits, session_hits) so callers can format them
    with distinct labels ([D1]... vs [S1]...).
    """
    # --- Intent-driven early exits and candidate adjustments ---
    if intent is not None:
        mode = intent.get("mode", "explore")
        if mode == "chat":
            return [], []

    model = _get_embedding_model(model)

    # Warn on model mismatch
    mismatch_warning = check_embedding_model(model)
    if mismatch_warning:
        print(mismatch_warning)

    query_embedding = embed_text(query, model)
    if query_embedding is None:
        return [], []

    # When re-ranking, fetch more candidates for the LLM to filter
    n_candidates = n_results * 2 if do_rerank else n_results
    # When session_boost is active, fetch 2x session candidates
    n_session_candidates = n_candidates * 2 if session_boost else n_candidates

    # Intent-driven candidate adjustments (applied after base calculation)
    if intent is not None:
        mode = intent.get("mode", "explore")
        scope = intent.get("scope", "all")

        if mode == "recall":
            n_session_candidates = n_candidates * 3

        if scope in ("session", "personal"):
            n_session_candidates = n_candidates * 3
            n_candidates = max(2, n_candidates // 2)
        elif scope == "docs" or scope == "documents":
            n_session_candidates = max(2, n_candidates // 2)

    # --- Document collection: hybrid search ---
    doc_hits: list[dict] = []
    try:
        doc_col = _get_collection(model)
        doc_count = doc_col.count()
        if doc_count > 0:
            n_fetch = min(n_candidates * 2, doc_count)

            # Vector search
            vec_results = doc_col.query(
                query_embeddings=[query_embedding],
                n_results=n_fetch,
            )

            # BM25 search
            bm25_results = _bm25_docs.search(query, n_results=n_fetch)

            # Build lookup: id -> result dict
            doc_data: dict[str, dict] = {}
            vec_ids: list[str] = []

            if vec_results and vec_results["ids"]:
                for i, doc_id in enumerate(vec_results["ids"][0]):
                    vec_ids.append(doc_id)
                    meta = vec_results["metadatas"][0][i] if vec_results["metadatas"] else {}
                    distance = vec_results["distances"][0][i] if vec_results["distances"] else 0.0
                    doc_data[doc_id] = {
                        "text": vec_results["documents"][0][i],
                        "source": meta.get("source", ""),
                        "score": round(distance, 4),
                        "chunk_index": meta.get("chunk_index", 0),
                        "section": meta.get("section", "untitled"),
                        "doc_type": meta.get("doc_type", ""),
                        "file_date": meta.get("file_date", 0.0),
                        "embedding_model": meta.get("embedding_model", ""),
                    }

            bm25_ids: list[str] = []
            for hit in bm25_results:
                bm25_ids.append(hit["id"])
                if hit["id"] not in doc_data:
                    meta = hit["metadata"]
                    doc_data[hit["id"]] = {
                        "text": hit["document"],
                        "source": meta.get("source", ""),
                        "score": 0.0,
                        "chunk_index": meta.get("chunk_index", 0),
                        "section": meta.get("section", "untitled"),
                        "doc_type": meta.get("doc_type", ""),
                        "file_date": meta.get("file_date", 0.0),
                        "embedding_model": meta.get("embedding_model", ""),
                    }

            # RRF fusion
            rrf_scores = _rrf_fuse(vec_ids, bm25_ids)
            sorted_ids = sorted(
                rrf_scores, key=lambda x: rrf_scores[x], reverse=True,
            )[:n_candidates]

            for doc_id in sorted_ids:
                hit = doc_data[doc_id]
                hit["rrf_score"] = round(rrf_scores[doc_id], 6)
                doc_hits.append(hit)
    except Exception:
        pass

    # --- Session memory collection: hybrid search ---
    session_hits: list[dict] = []
    try:
        sess_col = _get_session_collection()
        sess_count = sess_col.count()
        if sess_count > 0:
            n_fetch = min(n_session_candidates * 2, sess_count)

            # Vector search
            vec_results = sess_col.query(
                query_embeddings=[query_embedding],
                n_results=n_fetch,
            )

            # BM25 search
            bm25_results = _bm25_sessions.search(query, n_results=n_fetch)

            # Build lookup
            sess_data: dict[str, dict] = {}
            vec_ids_s: list[str] = []

            if vec_results and vec_results["ids"]:
                for i, doc_id in enumerate(vec_results["ids"][0]):
                    vec_ids_s.append(doc_id)
                    meta = vec_results["metadatas"][0][i] if vec_results["metadatas"] else {}
                    distance = vec_results["distances"][0][i] if vec_results["distances"] else 0.0
                    sess_data[doc_id] = {
                        "summary": vec_results["documents"][0][i],
                        "session_id": meta.get("session_id", ""),
                        "date": meta.get("date", ""),
                        "topic": meta.get("topic", ""),
                        "score": round(distance, 4),
                    }

            bm25_ids_s: list[str] = []
            for hit in bm25_results:
                bm25_ids_s.append(hit["id"])
                if hit["id"] not in sess_data:
                    meta = hit["metadata"]
                    sess_data[hit["id"]] = {
                        "summary": hit["document"],
                        "session_id": meta.get("session_id", ""),
                        "date": meta.get("date", ""),
                        "topic": meta.get("topic", ""),
                        "score": 0.0,
                    }

            # RRF fusion
            rrf_scores = _rrf_fuse(vec_ids_s, bm25_ids_s)
            sorted_ids = sorted(
                rrf_scores, key=lambda x: rrf_scores[x], reverse=True,
            )[:n_session_candidates]

            for doc_id in sorted_ids:
                hit = sess_data[doc_id]
                hit["rrf_score"] = round(rrf_scores[doc_id], 6)
                session_hits.append(hit)
    except Exception:
        pass

    # --- LLM re-ranking ---
    if do_rerank and (doc_hits or session_hits):
        doc_hits = rerank(
            query, doc_hits,
            llm_model=rerank_model, top_k=n_results, text_key="text",
        )
        session_hits = rerank(
            query, session_hits,
            llm_model=rerank_model, top_k=n_results, text_key="summary",
        )

    return doc_hits, session_hits


# --- Embedding Model Tracking ---

_MODEL_FILE = CHROMA_DIR / "embedding_model.json"


def _save_embedding_model(model: str) -> None:
    """Record which embedding model was used for indexing."""
    _MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    _MODEL_FILE.write_text(json.dumps({"embedding_model": model}))


def _load_embedding_model() -> str | None:
    """Load the stored embedding model name, or None if not set."""
    try:
        data = json.loads(_MODEL_FILE.read_text())
        return data.get("embedding_model")
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _normalize_model_name(name: str) -> str:
    """Strip the ':latest' default tag suffix from an Ollama model name."""
    return name.removesuffix(":latest")


def check_embedding_model(current_model: str) -> str | None:
    """Check if current model matches the stored index model.

    Returns a warning string if there's a mismatch, None if OK.
    """
    stored = _load_embedding_model()
    if stored is None:
        return None
    if _normalize_model_name(stored) == _normalize_model_name(current_model):
        return None
    return (
        f"⚠️  Library was indexed with {stored} but current model is "
        f"{current_model}. Results may be inaccurate. "
        f"Run: mycoswarm library reindex"
    )


# --- Public API ---


def ingest_file(path: Path, model: str | None = None) -> dict:
    """Ingest a single file: extract text, chunk, embed, store in ChromaDB.

    Returns {"file": name, "chunks": count, "model": model}.
    """
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return {"file": path.name, "chunks": 0, "model": "", "skipped": True}

    model = _get_embedding_model(model)
    text = extract_file_text(path)
    doc_type = path.suffix.lower()
    text = clean_text(text, doc_type)
    chunks = chunk_text(text)

    if not chunks:
        return {"file": path.name, "chunks": 0, "model": model}

    collection = _get_collection(model)

    # Extract section headings and file metadata
    sections = _extract_chunk_sections(text, chunks)
    try:
        file_date = os.path.getmtime(path)
    except OSError:
        file_date = 0.0

    indexed_at = datetime.now(timezone.utc).isoformat()

    ids: list[str] = []
    documents: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict] = []

    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk, model)
        if embedding is None:
            logger.warning(f"Failed to embed chunk {i} of {path.name}, skipping")
            continue

        ids.append(f"{path.name}::chunk_{i}")
        documents.append(chunk)
        embeddings.append(embedding)
        metadatas.append({
            "source": path.name,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "section": sections[i],
            "file_date": file_date,
            "doc_type": doc_type,
            "embedding_model": model,
            "indexed_at": indexed_at,
        })

    if ids:
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        _save_embedding_model(model)
        _bm25_docs.invalidate()

    return {"file": path.name, "chunks": len(ids), "model": model}


def ingest_directory(path: Path | None = None, model: str | None = None) -> list[dict]:
    """Ingest all supported files in a directory recursively.

    Defaults to ~/mycoswarm-docs/ if no path given. Creates the directory
    if it doesn't exist.
    """
    if path is None:
        path = LIBRARY_DIR
    path.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            result = ingest_file(file_path, model)
            results.append(result)

    return results


def search(
    query: str, n_results: int = 5, model: str | None = None
) -> list[dict]:
    """Search the document library for chunks matching a query.

    Returns [{"text": chunk, "source": filename, "score": distance, "chunk_index": i}].
    """
    model = _get_embedding_model(model)

    # Warn on model mismatch
    mismatch_warning = check_embedding_model(model)
    if mismatch_warning:
        print(mismatch_warning)

    query_embedding = embed_text(query, model)
    if query_embedding is None:
        return []

    collection = _get_collection(model)

    # Don't request more results than available
    count = collection.count()
    if count == 0:
        return []
    n = min(n_results, count)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
    )

    hits: list[dict] = []
    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else 0.0
            hits.append({
                "text": doc,
                "source": meta.get("source", ""),
                "score": round(distance, 4),
                "chunk_index": meta.get("chunk_index", 0),
                "section": meta.get("section", "untitled"),
                "doc_type": meta.get("doc_type", ""),
                "file_date": meta.get("file_date", 0.0),
                "embedding_model": meta.get("embedding_model", ""),
            })

    return hits


def list_documents() -> list[dict]:
    """List all indexed documents with chunk counts.

    Returns [{"file": name, "chunks": count}].
    """
    try:
        collection = _get_collection()
    except Exception:
        return []

    all_data = collection.get(include=["metadatas"])
    if not all_data or not all_data["metadatas"]:
        return []

    # Group by source file
    file_chunks: dict[str, int] = {}
    for meta in all_data["metadatas"]:
        source = meta.get("source", "unknown")
        file_chunks[source] = file_chunks.get(source, 0) + 1

    return [{"file": name, "chunks": count} for name, count in sorted(file_chunks.items())]


def remove_document(filename: str) -> bool:
    """Remove all chunks for a document from the collection.

    Returns True if any chunks were deleted.
    """
    collection = _get_collection()

    # Find IDs matching this source file
    all_data = collection.get(include=["metadatas"])
    if not all_data or not all_data["ids"]:
        return False

    ids_to_delete = [
        id_
        for id_, meta in zip(all_data["ids"], all_data["metadatas"])
        if meta.get("source") == filename
    ]

    if not ids_to_delete:
        return False

    collection.delete(ids=ids_to_delete)
    _bm25_docs.invalidate()
    return True


def reindex(model: str | None = None, path: Path | None = None) -> list[dict]:
    """Drop all chunks and re-ingest everything from the docs directory.

    Returns the ingest results list.
    """
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Drop the collection entirely
    try:
        client.delete_collection("mycoswarm_docs")
    except (ValueError, Exception):
        pass  # collection didn't exist

    _bm25_docs.invalidate()

    # Remove stale model file so it gets rewritten on ingest
    try:
        _MODEL_FILE.unlink()
    except FileNotFoundError:
        pass

    return ingest_directory(path, model)


# --- Auto-Update Pipeline ---


def check_stale_documents(
    docs_dir: Path | None = None,
) -> dict[str, list[str]]:
    """Compare files on disk against indexed metadata in ChromaDB.

    Returns {"stale": [...], "new": [...], "removed": [...]}.
    - stale:   files whose mtime is newer than their indexed_at timestamp
    - new:     files on disk that have no chunks in the collection
    - removed: sources in the collection with no file on disk
    """
    if docs_dir is None:
        docs_dir = LIBRARY_DIR
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Files on disk
    disk_files: dict[str, Path] = {}
    for fp in sorted(docs_dir.rglob("*")):
        if fp.is_file() and fp.suffix.lower() in SUPPORTED_EXTENSIONS:
            disk_files[fp.name] = fp

    # Indexed sources and their indexed_at timestamps
    try:
        collection = _get_collection()
        all_data = collection.get(include=["metadatas"])
    except Exception:
        # No collection yet — everything on disk is new
        return {
            "stale": [],
            "new": list(disk_files.keys()),
            "removed": [],
        }

    # Build mapping: source -> latest indexed_at (ISO string)
    indexed: dict[str, str] = {}
    if all_data and all_data.get("metadatas"):
        for meta in all_data["metadatas"]:
            source = meta.get("source", "")
            ts = meta.get("indexed_at", "")
            if source and ts:
                # Keep the latest indexed_at per source
                if source not in indexed or ts > indexed[source]:
                    indexed[source] = ts

    stale: list[str] = []
    new: list[str] = []
    removed: list[str] = []

    # Check each disk file
    for name, fp in disk_files.items():
        if name not in indexed:
            new.append(name)
        else:
            # Compare file mtime against indexed_at
            try:
                mtime = datetime.fromtimestamp(
                    os.path.getmtime(fp), tz=timezone.utc,
                )
                indexed_at = datetime.fromisoformat(indexed[name])
                if mtime > indexed_at:
                    stale.append(name)
            except (OSError, ValueError):
                stale.append(name)

    # Check for removed files (indexed but not on disk)
    for source in indexed:
        if source not in disk_files:
            removed.append(source)

    return {"stale": stale, "new": new, "removed": removed}


def auto_update(
    docs_dir: Path | None = None, model: str | None = None,
) -> dict[str, list[str]]:
    """Detect changed/new/removed files and update the index accordingly.

    Returns {"updated": [...], "added": [...], "removed": [...]}.
    """
    if docs_dir is None:
        docs_dir = LIBRARY_DIR

    changes = check_stale_documents(docs_dir)
    result: dict[str, list[str]] = {"updated": [], "added": [], "removed": []}

    # Re-ingest stale files
    for name in changes["stale"]:
        fp = docs_dir / name
        if fp.exists():
            # Remove old chunks first
            remove_document(name)
            ingest_file(fp, model=model)
            result["updated"].append(name)

    # Ingest new files
    for name in changes["new"]:
        fp = docs_dir / name
        if fp.exists():
            ingest_file(fp, model=model)
            result["added"].append(name)

    # Remove orphaned chunks
    for name in changes["removed"]:
        remove_document(name)
        result["removed"].append(name)

    return result

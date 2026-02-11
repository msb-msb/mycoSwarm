"""mycoSwarm document library with RAG (Retrieval-Augmented Generation).

Local document ingestion, chunking, embedding via Ollama, and vector search
via ChromaDB. Documents are stored in ~/mycoswarm-docs/ and indexed into
~/.config/mycoswarm/library/ for retrieval at query time.

Supports: PDF, TXT, MD, HTML, CSV, JSON.
"""

import json
import logging
import os
import re
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


def index_session_summary(
    session_id: str,
    summary: str,
    date: str,
    model: str | None = None,
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

    collection = _get_session_collection()
    collection.upsert(
        ids=[session_id],
        documents=[summary],
        embeddings=[embedding],
        metadatas=[{
            "session_id": session_id,
            "date": date,
            "topic_keywords": topic_keywords,
            "embedding_model": model,
        }],
    )
    return True


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
                "score": round(distance, 4),
            })

    return hits


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
        })

    if ids:
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        _save_embedding_model(model)

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
    except ValueError:
        pass  # collection didn't exist

    # Remove stale model file so it gets rewritten on ingest
    try:
        _MODEL_FILE.unlink()
    except FileNotFoundError:
        pass

    return ingest_directory(path, model)

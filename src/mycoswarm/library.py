"""mycoSwarm document library with RAG (Retrieval-Augmented Generation).

Local document ingestion, chunking, embedding via Ollama, and vector search
via ChromaDB. Documents are stored in ~/mycoswarm-docs/ and indexed into
~/.config/mycoswarm/library/ for retrieval at query time.

Supports: PDF, TXT, MD, HTML, CSV, JSON.
"""

import logging
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


# --- Public API ---


def ingest_file(path: Path, model: str | None = None) -> dict:
    """Ingest a single file: extract text, chunk, embed, store in ChromaDB.

    Returns {"file": name, "chunks": count, "model": model}.
    """
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return {"file": path.name, "chunks": 0, "model": "", "skipped": True}

    model = _get_embedding_model(model)
    text = extract_file_text(path)
    chunks = chunk_text(text)

    if not chunks:
        return {"file": path.name, "chunks": 0, "model": model}

    collection = _get_collection(model)

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
        })

    if ids:
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

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

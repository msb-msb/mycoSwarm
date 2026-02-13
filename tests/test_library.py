"""Tests for the mycoSwarm document library (RAG)."""

from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mycoswarm.library import (
    chunk_text,
    clean_text,
    extract_file_text,
    embed_text,
    ingest_file,
    index_session_summary,
    reindex_sessions,
    search,
    search_all,
    search_sessions,
    list_documents,
    remove_document,
    reindex,
    rerank,
    check_embedding_model,
    check_stale_documents,
    auto_update,
    _extract_chunk_sections,
    _save_embedding_model,
    _load_embedding_model,
    _rrf_fuse,
    BM25Index,
    _bm25_docs,
    _bm25_sessions,
    _MODEL_FILE,
    EMBEDDING_MODEL,
    SUPPORTED_EXTENSIONS,
)


# --- clean_text ---


class TestCleanText:
    def test_collapses_multiple_newlines(self):
        text = "Hello\n\n\n\n\nWorld"
        result = clean_text(text)
        assert result == "Hello\n\nWorld"

    def test_collapses_multiple_spaces(self):
        text = "Hello    world   here"
        result = clean_text(text)
        assert result == "Hello world here"

    def test_strips_line_whitespace(self):
        text = "  hello  \n  world  "
        result = clean_text(text)
        assert result == "hello\nworld"

    def test_removes_standalone_page_numbers(self):
        text = "Some text\n42\nMore text\n123\nEnd"
        result = clean_text(text)
        assert result == "Some text\nMore text\nEnd"

    def test_removes_page_x_of_y(self):
        text = "Content here\nPage 3 of 10\nMore content"
        result = clean_text(text)
        assert result == "Content here\nMore content"

    def test_removes_copyright_notices(self):
        text = "Good content\n© 2024 Acme Corp. All rights reserved.\nMore content"
        result = clean_text(text)
        assert result == "Good content\nMore content"

    def test_removes_copyright_with_c(self):
        text = "Good content\nCopyright 2024 Acme Corp\nMore content"
        result = clean_text(text)
        assert result == "Good content\nMore content"

    def test_removes_confidential_watermark(self):
        text = "Intro\nConfidential\nReal content\nCONFIDENTIAL\nMore"
        result = clean_text(text)
        assert result == "Intro\nReal content\nMore"

    def test_removes_repeated_headers_footers(self):
        header = "ACME CORP INTERNAL DOCUMENT"
        text = f"{header}\nPage one content\n{header}\nPage two content\n{header}\nPage three"
        result = clean_text(text)
        assert header not in result
        assert "Page one content" in result
        assert "Page two content" in result

    def test_preserves_markdown_headings(self):
        text = "# Introduction\nSome text\n## Details\nMore text"
        result = clean_text(text)
        assert "# Introduction" in result
        assert "## Details" in result

    def test_preserves_repeated_headings(self):
        """Markdown headings should survive even if they appear 3+ times."""
        text = "# Section\nA\n# Section\nB\n# Section\nC"
        result = clean_text(text)
        assert result.count("# Section") == 3

    def test_empty_text(self):
        assert clean_text("") == ""
        assert clean_text("   \n  \n  ") == ""


# --- chunk_text ---


class TestChunkText:
    def test_basic_chunking(self):
        text = " ".join(f"word{i}" for i in range(100))
        chunks = chunk_text(text, chunk_size=30, overlap=5)
        assert len(chunks) > 1
        # First chunk has 30 words
        assert len(chunks[0].split()) == 30

    def test_overlap(self):
        text = " ".join(f"w{i}" for i in range(60))
        chunks = chunk_text(text, chunk_size=30, overlap=10)
        # With 60 words, chunk_size=30, overlap=10:
        # chunk 0: 0..30, chunk 1: 20..50, chunk 2: 40..60
        assert len(chunks) == 3
        # Words at the overlap boundary should appear in both chunks
        words_0 = set(chunks[0].split())
        words_1 = set(chunks[1].split())
        assert len(words_0 & words_1) == 10

    def test_empty_text(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_shorter_than_chunk_size(self):
        text = "hello world foo bar"
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_exact_chunk_size(self):
        text = " ".join(f"w{i}" for i in range(30))
        chunks = chunk_text(text, chunk_size=30, overlap=5)
        assert len(chunks) == 1


# --- _extract_chunk_sections ---


class TestExtractChunkSections:
    def test_headings_mapped_to_chunks(self):
        text = "# Intro\nSome intro text here.\n## Details\nMore details follow."
        chunks = ["Some intro text here.", "More details follow."]
        sections = _extract_chunk_sections(text, chunks)
        assert sections[0] == "Intro"
        assert sections[1] == "Details"

    def test_no_headings(self):
        text = "Just plain text without any headings."
        chunks = ["Just plain text without any headings."]
        sections = _extract_chunk_sections(text, chunks)
        assert sections[0] == "untitled"

    def test_all_chunks_under_one_heading(self):
        words = " ".join(f"w{i}" for i in range(100))
        text = f"# Only Section\n{words}"
        chunks = [" ".join(f"w{i}" for i in range(50)),
                  " ".join(f"w{i}" for i in range(50, 100))]
        sections = _extract_chunk_sections(text, chunks)
        assert sections[0] == "Only Section"
        assert sections[1] == "Only Section"


# --- extract_file_text ---


class TestExtractFileText:
    def test_txt_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello, world!")
        result = extract_file_text(f)
        assert result == "Hello, world!"

    def test_md_file(self, tmp_path):
        f = tmp_path / "readme.md"
        f.write_text("# Title\n\nSome content")
        result = extract_file_text(f)
        assert "Title" in result
        assert "Some content" in result

    def test_json_file(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text('{"key": "value"}')
        result = extract_file_text(f)
        assert "key" in result
        assert "value" in result

    def test_csv_file(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("name,age\nAlice,30\nBob,25")
        result = extract_file_text(f)
        assert "Alice" in result


# --- embed_text ---


class TestEmbedText:
    @patch("mycoswarm.library.httpx.post")
    def test_successful_embedding(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = embed_text("hello world", model="nomic-embed-text")
        assert result == [0.1, 0.2, 0.3]

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["json"]["prompt"] == "hello world"
        assert call_kwargs[1]["json"]["model"] == "nomic-embed-text"

    @patch("mycoswarm.library.httpx.post")
    def test_embedding_failure(self, mock_post):
        import httpx
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        result = embed_text("hello world")
        assert result is None

    @patch("mycoswarm.library.httpx.post")
    def test_empty_embedding(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embedding": []}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = embed_text("hello")
        assert result is None


# --- ingest_file ---


class TestIngestFile:
    @patch("mycoswarm.library._get_collection")
    @patch("mycoswarm.library.embed_text")
    @patch("mycoswarm.library._get_embedding_model")
    def test_ingest_txt_file(self, mock_get_model, mock_embed, mock_collection, tmp_path):
        mock_get_model.return_value = "nomic-embed-text"
        mock_embed.return_value = [0.1, 0.2, 0.3]

        mock_col = MagicMock()
        mock_collection.return_value = mock_col

        f = tmp_path / "test.txt"
        f.write_text("This is a test document with enough words to form a chunk.")

        result = ingest_file(f)
        assert result["file"] == "test.txt"
        assert result["chunks"] >= 1
        assert result["model"] == "nomic-embed-text"
        mock_col.upsert.assert_called_once()

        # Verify metadata fields on every chunk
        upsert_kwargs = mock_col.upsert.call_args[1]
        metadatas = upsert_kwargs["metadatas"]
        for meta in metadatas:
            assert "source" in meta
            assert "section" in meta
            assert "file_date" in meta
            assert "doc_type" in meta
            assert meta["doc_type"] == ".txt"
            assert "embedding_model" in meta
            assert meta["embedding_model"] == "nomic-embed-text"
            assert "chunk_index" in meta

    @patch("mycoswarm.library._get_collection")
    @patch("mycoswarm.library.embed_text")
    @patch("mycoswarm.library._get_embedding_model")
    def test_ingest_md_file_with_sections(self, mock_get_model, mock_embed, mock_collection, tmp_path):
        mock_get_model.return_value = "nomic-embed-text"
        mock_embed.return_value = [0.1, 0.2, 0.3]

        mock_col = MagicMock()
        mock_collection.return_value = mock_col

        f = tmp_path / "guide.md"
        # Write enough content under each heading to form separate chunks
        intro_words = " ".join(f"intro{i}" for i in range(400))
        details_words = " ".join(f"detail{i}" for i in range(400))
        f.write_text(f"# Introduction\n{intro_words}\n## Details\n{details_words}\n")

        result = ingest_file(f)
        assert result["chunks"] >= 2
        assert result["model"] == "nomic-embed-text"

        upsert_kwargs = mock_col.upsert.call_args[1]
        metadatas = upsert_kwargs["metadatas"]
        # First chunk should be under "Introduction", later chunks under "Details"
        assert metadatas[0]["section"] == "Introduction"
        assert metadatas[-1]["section"] == "Details"
        assert metadatas[0]["doc_type"] == ".md"

    @patch("mycoswarm.library._get_embedding_model")
    def test_ingest_unsupported_extension(self, mock_get_model, tmp_path):
        f = tmp_path / "test.xyz"
        f.write_text("content")

        result = ingest_file(f)
        assert result.get("skipped") is True
        assert result["chunks"] == 0
        mock_get_model.assert_not_called()

    @patch("mycoswarm.library._get_collection")
    @patch("mycoswarm.library.embed_text")
    @patch("mycoswarm.library._get_embedding_model")
    def test_ingest_embed_failure(self, mock_get_model, mock_embed, mock_collection, tmp_path):
        mock_get_model.return_value = "nomic-embed-text"
        mock_embed.return_value = None  # all embeddings fail

        f = tmp_path / "test.txt"
        f.write_text("Some content here")

        result = ingest_file(f)
        assert result["chunks"] == 0


# --- search ---


class TestSearch:
    @patch("mycoswarm.library.check_embedding_model", return_value=None)
    @patch("mycoswarm.library._get_collection")
    @patch("mycoswarm.library.embed_text")
    @patch("mycoswarm.library._get_embedding_model")
    def test_search_returns_results(self, mock_get_model, mock_embed, mock_collection, _mock_check):
        mock_get_model.return_value = "nomic-embed-text"
        mock_embed.return_value = [0.1, 0.2, 0.3]

        mock_col = MagicMock()
        mock_col.count.return_value = 5
        mock_col.query.return_value = {
            "documents": [["chunk one text", "chunk two text"]],
            "metadatas": [[
                {"source": "doc.txt", "chunk_index": 0, "section": "Intro",
                 "doc_type": ".txt", "file_date": 1700000000.0, "embedding_model": "nomic-embed-text"},
                {"source": "doc.txt", "chunk_index": 1, "section": "Details",
                 "doc_type": ".txt", "file_date": 1700000000.0, "embedding_model": "nomic-embed-text"},
            ]],
            "distances": [[0.15, 0.25]],
        }
        mock_collection.return_value = mock_col

        results = search("test query")
        assert len(results) == 2
        assert results[0]["source"] == "doc.txt"
        assert results[0]["score"] == 0.15
        assert results[0]["text"] == "chunk one text"
        assert results[0]["section"] == "Intro"
        assert results[0]["doc_type"] == ".txt"
        assert results[0]["embedding_model"] == "nomic-embed-text"
        assert results[1]["section"] == "Details"

    @patch("mycoswarm.library.check_embedding_model", return_value=None)
    @patch("mycoswarm.library.embed_text")
    @patch("mycoswarm.library._get_embedding_model")
    def test_search_no_embedding(self, mock_get_model, mock_embed, _mock_check):
        mock_get_model.return_value = "nomic-embed-text"
        mock_embed.return_value = None

        results = search("test query")
        assert results == []

    @patch("mycoswarm.library.check_embedding_model", return_value=None)
    @patch("mycoswarm.library._get_collection")
    @patch("mycoswarm.library.embed_text")
    @patch("mycoswarm.library._get_embedding_model")
    def test_search_empty_collection(self, mock_get_model, mock_embed, mock_collection, _mock_check):
        mock_get_model.return_value = "nomic-embed-text"
        mock_embed.return_value = [0.1, 0.2]

        mock_col = MagicMock()
        mock_col.count.return_value = 0
        mock_collection.return_value = mock_col

        results = search("test query")
        assert results == []


# --- list_documents ---


class TestListDocuments:
    @patch("mycoswarm.library._get_collection")
    def test_list_groups_by_source(self, mock_collection):
        mock_col = MagicMock()
        mock_col.get.return_value = {
            "metadatas": [
                {"source": "a.txt"},
                {"source": "a.txt"},
                {"source": "b.pdf"},
            ]
        }
        mock_collection.return_value = mock_col

        docs = list_documents()
        assert len(docs) == 2
        a_doc = next(d for d in docs if d["file"] == "a.txt")
        b_doc = next(d for d in docs if d["file"] == "b.pdf")
        assert a_doc["chunks"] == 2
        assert b_doc["chunks"] == 1

    @patch("mycoswarm.library._get_collection")
    def test_list_empty(self, mock_collection):
        mock_col = MagicMock()
        mock_col.get.return_value = {"metadatas": []}
        mock_collection.return_value = mock_col

        docs = list_documents()
        assert docs == []


# --- remove_document ---


class TestRemoveDocument:
    @patch("mycoswarm.library._get_collection")
    def test_remove_existing(self, mock_collection):
        mock_col = MagicMock()
        mock_col.get.return_value = {
            "ids": ["test.txt::chunk_0", "test.txt::chunk_1", "other.txt::chunk_0"],
            "metadatas": [
                {"source": "test.txt"},
                {"source": "test.txt"},
                {"source": "other.txt"},
            ],
        }
        mock_collection.return_value = mock_col

        result = remove_document("test.txt")
        assert result is True
        mock_col.delete.assert_called_once_with(
            ids=["test.txt::chunk_0", "test.txt::chunk_1"]
        )

    @patch("mycoswarm.library._get_collection")
    def test_remove_nonexistent(self, mock_collection):
        mock_col = MagicMock()
        mock_col.get.return_value = {
            "ids": ["other.txt::chunk_0"],
            "metadatas": [{"source": "other.txt"}],
        }
        mock_collection.return_value = mock_col

        result = remove_document("nope.txt")
        assert result is False

    @patch("mycoswarm.library._get_collection")
    def test_remove_empty_collection(self, mock_collection):
        mock_col = MagicMock()
        mock_col.get.return_value = {"ids": [], "metadatas": []}
        mock_collection.return_value = mock_col

        result = remove_document("test.txt")
        assert result is False


# --- embedding model tracking ---


class TestEmbeddingModelTracking:
    def test_save_and_load(self, tmp_path, monkeypatch):
        model_file = tmp_path / "embedding_model.json"
        monkeypatch.setattr("mycoswarm.library._MODEL_FILE", model_file)

        _save_embedding_model("nomic-embed-text")
        assert _load_embedding_model() == "nomic-embed-text"

    def test_load_missing_file(self, tmp_path, monkeypatch):
        model_file = tmp_path / "nonexistent" / "embedding_model.json"
        monkeypatch.setattr("mycoswarm.library._MODEL_FILE", model_file)

        assert _load_embedding_model() is None

    def test_check_no_mismatch(self, tmp_path, monkeypatch):
        model_file = tmp_path / "embedding_model.json"
        monkeypatch.setattr("mycoswarm.library._MODEL_FILE", model_file)

        _save_embedding_model("nomic-embed-text")
        assert check_embedding_model("nomic-embed-text") is None

    def test_check_normalizes_latest_tag(self, tmp_path, monkeypatch):
        """'nomic-embed-text' and 'nomic-embed-text:latest' are the same model."""
        model_file = tmp_path / "embedding_model.json"
        monkeypatch.setattr("mycoswarm.library._MODEL_FILE", model_file)

        # Stored without tag, queried with :latest
        _save_embedding_model("nomic-embed-text")
        assert check_embedding_model("nomic-embed-text:latest") is None

        # Stored with :latest, queried without tag
        _save_embedding_model("nomic-embed-text:latest")
        assert check_embedding_model("nomic-embed-text") is None

        # Both with :latest
        assert check_embedding_model("nomic-embed-text:latest") is None

    def test_check_mismatch_warns(self, tmp_path, monkeypatch):
        model_file = tmp_path / "embedding_model.json"
        monkeypatch.setattr("mycoswarm.library._MODEL_FILE", model_file)

        _save_embedding_model("nomic-embed-text")
        warning = check_embedding_model("mxbai-embed-large")
        assert warning is not None
        assert "nomic-embed-text" in warning
        assert "mxbai-embed-large" in warning
        assert "reindex" in warning

    def test_check_no_stored_model(self, tmp_path, monkeypatch):
        model_file = tmp_path / "nonexistent" / "embedding_model.json"
        monkeypatch.setattr("mycoswarm.library._MODEL_FILE", model_file)

        # No stored model = no warning (first-time use)
        assert check_embedding_model("nomic-embed-text") is None

    @patch("mycoswarm.library.ingest_directory")
    def test_reindex_deletes_collection_and_model_file(self, mock_ingest, tmp_path, monkeypatch):
        import chromadb

        chroma_dir = tmp_path / "chroma"
        chroma_dir.mkdir()
        monkeypatch.setattr("mycoswarm.library.CHROMA_DIR", chroma_dir)

        model_file = chroma_dir / "embedding_model.json"
        monkeypatch.setattr("mycoswarm.library._MODEL_FILE", model_file)
        model_file.write_text('{"embedding_model": "old-model"}')

        # Create a collection so delete_collection has something to drop
        client = chromadb.PersistentClient(path=str(chroma_dir))
        client.get_or_create_collection("mycoswarm_docs")

        mock_ingest.return_value = [{"file": "a.txt", "chunks": 3, "model": "new-model"}]

        results = reindex(model="new-model")
        assert len(results) == 1
        assert not model_file.exists()
        mock_ingest.assert_called_once()


# --- session memory ---


class TestSessionMemory:
    @patch("mycoswarm.library._get_session_collection")
    @patch("mycoswarm.library.embed_text")
    @patch("mycoswarm.library._get_embedding_model")
    def test_index_session_summary(self, mock_get_model, mock_embed, mock_collection):
        mock_get_model.return_value = "nomic-embed-text"
        mock_embed.return_value = [0.1, 0.2, 0.3]
        mock_col = MagicMock()
        mock_collection.return_value = mock_col

        result = index_session_summary(
            session_id="chat-20260210",
            summary="Discussed GPU clustering and distributed inference",
            date="2026-02-10",
        )
        assert result is True
        mock_col.upsert.assert_called_once()

        upsert_kwargs = mock_col.upsert.call_args[1]
        assert upsert_kwargs["ids"] == ["chat-20260210"]
        assert "GPU clustering" in upsert_kwargs["documents"][0]
        meta = upsert_kwargs["metadatas"][0]
        assert meta["session_id"] == "chat-20260210"
        assert meta["date"] == "2026-02-10"
        assert "topic_keywords" in meta
        assert "embedding_model" in meta

    @patch("mycoswarm.library.embed_text")
    @patch("mycoswarm.library._get_embedding_model")
    def test_index_session_embed_failure(self, mock_get_model, mock_embed):
        mock_get_model.return_value = "nomic-embed-text"
        mock_embed.return_value = None

        result = index_session_summary("s1", "some summary", "2026-02-10")
        assert result is False

    @patch("mycoswarm.library._get_session_collection")
    @patch("mycoswarm.library.embed_text")
    @patch("mycoswarm.library._get_embedding_model")
    def test_search_sessions_returns_hits(self, mock_get_model, mock_embed, mock_collection):
        mock_get_model.return_value = "nomic-embed-text"
        mock_embed.return_value = [0.1, 0.2, 0.3]

        mock_col = MagicMock()
        mock_col.count.return_value = 2
        mock_col.query.return_value = {
            "documents": [["Discussed GPUs", "Talked about RAG"]],
            "metadatas": [[
                {"session_id": "s1", "date": "2026-02-08"},
                {"session_id": "s2", "date": "2026-02-09"},
            ]],
            "distances": [[0.1, 0.3]],
        }
        mock_collection.return_value = mock_col

        hits = search_sessions("GPU inference")
        assert len(hits) == 2
        assert hits[0]["summary"] == "Discussed GPUs"
        assert hits[0]["session_id"] == "s1"
        assert hits[0]["date"] == "2026-02-08"
        assert hits[0]["score"] == 0.1

    @patch("mycoswarm.library._get_session_collection")
    @patch("mycoswarm.library.embed_text")
    @patch("mycoswarm.library._get_embedding_model")
    def test_search_sessions_empty(self, mock_get_model, mock_embed, mock_collection):
        mock_get_model.return_value = "nomic-embed-text"
        mock_embed.return_value = [0.1, 0.2]

        mock_col = MagicMock()
        mock_col.count.return_value = 0
        mock_collection.return_value = mock_col

        hits = search_sessions("anything")
        assert hits == []


# --- reindex_sessions ---


class TestReindexSessions:
    @patch("mycoswarm.library.index_session_summary")
    @patch("mycoswarm.library._get_embedding_model")
    def test_reindex_splits_and_indexes(self, mock_get_model, mock_index, tmp_path, monkeypatch):
        """reindex_sessions reads JSONL, splits topics, indexes each chunk."""
        import json as _json
        import chromadb

        mock_get_model.return_value = "nomic-embed-text"
        mock_index.return_value = True

        # Set up temp ChromaDB + sessions.jsonl
        chroma_dir = tmp_path / "chroma"
        chroma_dir.mkdir()
        monkeypatch.setattr("mycoswarm.library.CHROMA_DIR", chroma_dir)

        sessions_path = tmp_path / "sessions.jsonl"
        entries = [
            {"session_name": "s1", "model": "gemma3:27b", "timestamp": "2026-02-09T10:00:00", "summary": "Talked about bees", "message_count": 5},
            {"session_name": "s2", "model": "gemma3:27b", "timestamp": "2026-02-10T12:00:00", "summary": "Discussed crypto and tai chi", "message_count": 8},
        ]
        sessions_path.write_text("\n".join(_json.dumps(e) for e in entries) + "\n")

        from mycoswarm import memory
        monkeypatch.setattr(memory, "SESSIONS_PATH", sessions_path)

        # Mock split: s1 returns 1 topic, s2 returns 2
        def fake_split(summary, model):
            if "bees" in summary:
                return [{"topic": "bees", "summary": "Talked about bees"}]
            return [
                {"topic": "crypto", "summary": "Discussed crypto"},
                {"topic": "tai chi", "summary": "Discussed tai chi"},
            ]
        monkeypatch.setattr("mycoswarm.memory.split_session_topics", fake_split)

        stats = reindex_sessions(model="nomic-embed-text")

        assert stats["sessions"] == 2
        assert stats["topics"] == 3
        assert stats["failed"] == 0

        # Verify the 3 index calls
        assert mock_index.call_count == 3
        calls = [c[1] for c in mock_index.call_args_list]  # kwargs
        assert calls[0]["session_id"] == "s1::topic_0"
        assert calls[0]["topic"] == "bees"
        assert calls[0]["date"] == "2026-02-09"
        assert calls[1]["session_id"] == "s2::topic_0"
        assert calls[1]["topic"] == "crypto"
        assert calls[2]["session_id"] == "s2::topic_1"
        assert calls[2]["topic"] == "tai chi"

    @patch("mycoswarm.library.index_session_summary")
    @patch("mycoswarm.library._get_embedding_model")
    def test_reindex_drops_collection(self, mock_get_model, mock_index, tmp_path, monkeypatch):
        """reindex_sessions drops the existing session_memory collection."""
        import chromadb

        mock_get_model.return_value = "nomic-embed-text"
        mock_index.return_value = True

        chroma_dir = tmp_path / "chroma"
        chroma_dir.mkdir()
        monkeypatch.setattr("mycoswarm.library.CHROMA_DIR", chroma_dir)

        # Create a pre-existing collection with stale data
        client = chromadb.PersistentClient(path=str(chroma_dir))
        col = client.get_or_create_collection("session_memory")
        col.add(ids=["stale"], documents=["old data"], metadatas=[{"session_id": "stale"}])
        assert col.count() == 1

        # Empty JSONL
        sessions_path = tmp_path / "sessions.jsonl"
        sessions_path.write_text("")
        from mycoswarm import memory
        monkeypatch.setattr(memory, "SESSIONS_PATH", sessions_path)
        monkeypatch.setattr("mycoswarm.memory.split_session_topics", lambda s, m: [{"topic": "general", "summary": s}])

        stats = reindex_sessions()

        # Collection was dropped and recreated empty
        client2 = chromadb.PersistentClient(path=str(chroma_dir))
        col2 = client2.get_or_create_collection("session_memory")
        assert col2.count() == 0
        assert stats["topics"] == 0


# --- search_all ---


class TestSearchAll:
    @patch("mycoswarm.library.check_embedding_model", return_value=None)
    @patch("mycoswarm.library._bm25_sessions")
    @patch("mycoswarm.library._bm25_docs")
    @patch("mycoswarm.library._get_session_collection")
    @patch("mycoswarm.library._get_collection")
    @patch("mycoswarm.library.embed_text")
    @patch("mycoswarm.library._get_embedding_model")
    def test_returns_both_doc_and_session_hits(
        self, mock_get_model, mock_embed, mock_doc_col, mock_sess_col,
        mock_bm25_docs, mock_bm25_sess, _mock_check,
    ):
        mock_get_model.return_value = "nomic-embed-text"
        mock_embed.return_value = [0.1, 0.2, 0.3]

        # Document collection
        doc_col = MagicMock()
        doc_col.count.return_value = 2
        doc_col.query.return_value = {
            "ids": [["health.pdf::chunk_0"]],
            "documents": [["chunk about tai chi benefits"]],
            "metadatas": [[{
                "source": "health.pdf", "chunk_index": 0, "section": "Tai Chi",
                "doc_type": ".pdf", "file_date": 1700000000.0,
                "embedding_model": "nomic-embed-text",
            }]],
            "distances": [[0.12]],
        }
        mock_doc_col.return_value = doc_col
        mock_bm25_docs.search.return_value = []

        # Session collection
        sess_col = MagicMock()
        sess_col.count.return_value = 3
        sess_col.query.return_value = {
            "ids": [["chat-20260211::topic_0"]],
            "documents": [["Discussed tai chi for ADHD management"]],
            "metadatas": [[{
                "session_id": "chat-20260211::topic_0",
                "date": "2026-02-11",
                "topic": "tai chi ADHD",
            }]],
            "distances": [[0.08]],
        }
        mock_sess_col.return_value = sess_col
        mock_bm25_sess.search.return_value = []

        doc_hits, session_hits = search_all("tai chi ADHD")

        assert len(doc_hits) == 1
        assert doc_hits[0]["source"] == "health.pdf"
        assert doc_hits[0]["text"] == "chunk about tai chi benefits"

        assert len(session_hits) == 1
        assert session_hits[0]["summary"] == "Discussed tai chi for ADHD management"
        assert session_hits[0]["date"] == "2026-02-11"
        assert session_hits[0]["topic"] == "tai chi ADHD"

    @patch("mycoswarm.library.check_embedding_model", return_value=None)
    @patch("mycoswarm.library._bm25_sessions")
    @patch("mycoswarm.library._bm25_docs")
    @patch("mycoswarm.library._get_session_collection")
    @patch("mycoswarm.library._get_collection")
    @patch("mycoswarm.library.embed_text")
    @patch("mycoswarm.library._get_embedding_model")
    def test_returns_sessions_even_when_no_docs(
        self, mock_get_model, mock_embed, mock_doc_col, mock_sess_col,
        mock_bm25_docs, mock_bm25_sess, _mock_check,
    ):
        mock_get_model.return_value = "nomic-embed-text"
        mock_embed.return_value = [0.1, 0.2]

        # Empty document collection
        doc_col = MagicMock()
        doc_col.count.return_value = 0
        mock_doc_col.return_value = doc_col
        mock_bm25_docs.search.return_value = []

        # Session collection with hits
        sess_col = MagicMock()
        sess_col.count.return_value = 1
        sess_col.query.return_value = {
            "ids": [["s1"]],
            "documents": [["We talked about meditation"]],
            "metadatas": [[{"session_id": "s1", "date": "2026-02-10", "topic": "meditation"}]],
            "distances": [[0.15]],
        }
        mock_sess_col.return_value = sess_col
        mock_bm25_sess.search.return_value = []

        doc_hits, session_hits = search_all("meditation")

        assert len(doc_hits) == 0
        assert len(session_hits) == 1
        assert session_hits[0]["summary"] == "We talked about meditation"

    @patch("mycoswarm.library.check_embedding_model", return_value=None)
    @patch("mycoswarm.library.embed_text")
    @patch("mycoswarm.library._get_embedding_model")
    def test_returns_empty_when_no_embedding(self, mock_get_model, mock_embed, _mock_check):
        mock_get_model.return_value = "nomic-embed-text"
        mock_embed.return_value = None

        doc_hits, session_hits = search_all("anything")
        assert doc_hits == []
        assert session_hits == []

    @patch("mycoswarm.library.check_embedding_model", return_value=None)
    @patch("mycoswarm.library._bm25_sessions")
    @patch("mycoswarm.library._bm25_docs")
    @patch("mycoswarm.library._get_session_collection")
    @patch("mycoswarm.library._get_collection")
    @patch("mycoswarm.library.embed_text")
    @patch("mycoswarm.library._get_embedding_model")
    def test_embeds_query_only_once(
        self, mock_get_model, mock_embed, mock_doc_col, mock_sess_col,
        mock_bm25_docs, mock_bm25_sess, _mock_check,
    ):
        """search_all should call embed_text exactly once, not twice."""
        mock_get_model.return_value = "nomic-embed-text"
        mock_embed.return_value = [0.1, 0.2]

        doc_col = MagicMock()
        doc_col.count.return_value = 0
        mock_doc_col.return_value = doc_col
        mock_bm25_docs.search.return_value = []

        sess_col = MagicMock()
        sess_col.count.return_value = 0
        mock_sess_col.return_value = sess_col
        mock_bm25_sess.search.return_value = []

        search_all("test query")

        assert mock_embed.call_count == 1


# --- BM25Index ---


class TestBM25Index:
    @patch("mycoswarm.library._get_collection")
    def test_builds_from_collection_documents(self, mock_get_col):
        """BM25 index should build from all documents in a collection."""
        mock_col = MagicMock()
        mock_col.get.return_value = {
            "ids": ["doc1::chunk_0", "doc1::chunk_1", "doc2::chunk_0"],
            "documents": [
                "the quick brown fox jumps over the lazy dog",
                "lorem ipsum dolor sit amet consectetur",
                "the fox ran quickly across the field",
            ],
            "metadatas": [
                {"source": "doc1.txt", "chunk_index": 0},
                {"source": "doc1.txt", "chunk_index": 1},
                {"source": "doc2.txt", "chunk_index": 0},
            ],
        }
        mock_get_col.return_value = mock_col

        idx = BM25Index("mycoswarm_docs")
        idx._build()

        assert idx._built is True
        assert len(idx._doc_ids) == 3
        assert len(idx._documents) == 3
        assert idx._index is not None

    @patch("mycoswarm.library._get_collection")
    def test_search_returns_ranked_results(self, mock_get_col):
        """BM25 search should return results ranked by keyword relevance."""
        mock_col = MagicMock()
        mock_col.get.return_value = {
            "ids": ["a::0", "b::0", "c::0"],
            "documents": [
                "machine learning algorithms for classification",
                "the weather is sunny and warm today",
                "deep learning and machine learning models",
            ],
            "metadatas": [
                {"source": "a.txt"}, {"source": "b.txt"}, {"source": "c.txt"},
            ],
        }
        mock_get_col.return_value = mock_col

        idx = BM25Index("mycoswarm_docs")
        results = idx.search("machine learning", n_results=3)

        assert len(results) >= 2
        # The two ML docs should rank above the weather doc
        result_ids = [r["id"] for r in results]
        assert "a::0" in result_ids
        assert "c::0" in result_ids
        # Weather doc either absent or ranked last
        if "b::0" in result_ids:
            assert result_ids.index("b::0") == len(result_ids) - 1

    @patch("mycoswarm.library._get_collection")
    def test_invalidate_forces_rebuild(self, mock_get_col):
        """After invalidate(), next search should re-fetch and rebuild."""
        mock_col = MagicMock()
        mock_col.get.return_value = {
            "ids": ["x::0"],
            "documents": ["original content about bees"],
            "metadatas": [{"source": "x.txt"}],
        }
        mock_get_col.return_value = mock_col

        idx = BM25Index("mycoswarm_docs")
        results = idx.search("bees", n_results=5)
        assert len(results) == 1
        assert idx._built is True

        # Invalidate
        idx.invalidate()
        assert idx._built is False

        # Update mock to return different data
        mock_col.get.return_value = {
            "ids": ["x::0", "y::0"],
            "documents": [
                "original content about bees",
                "new content about honey bees",
            ],
            "metadatas": [{"source": "x.txt"}, {"source": "y.txt"}],
        }

        results = idx.search("bees", n_results=5)
        assert len(results) == 2
        assert idx._built is True

    def test_empty_collection(self):
        """BM25 search on empty collection returns empty list."""
        idx = BM25Index("mycoswarm_docs")
        # Force build with no data
        idx._built = True
        idx._index = None
        idx._doc_ids = []

        results = idx.search("anything", n_results=5)
        assert results == []


# --- _rrf_fuse ---


class TestRRFFuse:
    def test_merges_scores_correctly(self):
        """RRF should compute 1/(k+rank) for each list and sum."""
        vec_ids = ["a", "b", "c"]
        bm25_ids = ["b", "d", "a"]

        scores = _rrf_fuse(vec_ids, bm25_ids, k=60)

        # "a": vec rank 1 + bm25 rank 3 = 1/61 + 1/63
        expected_a = 1.0 / 61 + 1.0 / 63
        assert abs(scores["a"] - expected_a) < 1e-10

        # "b": vec rank 2 + bm25 rank 1 = 1/62 + 1/61
        expected_b = 1.0 / 62 + 1.0 / 61
        assert abs(scores["b"] - expected_b) < 1e-10

        # "c": only vec rank 3 = 1/63
        expected_c = 1.0 / 63
        assert abs(scores["c"] - expected_c) < 1e-10

        # "d": only bm25 rank 2 = 1/62
        expected_d = 1.0 / 62
        assert abs(scores["d"] - expected_d) < 1e-10

    def test_items_in_both_lists_score_higher(self):
        """Documents appearing in both lists should have higher RRF scores."""
        vec_ids = ["shared", "vec_only"]
        bm25_ids = ["shared", "bm25_only"]

        scores = _rrf_fuse(vec_ids, bm25_ids, k=60)

        # "shared" appears in both at rank 1 -> highest score
        assert scores["shared"] > scores["vec_only"]
        assert scores["shared"] > scores["bm25_only"]

    def test_empty_lists(self):
        """RRF with empty lists returns empty dict."""
        assert _rrf_fuse([], []) == {}

    def test_single_list(self):
        """RRF with one empty list degrades to single-list ranking."""
        scores = _rrf_fuse(["a", "b"], [], k=60)
        assert len(scores) == 2
        assert scores["a"] > scores["b"]


# --- Hybrid search_all ---


class TestHybridSearchAll:
    @patch("mycoswarm.library.check_embedding_model", return_value=None)
    @patch("mycoswarm.library._bm25_sessions")
    @patch("mycoswarm.library._bm25_docs")
    @patch("mycoswarm.library._get_session_collection")
    @patch("mycoswarm.library._get_collection")
    @patch("mycoswarm.library.embed_text")
    @patch("mycoswarm.library._get_embedding_model")
    def test_hybrid_returns_bm25_only_hits(
        self, mock_get_model, mock_embed, mock_doc_col, mock_sess_col,
        mock_bm25_docs, mock_bm25_sess, _mock_check,
    ):
        """BM25-only hits (exact keyword match missed by vector) appear in results."""
        mock_get_model.return_value = "nomic-embed-text"
        mock_embed.return_value = [0.1, 0.2, 0.3]

        # Vector search returns only doc A
        doc_col = MagicMock()
        doc_col.count.return_value = 3
        doc_col.query.return_value = {
            "ids": [["a::0"]],
            "documents": [["semantic match about neural networks"]],
            "metadatas": [[{"source": "a.txt", "chunk_index": 0, "section": "Intro",
                           "doc_type": ".txt", "file_date": 0.0, "embedding_model": "nomic"}]],
            "distances": [[0.1]],
        }
        mock_doc_col.return_value = doc_col

        # BM25 returns doc A (also) and doc B (keyword-only hit)
        mock_bm25_docs.search.return_value = [
            {"id": "a::0", "document": "semantic match about neural networks",
             "metadata": {"source": "a.txt", "chunk_index": 0, "section": "Intro",
                         "doc_type": ".txt", "file_date": 0.0, "embedding_model": "nomic"},
             "bm25_score": 2.5},
            {"id": "b::0", "document": "the exact keyword xylophone appears here",
             "metadata": {"source": "b.txt", "chunk_index": 0, "section": "untitled",
                         "doc_type": ".txt", "file_date": 0.0, "embedding_model": "nomic"},
             "bm25_score": 1.8},
        ]

        # Empty session collection
        sess_col = MagicMock()
        sess_col.count.return_value = 0
        mock_sess_col.return_value = sess_col
        mock_bm25_sess.search.return_value = []

        doc_hits, session_hits = search_all("xylophone neural networks", n_results=5)

        # Both docs should appear
        sources = [h["source"] for h in doc_hits]
        assert "a.txt" in sources, "Vector hit should be present"
        assert "b.txt" in sources, "BM25-only hit should be present"

        # doc A should rank higher (in both lists)
        a_idx = sources.index("a.txt")
        b_idx = sources.index("b.txt")
        assert a_idx < b_idx

    @patch("mycoswarm.library.check_embedding_model", return_value=None)
    @patch("mycoswarm.library._bm25_sessions")
    @patch("mycoswarm.library._bm25_docs")
    @patch("mycoswarm.library._get_session_collection")
    @patch("mycoswarm.library._get_collection")
    @patch("mycoswarm.library.embed_text")
    @patch("mycoswarm.library._get_embedding_model")
    def test_rrf_score_in_results(
        self, mock_get_model, mock_embed, mock_doc_col, mock_sess_col,
        mock_bm25_docs, mock_bm25_sess, _mock_check,
    ):
        """Results should include rrf_score field."""
        mock_get_model.return_value = "nomic-embed-text"
        mock_embed.return_value = [0.1, 0.2]

        doc_col = MagicMock()
        doc_col.count.return_value = 1
        doc_col.query.return_value = {
            "ids": [["x::0"]],
            "documents": [["test document"]],
            "metadatas": [[{"source": "x.txt", "chunk_index": 0, "section": "untitled",
                           "doc_type": ".txt", "file_date": 0.0, "embedding_model": "nomic"}]],
            "distances": [[0.2]],
        }
        mock_doc_col.return_value = doc_col
        mock_bm25_docs.search.return_value = []

        sess_col = MagicMock()
        sess_col.count.return_value = 0
        mock_sess_col.return_value = sess_col
        mock_bm25_sess.search.return_value = []

        doc_hits, _ = search_all("test", n_results=5)

        assert len(doc_hits) == 1
        assert "rrf_score" in doc_hits[0]
        assert doc_hits[0]["rrf_score"] > 0

    @patch("mycoswarm.library.check_embedding_model", return_value=None)
    @patch("mycoswarm.library._bm25_sessions")
    @patch("mycoswarm.library._bm25_docs")
    @patch("mycoswarm.library._get_session_collection")
    @patch("mycoswarm.library._get_collection")
    @patch("mycoswarm.library.embed_text")
    @patch("mycoswarm.library._get_embedding_model")
    def test_hybrid_session_search(
        self, mock_get_model, mock_embed, mock_doc_col, mock_sess_col,
        mock_bm25_docs, mock_bm25_sess, _mock_check,
    ):
        """Hybrid search works for session memory collection too."""
        mock_get_model.return_value = "nomic-embed-text"
        mock_embed.return_value = [0.1, 0.2]

        # Empty doc collection
        doc_col = MagicMock()
        doc_col.count.return_value = 0
        mock_doc_col.return_value = doc_col
        mock_bm25_docs.search.return_value = []

        # Session vector returns one hit
        sess_col = MagicMock()
        sess_col.count.return_value = 2
        sess_col.query.return_value = {
            "ids": [["s1::topic_0"]],
            "documents": [["Discussed beekeeping techniques"]],
            "metadatas": [[{"session_id": "s1::topic_0", "date": "2026-02-10", "topic": "beekeeping"}]],
            "distances": [[0.15]],
        }
        mock_sess_col.return_value = sess_col

        # BM25 returns same + another session hit
        mock_bm25_sess.search.return_value = [
            {"id": "s1::topic_0", "document": "Discussed beekeeping techniques",
             "metadata": {"session_id": "s1::topic_0", "date": "2026-02-10", "topic": "beekeeping"},
             "bm25_score": 3.0},
            {"id": "s2::topic_0", "document": "Bees and honey production overview",
             "metadata": {"session_id": "s2::topic_0", "date": "2026-02-11", "topic": "bees"},
             "bm25_score": 1.5},
        ]

        _, session_hits = search_all("beekeeping", n_results=5)

        assert len(session_hits) == 2
        summaries = [h["summary"] for h in session_hits]
        assert "Discussed beekeeping techniques" in summaries
        assert "Bees and honey production overview" in summaries
        # First hit (in both lists) should have higher rrf_score
        assert session_hits[0]["rrf_score"] > session_hits[1]["rrf_score"]


# --- check_stale_documents ---


class TestCheckStaleDocuments:
    def test_stale_file_detected(self, tmp_path, monkeypatch):
        """File with mtime newer than indexed_at is stale."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "notes.txt").write_text("hello world")

        # indexed_at is in the past
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()

        mock_col = MagicMock()
        mock_col.get.return_value = {
            "metadatas": [
                {"source": "notes.txt", "indexed_at": old_ts},
            ],
        }

        monkeypatch.setattr("mycoswarm.library.LIBRARY_DIR", docs_dir)
        with patch("mycoswarm.library._get_collection", return_value=mock_col):
            result = check_stale_documents(docs_dir)

        assert "notes.txt" in result["stale"]
        assert result["new"] == []
        assert result["removed"] == []

    def test_new_file_detected(self, tmp_path, monkeypatch):
        """File on disk with no indexed chunks is new."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "new_doc.md").write_text("# New")

        mock_col = MagicMock()
        mock_col.get.return_value = {"metadatas": []}

        monkeypatch.setattr("mycoswarm.library.LIBRARY_DIR", docs_dir)
        with patch("mycoswarm.library._get_collection", return_value=mock_col):
            result = check_stale_documents(docs_dir)

        assert "new_doc.md" in result["new"]
        assert result["stale"] == []
        assert result["removed"] == []

    def test_removed_file_detected(self, tmp_path, monkeypatch):
        """Source in collection with no file on disk is removed."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        # No files on disk

        now_ts = datetime.now(timezone.utc).isoformat()
        mock_col = MagicMock()
        mock_col.get.return_value = {
            "metadatas": [
                {"source": "deleted.txt", "indexed_at": now_ts},
            ],
        }

        monkeypatch.setattr("mycoswarm.library.LIBRARY_DIR", docs_dir)
        with patch("mycoswarm.library._get_collection", return_value=mock_col):
            result = check_stale_documents(docs_dir)

        assert "deleted.txt" in result["removed"]
        assert result["stale"] == []
        assert result["new"] == []

    def test_up_to_date_file(self, tmp_path, monkeypatch):
        """File indexed after its mtime should not be stale."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "current.txt").write_text("content")

        # indexed_at is in the future relative to the file
        future_ts = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        mock_col = MagicMock()
        mock_col.get.return_value = {
            "metadatas": [
                {"source": "current.txt", "indexed_at": future_ts},
            ],
        }

        monkeypatch.setattr("mycoswarm.library.LIBRARY_DIR", docs_dir)
        with patch("mycoswarm.library._get_collection", return_value=mock_col):
            result = check_stale_documents(docs_dir)

        assert result["stale"] == []
        assert result["new"] == []
        assert result["removed"] == []


# --- auto_update ---


class TestAutoUpdate:
    @patch("mycoswarm.library.ingest_file")
    @patch("mycoswarm.library.remove_document")
    @patch("mycoswarm.library.check_stale_documents")
    def test_reingest_stale_files(self, mock_check, mock_remove, mock_ingest, tmp_path):
        """auto_update re-ingests stale files."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "stale.txt").write_text("updated content")

        mock_check.return_value = {"stale": ["stale.txt"], "new": [], "removed": []}
        mock_ingest.return_value = {"file": "stale.txt", "chunks": 2, "model": "nomic"}

        result = auto_update(docs_dir)

        assert result["updated"] == ["stale.txt"]
        assert result["added"] == []
        assert result["removed"] == []
        mock_remove.assert_called_once_with("stale.txt")
        mock_ingest.assert_called_once()

    @patch("mycoswarm.library.ingest_file")
    @patch("mycoswarm.library.check_stale_documents")
    def test_adds_new_files(self, mock_check, mock_ingest, tmp_path):
        """auto_update ingests new files."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "fresh.md").write_text("# Fresh")

        mock_check.return_value = {"stale": [], "new": ["fresh.md"], "removed": []}
        mock_ingest.return_value = {"file": "fresh.md", "chunks": 1, "model": "nomic"}

        result = auto_update(docs_dir)

        assert result["added"] == ["fresh.md"]
        assert result["updated"] == []
        mock_ingest.assert_called_once()

    @patch("mycoswarm.library.remove_document")
    @patch("mycoswarm.library.check_stale_documents")
    def test_removes_orphaned_chunks(self, mock_check, mock_remove):
        """auto_update removes chunks for files no longer on disk."""
        mock_check.return_value = {"stale": [], "new": [], "removed": ["gone.pdf"]}

        result = auto_update()

        assert result["removed"] == ["gone.pdf"]
        assert result["updated"] == []
        assert result["added"] == []
        mock_remove.assert_called_once_with("gone.pdf")


# --- rerank ---


class TestRerank:
    @patch("mycoswarm.library._score_chunk")
    def test_sorts_by_llm_relevance(self, mock_score):
        """rerank() should sort chunks by LLM-assigned score descending."""
        # Assign scores: chunk C=9, chunk A=7, chunk B=3
        mock_score.side_effect = [7.0, 3.0, 9.0]

        chunks = [
            {"text": "chunk A about cats", "source": "a.txt"},
            {"text": "chunk B about weather", "source": "b.txt"},
            {"text": "chunk C about machine learning", "source": "c.txt"},
        ]

        result = rerank("machine learning", chunks, llm_model="test-model", top_k=3)

        assert len(result) == 3
        assert result[0]["source"] == "c.txt"  # score 9
        assert result[0]["rerank_score"] == 9.0
        assert result[1]["source"] == "a.txt"  # score 7
        assert result[2]["source"] == "b.txt"  # score 3

    @patch("mycoswarm.library._score_chunk")
    def test_top_k_truncation(self, mock_score):
        """rerank() returns at most top_k results."""
        mock_score.side_effect = [8.0, 6.0, 4.0, 2.0]

        chunks = [
            {"text": f"chunk {i}", "source": f"{i}.txt"}
            for i in range(4)
        ]

        result = rerank("query", chunks, llm_model="test-model", top_k=2)
        assert len(result) == 2

    @patch("mycoswarm.library._score_chunk")
    def test_handles_parse_failures_gracefully(self, mock_score):
        """Parse failures default to score 5, don't crash."""
        # First chunk gets a real score, second gets default 5
        mock_score.side_effect = [9.0, 5.0]

        chunks = [
            {"text": "relevant text", "source": "a.txt"},
            {"text": "ambiguous text", "source": "b.txt"},
        ]

        result = rerank("query", chunks, llm_model="test-model", top_k=2)

        assert len(result) == 2
        assert result[0]["rerank_score"] == 9.0
        assert result[1]["rerank_score"] == 5.0

    def test_empty_chunks(self):
        """rerank() with empty list returns empty."""
        assert rerank("query", [], llm_model="test-model") == []

    @patch("mycoswarm.library._pick_rerank_model", return_value=None)
    def test_no_llm_returns_truncated(self, _mock_pick):
        """When no LLM available, return first top_k chunks unchanged."""
        chunks = [
            {"text": f"chunk {i}", "source": f"{i}.txt"}
            for i in range(5)
        ]

        result = rerank("query", chunks, top_k=3)
        assert len(result) == 3
        assert result[0]["source"] == "0.txt"

    @patch("mycoswarm.library._score_chunk")
    def test_uses_summary_key_for_sessions(self, mock_score):
        """rerank() with text_key='summary' reads the summary field."""
        mock_score.return_value = 8.0

        chunks = [{"summary": "Discussed beekeeping", "session_id": "s1"}]
        result = rerank("bees", chunks, llm_model="test", top_k=1, text_key="summary")

        assert len(result) == 1
        # Verify _score_chunk was called with the summary text
        mock_score.assert_called_once_with("bees", "Discussed beekeeping", "test")


# --- search_all with rerank ---


class TestSearchAllRerank:
    @patch("mycoswarm.library.rerank")
    @patch("mycoswarm.library.check_embedding_model", return_value=None)
    @patch("mycoswarm.library._bm25_sessions")
    @patch("mycoswarm.library._bm25_docs")
    @patch("mycoswarm.library._get_session_collection")
    @patch("mycoswarm.library._get_collection")
    @patch("mycoswarm.library.embed_text")
    @patch("mycoswarm.library._get_embedding_model")
    def test_rerank_true_calls_rerank(
        self, mock_get_model, mock_embed, mock_doc_col, mock_sess_col,
        mock_bm25_docs, mock_bm25_sess, _mock_check, mock_rerank,
    ):
        """search_all with do_rerank=True should call rerank()."""
        mock_get_model.return_value = "nomic-embed-text"
        mock_embed.return_value = [0.1, 0.2]

        doc_col = MagicMock()
        doc_col.count.return_value = 3
        doc_col.query.return_value = {
            "ids": [["a::0", "b::0"]],
            "documents": [["doc A", "doc B"]],
            "metadatas": [[
                {"source": "a.txt", "chunk_index": 0, "section": "u",
                 "doc_type": ".txt", "file_date": 0.0, "embedding_model": "n"},
                {"source": "b.txt", "chunk_index": 0, "section": "u",
                 "doc_type": ".txt", "file_date": 0.0, "embedding_model": "n"},
            ]],
            "distances": [[0.1, 0.2]],
        }
        mock_doc_col.return_value = doc_col
        mock_bm25_docs.search.return_value = []

        sess_col = MagicMock()
        sess_col.count.return_value = 0
        mock_sess_col.return_value = sess_col
        mock_bm25_sess.search.return_value = []

        # rerank returns filtered results
        mock_rerank.side_effect = lambda q, chunks, **kw: chunks[:1]

        doc_hits, _ = search_all("test", n_results=5, do_rerank=True)

        assert mock_rerank.call_count == 2  # once for docs, once for sessions

    @patch("mycoswarm.library.rerank")
    @patch("mycoswarm.library.check_embedding_model", return_value=None)
    @patch("mycoswarm.library._bm25_sessions")
    @patch("mycoswarm.library._bm25_docs")
    @patch("mycoswarm.library._get_session_collection")
    @patch("mycoswarm.library._get_collection")
    @patch("mycoswarm.library.embed_text")
    @patch("mycoswarm.library._get_embedding_model")
    def test_rerank_false_skips_rerank(
        self, mock_get_model, mock_embed, mock_doc_col, mock_sess_col,
        mock_bm25_docs, mock_bm25_sess, _mock_check, mock_rerank,
    ):
        """search_all with do_rerank=False should NOT call rerank()."""
        mock_get_model.return_value = "nomic-embed-text"
        mock_embed.return_value = [0.1, 0.2]

        doc_col = MagicMock()
        doc_col.count.return_value = 1
        doc_col.query.return_value = {
            "ids": [["a::0"]],
            "documents": [["doc A"]],
            "metadatas": [[{"source": "a.txt", "chunk_index": 0, "section": "u",
                           "doc_type": ".txt", "file_date": 0.0, "embedding_model": "n"}]],
            "distances": [[0.1]],
        }
        mock_doc_col.return_value = doc_col
        mock_bm25_docs.search.return_value = []

        sess_col = MagicMock()
        sess_col.count.return_value = 0
        mock_sess_col.return_value = sess_col
        mock_bm25_sess.search.return_value = []

        search_all("test", n_results=5, do_rerank=False)

        mock_rerank.assert_not_called()

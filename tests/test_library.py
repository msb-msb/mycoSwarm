"""Tests for the mycoSwarm document library (RAG)."""

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
    search,
    search_sessions,
    list_documents,
    remove_document,
    reindex,
    check_embedding_model,
    _extract_chunk_sections,
    _save_embedding_model,
    _load_embedding_model,
    _MODEL_FILE,
    EMBEDDING_MODEL,
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

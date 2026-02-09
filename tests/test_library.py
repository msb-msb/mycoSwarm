"""Tests for the mycoSwarm document library (RAG)."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mycoswarm.library import (
    chunk_text,
    extract_file_text,
    embed_text,
    ingest_file,
    search,
    list_documents,
    remove_document,
    EMBEDDING_MODEL,
)


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
    @patch("mycoswarm.library._get_collection")
    @patch("mycoswarm.library.embed_text")
    @patch("mycoswarm.library._get_embedding_model")
    def test_search_returns_results(self, mock_get_model, mock_embed, mock_collection):
        mock_get_model.return_value = "nomic-embed-text"
        mock_embed.return_value = [0.1, 0.2, 0.3]

        mock_col = MagicMock()
        mock_col.count.return_value = 5
        mock_col.query.return_value = {
            "documents": [["chunk one text", "chunk two text"]],
            "metadatas": [[
                {"source": "doc.txt", "chunk_index": 0},
                {"source": "doc.txt", "chunk_index": 1},
            ]],
            "distances": [[0.15, 0.25]],
        }
        mock_collection.return_value = mock_col

        results = search("test query")
        assert len(results) == 2
        assert results[0]["source"] == "doc.txt"
        assert results[0]["score"] == 0.15
        assert results[0]["text"] == "chunk one text"

    @patch("mycoswarm.library.embed_text")
    @patch("mycoswarm.library._get_embedding_model")
    def test_search_no_embedding(self, mock_get_model, mock_embed):
        mock_get_model.return_value = "nomic-embed-text"
        mock_embed.return_value = None

        results = search("test query")
        assert results == []

    @patch("mycoswarm.library._get_collection")
    @patch("mycoswarm.library.embed_text")
    @patch("mycoswarm.library._get_embedding_model")
    def test_search_empty_collection(self, mock_get_model, mock_embed, mock_collection):
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

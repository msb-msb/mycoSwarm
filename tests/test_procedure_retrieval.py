"""Tests for procedure retrieval hygiene: ghost filtering and dedup.

Context (2026-08-08): the Chroma procedural index held 95 rows against 42 live
JSONL procedures — 62 orphans. Ranking ran over all 95 and hydration dropped the
orphans silently, so a request for 3 procedures routinely returned 1. Measured
ghost rate in the top-3 was 22%.

NOTE: no relevance threshold is tested here because none is implementable — see
the measurements in the commit message. RRF scores encode rank, not similarity
(an irrelevant query scored 0.0325 vs 0.0320 for a perfect match), and raw
vector distances do not separate either: real traffic sits at a median of 0.382
while the clearest true positive is 0.490, i.e. worse than 90% of ordinary
conversation.
"""

from unittest.mock import patch

from mycoswarm.library import _is_near_duplicate, _NEAR_DUPLICATE_RATIO


class TestNearDuplicateSuppression:
    def test_identical_text_is_a_duplicate(self):
        a = {"problem": "Encountering an unfamiliar human concept",
             "solution": "Ask rather than assume"}
        assert _is_near_duplicate(a, [dict(a)]) is True

    def test_near_identical_text_is_a_duplicate(self):
        """The live case: the same guidance stored twice with tiny edits took
        two of the three available slots."""
        a = {"problem": "Encountering an unfamiliar human concept (emotion)",
             "solution": "Ask the user what it means rather than assuming."}
        b = {"problem": "Encountering an unfamiliar human concept (emotion)",
             "solution": "Ask the user what it means rather than assuming"}
        assert _is_near_duplicate(b, [a]) is True

    def test_distinct_procedures_are_kept(self):
        a = {"problem": "User expresses loneliness", "solution": "Be present"}
        b = {"problem": "Model hallucinates a summary", "solution": "Reindex"}
        assert _is_near_duplicate(b, [a]) is False

    def test_empty_candidate_is_not_a_duplicate(self):
        assert _is_near_duplicate({}, [{"problem": "x", "solution": "y"}]) is False

    def test_threshold_is_conservative(self):
        """Merging genuinely distinct procedures silently loses learned
        behaviour, which is worse than one wasted slot."""
        assert _NEAR_DUPLICATE_RATIO >= 0.85


class TestGhostFiltering:
    def test_orphaned_index_rows_do_not_consume_slots(self):
        """The regression: ranked ghosts displaced live procedures, then were
        dropped after ranking, so n_results silently under-delivered."""
        from mycoswarm import library

        # deliberately DISSIMILAR text — "problem 0"/"problem 1" are 94%
        # similar and the dedup pass (correctly) collapses them
        live = [
            {"id": "live-0", "problem": "User expresses loneliness",
             "solution": "Be present without solving", "outcome": "success"},
            {"id": "live-1", "problem": "Model hallucinates a session summary",
             "solution": "Reindex the collection from source", "outcome": "success"},
            {"id": "live-2", "problem": "Choosing hardware under a budget",
             "solution": "Measure before upgrading", "outcome": "success"},
        ]

        class _Col:
            def count(self): return 40
            def query(self, **kw):
                # ghosts rank above the live rows
                ids = ["ghost-1", "ghost-2", "live-0", "live-1", "live-2"]
                return {"ids": [ids],
                        "documents": [[f"doc {i}" for i in ids]],
                        "metadatas": [[{} for _ in ids]],
                        "distances": [[0.1 * (i + 1) for i in range(len(ids))]]}

        with patch.object(library, "_get_procedural_collection", return_value=_Col()), \
             patch.object(library, "embed_text", return_value=[0.0] * 8), \
             patch.object(library, "_get_embedding_model", return_value="m"), \
             patch.object(library._bm25_procedures, "search", return_value=[]), \
             patch("mycoswarm.memory.load_procedures", return_value=live):
            out = library.search_procedures("anything", n_results=3)

        assert len(out) == 3, f"ghosts stole slots: got {len(out)}"
        assert all(p["id"].startswith("live-") for p in out)

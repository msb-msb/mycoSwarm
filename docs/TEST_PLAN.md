# mycoSwarm Smoke Test & Large RAG Test Plan

## Overview

Two test layers beyond unit tests (`pytest`):

1. **Smoke tests** — fast pipeline validation (~2 min)
2. **Book tests** — large-scale RAG stress testing (~5 min)

## Smoke Test Suite

Location: `tests/smoke/`

| Script | What it tests | Duration |
|--------|--------------|----------|
| `smoke_intent.sh` | Intent classification accuracy, deterministic overrides | ~30s |
| `smoke_rag.sh` | RAG grounding — model uses real doc content, not hallucinations | ~60s |
| `smoke_memory.sh` | Source priority, section boost, grounding scores, contradiction detection | ~10s |
| `smoke_poison.sh` | Poison resistance — low grounding entries blocked from index | ~30s |
| `smoke_swarm.sh` | Distributed intent classification across peers | ~20s |
| `smoke_book.sh` | Large RAG with book-length PDFs (run separately) | ~120s |

### Running

```bash
# Full suite (excludes book test)
bash tests/smoke/run_all.sh

# Individual test
bash tests/smoke/smoke_rag.sh

# Book test (requires prior ingestion)
mycoswarm library ingest /path/to/book.pdf
bash tests/smoke/smoke_book.sh "book.pdf"
```

## Large RAG / Book Ingestion Test Plan

### Why Books?

Our current test corpus is 3 small markdown files (~400 lines total, 10-25 chunks).
This leaves major blind spots:

| Dimension | Small corpus | Book corpus |
|-----------|-------------|-------------|
| Chunks | 10-25 | 100-500+ |
| RRF ranking pool | trivial | competitive |
| Section detection | markdown headers | chapter titles, subheadings |
| Chunk boundaries | clean paragraph breaks | mid-paragraph, footnotes |
| Context density | sparse technical notes | dense prose |
| Cross-topic confusion | minimal overlap | many related concepts |

### Recommended Test Books

| Book | Why | Expected chunks |
|------|-----|----------------|
| Wu Wei: Effortless Living (Gregory) | Philosophy, dense prose, Mark's domain expertise → can verify accuracy | ~150-200 |
| Any CC-licensed AI/ML textbook | Technical content, equations, code blocks | ~200-400 |
| Project Gutenberg classic (e.g. Tao Te Ching) | Free, short, verifiable content | ~30-50 |

### What Book Tests Expose

#### 1. Chunking Quality
- **Symptom**: Chunk starts mid-sentence, splits key concept across boundaries
- **How we detect**: `smoke_book.sh` Test 6 checks mid-sentence starts and short chunks
- **Fix path**: Improve chunker to respect paragraph and section boundaries
- **Phase 20 lesson**: "Phase 20" header was in one chunk, content in another

#### 2. Section Header Detection at Scale
- **Symptom**: All chunks labeled "untitled" — section boost can't fire
- **How we detect**: `smoke_book.sh` Test 2 counts unique vs untitled sections
- **Fix path**: PDF parser needs to detect chapter headings, not just markdown `##`

#### 3. RRF Ranking Under Competition
- **Symptom**: Relevant chunk buried below 5th position among 200 candidates
- **How we detect**: `smoke_book.sh` Test 3 checks score differentiation
- **Fix path**: May need to increase n_candidates, improve BM25 tokenization,
  or add TF-IDF boost for rare terms

#### 4. Context Window Pressure
- **Symptom**: 5 book chunks = 15K+ tokens, exceeding effective context
- **How we detect**: `smoke_book.sh` Test 4 estimates token count
- **Fix path**: Reduce chunk size for dense content, or reduce n_results to 3

#### 5. Cross-Topic Confusion
- **Symptom**: "What does the book say about Wu Wei?" returns chunks about
  unrelated chapters that happen to mention the term
- **How we detect**: Manual review of retrieval results with --debug
- **Fix path**: Better section-aware retrieval, chapter filtering (like source_filter)

#### 6. Grounding Under Noise
- **Symptom**: Model receives 5 dense chunks but generates from training data
- **How we detect**: `smoke_book.sh` Test 5 checks for citation markers
- **Fix path**: Stronger grounding prompt, reduce context to 3 chunks,
  put most relevant chunk first (already done via section boost)

### Test Execution Protocol

```bash
# Step 1: Ingest the book
mycoswarm library ingest ~/books/wu_wei_effortless_living.pdf

# Step 2: Verify ingestion
mycoswarm library search --query "effortless action Wu Wei"

# Step 3: Run book smoke test
bash tests/smoke/smoke_book.sh "wu_wei_effortless_living.pdf"

# Step 4: Manual verification with --debug
echo "what does the book say about non-action?" | \
    mycoswarm chat --debug --session book-manual-test

# Step 5: Compare against known content
# Ask questions you know the answers to from reading the book.
# This is the ultimate grounding test — can the system give you
# correct answers that you can verify?
```

### Metrics to Track

| Metric | Target | How to measure |
|--------|--------|---------------|
| Chunk count | 100-500 for a full book | `library.py _get_collection().count()` |
| Section detection rate | >50% non-untitled | smoke_book Test 2 |
| Top-1 relevance | Correct chunk #1 for specific queries | Manual with --debug |
| Citation rate | >80% of responses include [D1] etc | smoke_book Test 5 |
| Hallucination rate | <10% of responses contain fabricated claims | Manual verification |
| Context token estimate | <6000 tokens for 5 chunks | smoke_book Test 4 |

### Progressive Test Ladder

Start small, scale up:

1. **Tao Te Ching** (Gutenberg, ~5000 words) — baseline PDF ingestion
2. **Wu Wei: Effortless Living** (~50K words) — medium scale, verifiable
3. **Technical AI textbook** (~200K words) — stress test at scale

Each level exposes new issues. Fix them before moving to the next.

## Integration with CI

Future: Run smoke tests on every release:

```yaml
# .github/workflows/smoke.yml
- name: Run smoke tests
  run: |
    pip install mycoswarm
    bash tests/smoke/run_all.sh
```

Book tests remain manual due to PDF dependencies and model requirements.

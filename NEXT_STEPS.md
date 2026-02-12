# Next Steps: Litigation Document Pipeline

## Current Status (As of 2026-02-11)

### ✅ Completed Phases

**Phase 1: Citation Foundation**
- [x] Citation tracker with bbox-based line inference
- [x] PyMuPDF extractor for exact line-by-line deposition extraction
- [x] Bates stamp extraction from Docling JSON page footers
- [x] Format-specific handlers (depositions, patents, expert reports)

**Phase 2: Core Pipeline**
- [x] Post-processor with text cleaning
- [x] Footnote inline insertion for all legal documents (expert reports, pleadings, briefs, court opinions)
- [x] Text markers ([TEXT:N]) for citation linkage
- [x] Section-aware chunking preserving Q/A pairs
- [x] Context card generation with complete citation metadata
- [x] JSON cleanup (--cleanup-json flag, default on)

**Phase 3: Vector Search & Retrieval**
- [x] BM25 keyword search (`bm25_indexer.py`) — TF-IDF with BM25 scoring, <10ms queries
- [x] Semantic vector store (`vector_indexer.py`) — Chroma + Ollama nomic-embed-text
- [x] Hybrid search with RRF score fusion (`hybrid_retriever.py`) — three modes: bm25, semantic, hybrid
- [x] Search CLI (`lit_doc_retriever.py`) — build-index, search, stats commands
- [x] Graceful degradation to BM25-only when Chroma unavailable

**Phase 4: Cross-Encoder Reranker**
- [x] `reranker.py` — lazy-loaded `cross-encoder/ms-marco-MiniLM-L-6-v2` via sentence-transformers
- [x] Integrated into `hybrid_retriever.py` — `search(rerank=True)` fetches 10x candidates, reranks to top-k
- [x] CLI `--rerank` flag in `lit_doc_retriever.py`
- [x] Graceful degradation when sentence-transformers not installed
- [x] `reranker_score` field added to `SearchResult`

**Phase 5: LLM Enrichment**
- [x] `llm_enrichment.py` — Three backends: Ollama, Anthropic, Claude Code
- [x] Quote validation: only exact substrings kept (discards hallucinations per TRD 9.4)
- [x] Category/relevance validation with sensible defaults (14 valid categories)
- [x] Claims filtering to reject patent numbers (>100)
- [x] CLI integration: `--enrich`, `--enrich-backend`, `--case-type`, `--parties` flags
- [x] Retriever displays enrichment metadata in search results
- [x] Backup creation on first enrichment run
- [x] `/enrich-chunks` skill for Claude Code interactive use

**Test Coverage:** 153 total (137 passing, 16 skipped) | **Search Relevance:** 100% (known doc in top-5)

> **Note — ChromaDB dependency:** Currently using an interim build from ChromaDB's git main branch
> to work around a Pydantic v1 compatibility issue with Python 3.14. This should be replaced with
> ChromaDB >=1.5.1 (or next stable release) once it ships with Python 3.14 support.

---

## 🔧 Improvements & Refinements

### Citation Enhancements
- [x] ~~Improve paragraph number extraction for expert reports~~
  - [x] **COMPLETE:** Now 99.2% coverage (was 0%)
  - [x] Added support for numbered paragraph format "N. "
  - [x] Cole Report: 654/659 citations now have paragraph numbers
  - [x] Preserves existing ¶/§ symbol and "Paragraph N" patterns

- [x] ~~Bates stamp validation~~
  - [x] **COMPLETE:** Sequential numbering validation implemented
  - [x] Flags gaps in Bates sequences
  - [x] Detects duplicate Bates stamps on same page
  - [x] Logs warnings when issues detected

- [x] ~~Column number extraction for patents~~
  - [x] **COMPLETE:** Already exceeds goal at 84.5% (was misreported as 12.6%)
  - [x] Spec pages (24-34): 174/206 column citations (84.5%)
  - [x] Figure pages (1-23): Correctly classified as patent_figure
  - [x] Overall 12.6% is accurate but includes figure pages

### Chunking Improvements
- [ ] Claim-aware chunking for patents
  - [ ] Detect claim boundaries (CLAIM 1, CLAIM 2, etc.)
  - [ ] Preserve dependent/independent claim structure
  - [ ] Track claim numbers in citation metadata

- [ ] Intelligent overlap for expert reports
  - [ ] Currently no overlap (paragraphs are self-contained)
  - [ ] Consider sentence-level overlap for long paragraphs
  - [ ] Preserve cross-references between paragraphs

- [ ] Chunk quality metrics
  - [ ] Average tokens per chunk
  - [ ] Citation coverage per chunk
  - [ ] Q/A pair completeness for depositions

### Performance Optimization
- [ ] Parallel document processing
  - [ ] Process multiple PDFs concurrently
  - [ ] PyMuPDF is already fast (<1 sec per deposition)
  - [ ] Parallelize Docling OCR for scanned docs

- [ ] Incremental indexing
  - [ ] Track processed documents
  - [ ] Only reindex changed files
  - [ ] Support document updates/additions

- [ ] Timeout handling for large scanned documents
  - [ ] INTEL_PROX_00002382 times out at 5 minutes
  - [ ] Implement page-range chunking for OCR
  - [ ] Process in 50-page batches

---

## 📊 Quality Assurance

### Validation Suite
- [x] ~~End-to-end integration test~~
  - [x] Run full pipeline on 5/6 test documents (1 timeout on 300+ page doc)
  - [x] Verify chunk count, citation coverage
  - [x] Test search relevance on known queries

- [x] ~~Regression tests~~ (`tests/test_regression.py` — 38 tests, all passing)
  - [x] No garbage text from images (no base64, no embedded data in placeholders)
  - [x] Page numbers preserved correctly (100% across depositions, expert reports, patents)
  - [x] Line numbers accurate (100% for Alexander deposition, all lines 1-25, start <= end)
  - [x] Footnotes included in relevant chunks (8/15 Cole chunks contain footnotes, correct format)
  - [x] Citation coverage thresholds (3916 deposition, 659 expert report, 1376 patent)
  - [x] Chunk structure integrity (unique IDs, sequential numbering, correct doc_types)

- [x] ~~Search quality metrics~~
  - [x] **COMPLETE:** `benchmark.py` measures Precision@5 for 20 known queries
  - [x] BM25-only: 98.0% mean Precision@5, 100% hit rate, <1ms latency
  - [x] BM25+Rerank: 98.0% mean Precision@5, 100% hit rate, ~443ms latency
  - [x] Reranker adds no precision benefit on small corpus (56 chunks) but maintains quality
  - [x] Reranker improved 1 query (+0.20), degraded 1 query (-0.20), no change on 18

### Documentation
- [x] ~~Update README.md with usage examples~~
- [x] ~~Create ARCHITECTURE.md documenting pipeline design~~
- [ ] Document citation linkage system ([TEXT:N] markers)
- [ ] Add examples of chunk output format

---

## 🚀 Production Readiness

### CLI Interface
- [x] ~~Create main CLI entry point~~ (`lit_pipeline.py`)
  - [x] `lit-pipeline process <input_dir>` - Run full pipeline
  - [x] `lit-pipeline search <query>` - Search indexed documents
  - [x] `lit-pipeline stats <index_dir>` - Show index statistics
  - [x] `lit-pipeline index <dir>` - Build search indexes
  - [x] `lit-pipeline enrich <dir>` - Standalone LLM enrichment

### Configuration
- [x] ~~Support config files for pipeline parameters~~ (`config_loader.py`)
  - [x] Chunk size targets per document type
  - [x] Bates pattern customization
  - [x] JSON/YAML support with intelligent config file search order

### Error Handling
- [x] Graceful degradation for missing dependencies (ChromaDB, sentence-transformers)
- [x] Better timeout handling for long OCR jobs (configurable --conversion-timeout)
- [x] Recovery from partial pipeline failures (checkpoint/resume with .lit-pipeline-state.json)
- [x] Try-catch blocks around each pipeline stage
- [x] Automatic retry limiting (max 3 attempts per document)
- [x] --resume, --force, --no-skip-failed flags

---

## 📈 Next Immediate Actions

**Production Polish** ✅ **COMPLETE**
1. [x] Unified CLI entry point (`lit-pipeline` command with 5 subcommands)
2. [x] Config file support (JSON/YAML with CLI override)
3. [x] ARCHITECTURE.md documentation (comprehensive technical design)
4. [x] End-to-end integration tests (12 new tests)

**Priority 1: Benchmark & Optimization** ✅ **COMPLETE**
1. [x] Installed sentence-transformers 5.2.2 — cross-encoder `ms-marco-MiniLM-L-6-v2` loads and works
2. [x] Measured Precision@5: BM25=98.0%, BM25+Rerank=98.0% (no improvement on 56-chunk corpus)
3. [x] Measured rerank latency: mean 443ms, P95 601ms for 50 candidates (within <1s target)
4. [ ] Profile enrichment performance (Ollama vs Anthropic) — deferred, requires Ollama running

**Benchmark Findings:**
- BM25 alone achieves 98.0% Precision@5 on the current 56-chunk test corpus
- Cross-encoder reranking maintains precision but adds ~443ms latency overhead
- Reranker value will increase with larger, more diverse corpora where keyword matching is noisier
- Full results: `benchmark.py` (20 queries, per-query comparison)

**Priority 2: Quality Assurance** ✅ **COMPLETE**
1. [x] Run full pipeline on 5/6 test documents (1 timeout on 300+ page doc)
2. [x] Verify chunk count, citation coverage — `test_regression.py` (38 tests all passing)
3. [x] Test search relevance on known queries (100% hit rate, 98% precision)
4. [ ] Measure enrichment quality (quote validation rate, category accuracy) — deferred, requires LLM

**Regression Test Results (38 tests):**
- No garbage text: 6 tests (base64, image placeholders, chunk content)
- Page numbers: 6 tests (all doc types have 100% page coverage)
- Line numbers: 4 tests (100% coverage, valid range 1-25, correct format)
- Footnotes: 4 tests (8/15 Cole chunks include footnotes, proper format)
- Citation coverage: 4 tests (3916 deposition, 659 expert, 1376 patent)
- Chunk integrity: 14 tests (unique IDs, required fields, correct doc types)

**Next Priority: Chunking Improvements & Performance**

### Maintenance
- [ ] Replace ChromaDB git build with stable >=1.5.1 when available

---

## 🎯 Success Metrics

### Current Achievements
- ✅ **Citation Accuracy:** 100% for text-based depositions, 99.8% Bates coverage
- ✅ **Processing Speed:** <1 second for text depositions, <5 min per 100 pages for OCR
- ✅ **Storage Efficiency:** 76% reduction with JSON cleanup
- ✅ **Test Coverage:** 153 tests (137 passing, 16 skipped)
- ✅ **BM25 Search Latency:** <1ms per query (56-chunk corpus)
- ✅ **BM25 Index Build:** 0.04s for 56 chunks
- ✅ **Search Relevance:** 100% hit rate (known doc in top-5)
- ✅ **Cross-Encoder Reranker:** Installed and benchmarked (sentence-transformers 5.2.2)
- ✅ **LLM Enrichment:** Three backends (Ollama/Anthropic/Claude Code) with quote validation

### Benchmark Results (20 queries, 56 chunks)
| Metric | BM25-only | BM25+Rerank |
|--------|-----------|-------------|
| Mean Precision@5 | 98.0% | 98.0% |
| Hit Rate | 100% | 100% |
| Mean Latency | <1ms | 443ms |
| P95 Latency | 1.2ms | 601ms |

> **Note:** Reranker adds no measurable precision improvement on this small, focused corpus.
> Its value is expected to increase with larger, more diverse document sets where BM25
> keyword matching produces noisier results.

---

## 📝 Notes

**Document Types Tested:**
- ✅ Text-based deposition (Alexander) - Perfect extraction
- ✅ Expert report (Cole Report) - Footnotes inline, 99.2% paragraph coverage
- ✅ Patent (INTEL_PROX_00006214) - Column detection 84.5% on spec pages
- ✅ IEEE standards (INTEL_PROX_00001770, etc.) - OCR working
- ⏸️ Long scanned documents (INTEL_PROX_00002382) - Timeout issue

**Known Limitations:**
- ~~Paragraph number extraction: 0% coverage~~ → **FIXED:** 99.2% coverage
- ~~Column detection: 12.6% coverage~~ → **FIXED:** 84.5% on spec pages
- OCR timeout: Documents >200 pages with dense scans (INTEL_PROX_00002382)
- Bates stamps: Only extracted if present in page footers
- ChromaDB: Using interim git build until stable Python 3.14 support ships

**Architecture Decisions:**
- PostProcessor handles content transformation (footnotes, text markers)
- CitationTracker handles geometric metadata extraction (bbox → lines)
- Chunking reads markdown, not JSON (better for maintainability)
- Text markers provide lightweight linkage without bloating markdown
- Hybrid search uses Reciprocal Rank Fusion (RRF) instead of weighted score fusion
- Lazy imports for ChromaDB and sentence-transformers to allow graceful degradation

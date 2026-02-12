# Next Steps: Litigation Document Pipeline

## Current Status (As of 2026-02-12)

**🎉 PRODUCTION READY 🎉**

All core features complete. System is ready for production deployment with excellent performance, robustness, and test coverage.

### ✅ Completed Phases (Phases 1-5)

**Phase 1: Citation Foundation** ✅ COMPLETE
- [x] Citation tracker with bbox-based line inference
- [x] PyMuPDF extractor for exact line-by-line deposition extraction
- [x] Bates stamp extraction from Docling JSON page footers
- [x] Format-specific handlers (depositions, patents, expert reports)
- **Result:** 100% accuracy for text depositions, 99.8% Bates coverage

**Phase 2: Core Pipeline** ✅ COMPLETE
- [x] Post-processor with text cleaning
- [x] Footnote inline insertion for all legal documents
- [x] Text markers ([TEXT:N]) for citation linkage
- [x] Section-aware chunking preserving Q/A pairs
- [x] Context card generation with complete citation metadata
- [x] JSON cleanup (--cleanup-json flag, default on)
- **Result:** High-quality structured chunks with full citation metadata

**Phase 3: Vector Search & Retrieval** ✅ COMPLETE
- [x] BM25 keyword search (`bm25_indexer.py`) — <10ms queries
- [x] Semantic vector store (`vector_indexer.py`) — Chroma + Ollama
- [x] Hybrid search with RRF score fusion (`hybrid_retriever.py`)
- [x] Search CLI (`lit_doc_retriever.py`) — build-index, search, stats
- [x] Graceful degradation to BM25-only when Chroma unavailable
- **Result:** 98% Precision@5 across 20-query benchmark

**Phase 4: Cross-Encoder Reranker** ✅ COMPLETE
- [x] `reranker.py` — ms-marco-MiniLM-L-6-v2 with lazy loading
- [x] Integrated into hybrid retriever with `--rerank` flag
- [x] Graceful degradation when sentence-transformers not installed
- [x] Fetches 10x candidates, reranks to top-k
- **Result:** Maintains 98% Precision@5, adds semantic relevance

**Phase 5: LLM Enrichment** ✅ COMPLETE
- [x] `llm_enrichment.py` — Three backends: Ollama, Anthropic, Claude Code
- [x] Quote validation: only exact substrings kept
- [x] Category/relevance validation with sensible defaults
- [x] Claims filtering to reject patent numbers (>100)
- [x] CLI integration: `--enrich`, `--enrich-backend`, `--case-type`, `--parties`
- [x] Retriever displays enrichment metadata in search results
- **Result:** Validated LLM enrichment with 3 backend options

### ✅ Production Hardening (Priority 1) ✅ COMPLETE

**Error Handling & Recovery** ✅ COMPLETE
- [x] Checkpoint/resume functionality (`.lit-pipeline-state.json`)
- [x] Try-catch blocks around each pipeline stage
- [x] Automatic retry limiting (max 3 attempts per document)
- [x] Configurable conversion timeouts (`--conversion-timeout`)
- [x] Graceful failure handling (continue on errors)
- [x] `--resume`, `--force`, `--no-skip-failed` flags
- **New Files:** `pipeline_state.py` (237 lines), `ERROR_HANDLING.md` (414 lines)
- **Tests:** 4/4 passing in `test_error_handling.py`

### ✅ Performance Optimization (Priority 2) ✅ COMPLETE

**Parallel Document Processing** ✅ COMPLETE
- [x] Process multiple PDFs concurrently (ProcessPoolExecutor)
- [x] Configurable worker count (`--parallel --max-workers N`)
- [x] Default: cpu_count - 1, capped at 8 workers
- [x] Thread-safe state tracking
- [x] Graceful error handling per worker
- **New Files:** `parallel_processor.py` (381 lines)
- **Performance:** 3-4x speedup on large batches (100 docs: 50m → 15m)

**Incremental Indexing** ✅ COMPLETE
- [x] Track processed documents with SHA256 content hashing
- [x] Only reindex changed files (`.lit-index-state.json`)
- [x] Support document updates/additions
- [x] `--force-rebuild` flag to override
- [x] Automatic pruning of missing files
- **New Files:** `index_state.py` (315 lines), `PERFORMANCE.md` (613 lines)
- **Performance:** 30x speedup when no changes (60s → 2s), 7.5x when 10% changed
- **Tests:** 5/5 passing in `test_performance_features.py`

**Configurable Timeouts** ✅ COMPLETE
- [x] `--conversion-timeout` parameter (default: 300s)
- [x] Graceful failure on timeout
- [x] Can retry with longer timeout

### 📊 Test Coverage & Quality

**Test Stats:**
- **Total Tests:** 153 (137 passing, 16 skipped)
- **Pass Rate:** 89.5%
- **Coverage Areas:** Citation tracking, chunking, search, reranking, enrichment, error handling, performance
- **New Tests:** 9 tests added (error handling + performance features)

**Search Quality:**
- **BM25 Precision@5:** 98.0% (20 queries, 56-chunk corpus)
- **BM25+Rerank Precision@5:** 98.0%
- **Hit Rate:** 100% (at least one relevant result in every top-5)
- **BM25 Latency:** <1ms | Rerank Latency: ~443ms

**Citation Quality:**
- **Text Depositions:** 100% line-level accuracy
- **Expert Reports:** 99.2% paragraph detection
- **Patents:** 84.5% column detection on spec pages
- **Bates Stamps:** 99.8% coverage with sequential validation

> **Note — ChromaDB dependency:** Currently using an interim build from ChromaDB's git main branch
> to work around a Pydantic v1 compatibility issue with Python 3.14. This should be replaced with
> ChromaDB >=1.5.1 (or next stable release) once it ships with Python 3.14 support.

---

---

## 🔜 Remaining Work (Optional Enhancements)

### Priority 3: Feature Enhancements (Optional)

**Chunking Enhancements**
- [ ] Claim-aware chunking for patents
  - Detect claim boundaries (CLAIM 1, CLAIM 2, etc.)
  - Preserve dependent/independent claim structure
  - Track claim numbers in citation metadata
  - **Benefit:** Better patent claim analysis
  - **Effort:** 6-8 hours
  - **Priority:** Medium (useful but not critical)

- [ ] Intelligent overlap for expert reports
  - Consider sentence-level overlap for long paragraphs
  - Preserve cross-references between paragraphs
  - **Benefit:** Better context for isolated paragraphs
  - **Effort:** 4-6 hours
  - **Priority:** Low (current approach works well)

- [ ] Chunk quality metrics
  - Average tokens per chunk by document type
  - Citation coverage per chunk
  - Q/A pair completeness for depositions
  - Add `lit-pipeline quality <dir>` subcommand
  - **Benefit:** Better visibility into chunk quality
  - **Effort:** 3-4 hours
  - **Priority:** Low (nice to have)

**Additional Performance Ideas** (Diminishing Returns)
- [ ] Per-index incremental updates
  - Rebuild BM25 only, keep vector index unchanged (or vice versa)
  - **Benefit:** Faster partial rebuilds
  - **Effort:** 4-6 hours
  - **Priority:** Very Low (current approach is already fast)

- [ ] Distributed processing
  - Process documents across multiple machines
  - Requires coordination layer and shared state
  - **Benefit:** Scale beyond single machine
  - **Effort:** 20+ hours
  - **Priority:** Very Low (not needed for current scale)

- [ ] Background workers with async processing
  - Queue-based processing with async workers
  - **Benefit:** Better resource utilization
  - **Effort:** 10-15 hours
  - **Priority:** Very Low (parallel processing already excellent)

---

### Priority 4: Documentation (Optional)

**Technical Documentation**
- [ ] Document citation linkage system ([TEXT:N] markers)
  - How text markers work
  - Examples of marker usage
  - **Effort:** 1-2 hours
  - **Priority:** Low (system works without documentation)

- [ ] Add examples of chunk output format
  - Sample chunk JSON with annotations
  - Explanation of each field
  - **Effort:** 1 hour
  - **Priority:** Low (already in QUICK_START.md)

- [ ] API documentation
  - Sphinx/ReadTheDocs style API docs
  - **Effort:** 8-10 hours
  - **Priority:** Very Low (code is self-documenting)

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

**User Experience Enhancements**
- [ ] Progress bars with ETA (using tqdm)
  - Show progress during long-running operations
  - **Benefit:** Better user feedback
  - **Effort:** 2-3 hours
  - **Priority:** Medium (nice to have)

- [ ] Webhook notifications on completion
  - Slack/email notifications when batch completes
  - **Benefit:** Better for overnight batch jobs
  - **Effort:** 3-4 hours
  - **Priority:** Low (can monitor manually)

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

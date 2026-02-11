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
- [x] Footnote inline insertion for expert reports
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

**Test Coverage:** 109 total (93 passing, 16 skipped) | **Search Relevance:** 100% (known doc in top-5)

> **Note — ChromaDB dependency:** Currently using an interim build from ChromaDB's git main branch
> to work around a Pydantic v1 compatibility issue with Python 3.14. This should be replaced with
> ChromaDB >=1.5.1 (or next stable release) once it ships with Python 3.14 support.

---

## 🔧 Improvements & Refinements

### Citation Enhancements
- [ ] Improve paragraph number extraction for expert reports
  - [ ] Current coverage: 0% (not detecting ¶ markers)
  - [ ] Parse numbered paragraphs from markdown
  - [ ] Update citation strings to include paragraph ranges

- [ ] Bates stamp validation
  - [ ] Verify sequential numbering
  - [ ] Flag gaps in Bates sequences
  - [ ] Handle multi-document Bates ranges

- [ ] Column number extraction for patents
  - [ ] Current coverage: 12.6% (174/1,376 citations)
  - [ ] Improve column detection algorithm
  - [ ] Handle inline column citations (e.g., "as described in col. 3...")

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
- [ ] End-to-end integration test
  - [ ] Run full pipeline on all 6 test documents
  - [ ] Verify chunk count, citation coverage
  - [ ] Test search relevance on known queries

- [ ] Regression tests
  - [ ] No garbage text from images (verify with --image-export-mode placeholder)
  - [ ] Page numbers preserved correctly
  - [ ] Line numbers accurate (100% for text-based depositions)
  - [ ] Footnotes included in relevant chunks

- [ ] Search quality metrics
  - [ ] Precision@5 for known queries
  - [ ] Recall@10 for document coverage
  - [ ] Citation accuracy (verify citation strings match source docs)

### Documentation
- [ ] Update README.md with usage examples
- [ ] Create ARCHITECTURE.md documenting pipeline design
- [ ] Document citation linkage system ([TEXT:N] markers)
- [ ] Add examples of chunk output format

---

## 🚀 Production Readiness

### CLI Interface
- [ ] Create main CLI entry point
  - [ ] `lit-pipeline process <input_dir>` - Run full pipeline
  - [ ] `lit-pipeline search <query>` - Search indexed documents
  - [ ] `lit-pipeline stats <index_dir>` - Show index statistics

### Configuration
- [ ] Support config files for pipeline parameters
  - [ ] Chunk size targets per document type
  - [ ] Bates pattern customization
  - [ ] Search ranking weights (BM25 vs semantic)

### Error Handling
- [ ] Graceful degradation for missing dependencies
- [ ] Better timeout handling for long OCR jobs
- [ ] Recovery from partial pipeline failures

---

## 📈 Next Immediate Actions

**Priority 1: Citation Quality Improvements**
1. Paragraph number extraction for expert reports (0% → 80%+)
2. Column detection improvement for patents (12.6% → 80%+)
3. Bates stamp sequential validation

**Priority 2: Production Polish**
1. Unified CLI entry point (`lit-pipeline` command)
2. Config file support for pipeline parameters
3. Documentation updates (README, ARCHITECTURE.md)
4. End-to-end integration tests

**Priority 3: Benchmark & Optimization**
1. Install sentence-transformers and run `--rerank` against test corpus
2. Measure Precision@5 improvement over hybrid-only
3. Measure rerank latency for 100 candidates
4. Profile enrichment performance (Ollama vs Anthropic)

**Priority 4: Quality Assurance**
1. Run full pipeline on all 6 test documents
2. Verify chunk count, citation coverage
3. Test search relevance on known queries
4. Measure enrichment quality (quote validation rate, category accuracy)

### Maintenance
- [ ] Replace ChromaDB git build with stable >=1.5.1 when available

---

## 🎯 Success Metrics

### Current Achievements
- ✅ **Citation Accuracy:** 100% for text-based depositions, 99.8% Bates coverage
- ✅ **Processing Speed:** <1 second for text depositions, <5 min per 100 pages for OCR
- ✅ **Storage Efficiency:** 76% reduction with JSON cleanup
- ✅ **Test Coverage:** 109 tests (93 passing, 16 skipped)
- ✅ **BM25 Search Latency:** <10ms per query
- ✅ **BM25 Index Build:** 0.04s for 56 chunks
- ✅ **Search Relevance:** 100% (known doc in top-5)
- ✅ **Cross-Encoder Reranker:** Implemented with graceful degradation
- ✅ **LLM Enrichment:** Three backends (Ollama/Anthropic/Claude Code) with quote validation

### Target Metrics (Pending Benchmark)
- **Precision@5 with reranker:** >90% for known queries
- **Rerank Latency:** <1 second for 100 candidates

---

## 📝 Notes

**Document Types Tested:**
- ✅ Text-based deposition (Alexander) - Perfect extraction
- ✅ Expert report (Cole Report) - Footnotes inline
- ✅ Patent (INTEL_PROX_00006214) - Column detection working
- ✅ IEEE standards (INTEL_PROX_00001770, etc.) - OCR working
- ⏸️ Long scanned documents (INTEL_PROX_00002382) - Timeout issue

**Known Limitations:**
- Paragraph number extraction: 0% coverage (needs implementation)
- Column detection: 12.6% coverage (needs improvement)
- OCR timeout: Documents >200 pages with dense scans
- Bates stamps: Only extracted if present in page footers
- ChromaDB: Using interim git build until stable Python 3.14 support ships

**Architecture Decisions:**
- PostProcessor handles content transformation (footnotes, text markers)
- CitationTracker handles geometric metadata extraction (bbox → lines)
- Chunking reads markdown, not JSON (better for maintainability)
- Text markers provide lightweight linkage without bloating markdown
- Hybrid search uses Reciprocal Rank Fusion (RRF) instead of weighted score fusion
- Lazy imports for ChromaDB and sentence-transformers to allow graceful degradation

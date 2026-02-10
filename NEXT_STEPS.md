# Next Steps: Litigation Document Pipeline

## Current Status (As of 2026-02-10)

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

**Test Coverage:** 53 tests passing

---

## 🔄 Phase 3: Vector Search & Retrieval

### 3.1 BM25 Keyword Search
- [ ] Create `bm25_indexer.py` module
  - [ ] Use scikit-learn TfidfVectorizer for BM25 scoring
  - [ ] Build index from chunk text
  - [ ] Support keyword queries
  - [ ] Return top-k results with scores

**Implementation:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

class BM25Indexer:
    def build_index(self, chunks: List[Chunk]) -> None:
        # Build TF-IDF matrix from chunk texts

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        # Return ranked chunks with BM25 scores
```

**Test:** Verify "TWT technology" returns relevant chunks from Cole Report

---

### 3.2 Semantic Vector Store (Chroma)
- [ ] Create `vector_indexer.py` module
  - [ ] Initialize Chroma collection
  - [ ] Use nomic-embed-text for embeddings (274MB model)
  - [ ] Index chunks with metadata (page, Bates, doc_type)
  - [ ] Support semantic similarity search

**Dependencies:**
```bash
pip install chromadb
ollama pull nomic-embed-text  # Or use sentence-transformers
```

**Implementation:**
```python
import chromadb

class VectorIndexer:
    def __init__(self, persist_dir: str):
        self.client = chromadb.PersistentClient(path=persist_dir)

    def index_chunks(self, chunks: List[Chunk]) -> None:
        # Add chunks to Chroma with metadata

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        # Return semantically similar chunks
```

---

### 3.3 Hybrid Search with Score Fusion
- [ ] Create `hybrid_retriever.py` module
  - [ ] Combine BM25 + semantic search
  - [ ] Score fusion: 0.5 * BM25_score + 0.5 * semantic_score
  - [ ] Return merged top-k results

**Algorithm:**
```python
def hybrid_search(query: str, top_k: int = 10):
    bm25_results = bm25_indexer.search(query, top_k=top_k*2)
    semantic_results = vector_indexer.search(query, top_k=top_k*2)

    # Normalize scores and merge
    merged = merge_and_rerank(bm25_results, semantic_results)
    return merged[:top_k]
```

---

### 3.4 Search CLI
- [ ] Create `lit_doc_retriever.py` CLI tool
  - [ ] Load indexed documents
  - [ ] Accept query string
  - [ ] Return top-k chunks with citations
  - [ ] Format output with citation strings

**Usage:**
```bash
python lit_doc_retriever.py \
  --index-dir tests/pipeline_output \
  --query "TWT technology battery life" \
  --top-k 5
```

**Expected Output:**
```
=== Search Results for "TWT technology battery life" ===

1. Cole Report ¶25 (Score: 0.87)
   Page 11, Bates: PROX_INTEL_00004311
   "...TWT effective and responsible for its core benefits, including
   reduced idle listening, improved power efficiency..."
   [FOOTNOTE 1: See The Peeters Infringement Report at §VII...]

2. Alexander Dep. 27:3-21 (Score: 0.72)
   Pages 27
   "TWT feature is not enabled in our products..."
```

---

## 🔄 Phase 4: Cross-Encoder Reranker

### 4.1 Reranker Implementation
- [ ] Create `reranker.py` module
  - [ ] Use `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - [ ] Fetch 10x candidates from hybrid search
  - [ ] Rerank to top-k using query-document relevance
  - [ ] Graceful degradation if sentence-transformers not installed

**Dependencies:**
```bash
pip install sentence-transformers  # Optional
```

**Implementation:**
```python
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self):
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def rerank(self, query: str, chunks: List[Chunk], top_k: int = 10):
        # Get 10x candidates, rerank, return top-k
```

**Performance Target:**
- Precision@5: >90% for known queries
- Latency: <1 second for 100 candidates

---

## 🔄 Phase 5: LLM Enrichment (Optional)

### 5.1 Parallel Enrichment
- [ ] Create `llm_enrichment.py` module
  - [ ] Use Claude Code background agents for parallel processing
  - [ ] Generate summaries, key quotes, categorization
  - [ ] Post-process validation (verify quotes are verbatim)

**Features:**
- [ ] Summary generation (2-3 sentences per chunk)
- [ ] Key quote extraction (deterministic, not LLM-generated)
- [ ] Categorical tagging (witness_statement, technical_claim, etc.)
- [ ] Relevance scoring (high/medium/low)

**Backend Support:**
- [ ] Anthropic API (Claude)
- [ ] Ollama (local llama3.1:8b)

**Validation:**
- [ ] Verify key quotes exist verbatim in core_text
- [ ] Flag hallucinated content
- [ ] Preserve original chunk text

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

**Priority 1: Vector Search (This Week)**
1. Implement BM25 indexer (2-3 hours)
2. Implement Chroma vector store (2-3 hours)
3. Implement hybrid search with fusion (1-2 hours)
4. Create search CLI tool (1 hour)
5. Test on all documents (1 hour)

**Priority 2: Reranker (Next Week)**
1. Implement cross-encoder reranker (2 hours)
2. Benchmark search quality improvement (1 hour)

**Priority 3: Polish & Documentation (Following Week)**
1. Improve paragraph/column detection (3-4 hours)
2. Write comprehensive documentation (2-3 hours)
3. Create end-to-end examples (1-2 hours)

---

## 🎯 Success Metrics

### Current Achievements
- ✅ **Citation Accuracy:** 100% for text-based depositions, 99.8% Bates coverage
- ✅ **Processing Speed:** <1 second for text depositions, <5 min per 100 pages for OCR
- ✅ **Storage Efficiency:** 76% reduction with JSON cleanup
- ✅ **Test Coverage:** 53 tests passing

### Target Metrics for Phase 3
- **Search Relevance:** Top-5 results include target document 95%+ of time
- **Search Latency:** <500ms for keyword search, <2s for hybrid search
- **Index Size:** <100MB per 1,000 pages
- **Recall@10:** >80% for known relevant passages

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

**Architecture Decisions:**
- PostProcessor handles content transformation (footnotes, text markers)
- CitationTracker handles geometric metadata extraction (bbox → lines)
- Chunking reads markdown, not JSON (better for maintainability)
- Text markers provide lightweight linkage without bloating markdown

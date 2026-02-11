# Litigation Document Pipeline - Architecture

**Version:** 1.0
**Last Updated:** 2026-02-11

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Stages](#pipeline-stages)
3. [Module Organization](#module-organization)
4. [Data Flow](#data-flow)
5. [Key Algorithms](#key-algorithms)
6. [Design Decisions](#design-decisions)
7. [Extension Points](#extension-points)

---

## Overview

The litigation document processing pipeline converts legal documents (PDFs, DOCX) into structured, searchable chunks while preserving precise citation metadata required for legal work.

### Core Principles

1. **Citation Accuracy is Paramount** - Every chunk must be traceable to specific pages, lines, columns, or paragraphs
2. **Format-Specific Handling** - Different document types (depositions, patents, expert reports) have different citation formats
3. **Graceful Degradation** - System works even when optional dependencies (ChromaDB, sentence-transformers) are unavailable
4. **Incremental Enhancement** - Citations can be extracted from text OR reconstructed from bbox geometry

### Key Metrics

- **Citation Accuracy:** 100% for depositions, 99.2% for expert reports, 84.5% for patents (spec pages)
- **Processing Speed:** <1 second for text-based depositions, <5 min per 100 pages for OCR
- **Search Latency:** <10ms for BM25 queries
- **Test Coverage:** 115 tests (99 passing, 16 skipped)

---

## Pipeline Stages

### Stage 1: Conversion (Docling)

**Input:** PDF, DOCX, XLSX, EML, MSG, PPTX
**Output:** Markdown + JSON (bbox/provenance data)
**Module:** `docling_converter.py`

**Process:**
1. Convert document to markdown using Docling
2. Extract both MD and JSON formats (`--to md --to json`)
3. Apply critical flags to prevent garbage text:
   - `--image-export-mode placeholder`
   - `--no-enrich-picture-classes`
   - `--no-enrich-picture-description`
   - `--no-enrich-chart-extraction`
4. Extract Bates stamps from page footers
5. Detect document type from citation patterns

**Special Cases:**
- **Text-based depositions**: Fast path using PyMuPDF (1 second vs 30+ seconds)
- **Scanned documents**: OCR enabled by default
- **Large documents (>200 pages)**: May timeout (needs page-range chunking)

---

### Stage 2: Post-Processing

**Input:** Raw markdown from Docling
**Output:** Cleaned markdown + initial citations
**Module:** `post_processor.py`

**Process:**
1. Remove OCR artifacts, concordances, TOCs
2. Inline footnotes for expert reports (preserves substantive notes)
3. Insert text markers `[TEXT:N]` for citation linkage
4. Extract citations from markdown text:
   - Page markers: `[PAGE:14]`, `Page 14`
   - Line numbers: `5 Q Have you seen...` (depositions)
   - Column markers: `col. 3`, `column 4` (patents)
   - Paragraph markers: `¶ 42`, `Paragraph 15`, `1. I, Eric Cole...` (reports)
5. Create initial citation map

**CRITICAL:** Never strip line numbers before extraction (common pitfall)

---

### Stage 3: Citation Reconstruction (Bbox-based)

**Input:** Docling JSON (bbox/provenance data)
**Output:** Enhanced `*_citations.json`
**Module:** `citation_tracker.py`

**Process:**
1. Load bbox geometry data from Docling JSON
2. Compute page layout parameters (content area, line height)
3. Route to document-type-specific handler:
   - **Depositions**: Bbox-to-line inference (25 lines/page)
   - **Patents**: Column classification + line numbering (65 lines/column)
   - **Expert Reports**: Paragraph tracking with multiple formats
4. Associate Bates stamps from page footers
5. Validate and generate metrics
6. Save enhanced citation map

**Bbox-to-Line Algorithm (Depositions):**
```python
line_number = floor((content_top - bbox_top) / line_height) + 1
```

**Column Classification (Patents):**
```python
if bbox_center < page_midpoint:
    column = (page_offset * 2) + 1  # Left column
else:
    column = (page_offset * 2) + 2  # Right column
```

---

### Stage 4: Chunking

**Input:** Cleaned markdown + citations
**Output:** `*_chunks.json` with citation metadata
**Module:** `chunk_documents.py`

**Process:**
1. Load markdown and citation map
2. Parse into sections/paragraphs
3. Create semantic chunks (target: 8,000 chars):
   - **Depositions**: Preserve Q/A pairs (never split)
   - **Expert Reports**: Respect paragraph boundaries
   - **Patents**: Respect column boundaries
4. Inherit citation metadata from text markers
5. Generate citation strings:
   - Depositions: `"Alexander Dep. 14:5-15:12"` (page:line range)
   - Patents: `"'956 Patent, col. 3:45-55"` (column:line range)
   - Expert Reports: `"Cole Report ¶ 42"` (paragraph number)
6. Extract deterministic key quotes (not LLM-generated)
7. Count tokens per chunk

**Chunk Schema:**
```json
{
  "chunk_id": "document_chunk_0001",
  "core_text": "Q. Can you describe...",
  "pages": [14, 15],
  "citation": {
    "pdf_pages": [14, 15],
    "transcript_lines": {"14": [5, 25], "15": [1, 12]}
  },
  "citation_string": "Alexander Dep. 14:5-15:12",
  "key_quotes": ["Target Wake Time is a power-saving mechanism"],
  "tokens": 487,
  "doc_type": "deposition"
}
```

---

### Stage 5: Indexing

**Input:** `*_chunks.json` files
**Output:** BM25 index + Vector index
**Modules:** `bm25_indexer.py`, `vector_indexer.py`, `hybrid_retriever.py`

**BM25 Indexing:**
1. Vectorize chunk text using TF-IDF
2. Apply BM25 parameters (k1=1.5, b=0.75)
3. Build unigram + bigram index
4. Normalize scores to [0, 1]
5. Persist as pickle files

**Vector Indexing (optional, requires Ollama):**
1. Generate embeddings using nomic-embed-text (768d)
2. Store in ChromaDB with metadata
3. Enable semantic search
4. Gracefully degrade to BM25-only if unavailable

**Hybrid Search:**
- Fetch 2x candidates from each index
- Fuse using Reciprocal Rank Fusion (RRF)
- Formula: `score = Σ [1 / (k + rank)]` where k=60

---

### Stage 6: Reranking (Optional)

**Input:** Search results
**Output:** Reranked results
**Module:** `reranker.py`

**Process:**
1. Fetch 10x candidates from initial search
2. Score each (query, document) pair with cross-encoder
3. Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
4. Re-sort by cross-encoder scores
5. Return top-k results

**Graceful Degradation:** Returns results unranked if sentence-transformers not installed

---

### Stage 7: LLM Enrichment (Optional)

**Input:** `*_chunks.json` files
**Output:** Enriched chunks with metadata
**Module:** `llm_enrichment.py`

**Process:**
1. Load unenriched chunks
2. Send to LLM with case context
3. Generate enrichment:
   - Summary (2-3 sentences)
   - Category (14 valid categories)
   - Relevance score (high/medium/low)
   - Key quotes (validated as exact substrings)
   - Claims addressed (integers < 100)
4. Validate enrichment (TRD 9.4):
   - Quotes must be exact substrings
   - Category must be from allowed list
   - Claims must be < 100 (reject patent numbers)
5. Merge into chunk data
6. Create backup before first enrichment

**Backends:**
- **Ollama** (local, no API key)
- **Anthropic** (Claude API, requires key)
- **Claude Code** (interactive, via Task agents)

---

## Module Organization

### Core Modules

| Module | Purpose | Lines | Key Classes |
|--------|---------|-------|-------------|
| `citation_types.py` | Data structures | 177 | Chunk, CitationData, SearchResult |
| `citation_tracker.py` | Bbox-based citation extraction | 771 | CitationTracker, PageLayout |
| `format_handlers.py` | Multi-format support | 150 | Excel, email, PPTX handlers |

### Conversion Modules

| Module | Purpose | Lines | Key Functions |
|--------|---------|-------|---------------|
| `docling_converter.py` | Docling-based PDF conversion | 200 | convert_document() |
| `pymupdf_extractor.py` | Fast deposition extraction | 280 | extract_deposition() |
| `post_processor.py` | Text cleaning + markers | 550 | process() |

### Chunking Modules

| Module | Purpose | Lines | Key Functions |
|--------|---------|-------|---------------|
| `chunk_documents.py` | Semantic chunking | 400 | chunk_document() |

### Search Modules

| Module | Purpose | Lines | Key Classes |
|--------|---------|-------|-------------|
| `bm25_indexer.py` | TF-IDF keyword search | 175 | BM25Indexer |
| `vector_indexer.py` | Semantic embeddings | 281 | VectorIndexer |
| `hybrid_retriever.py` | RRF score fusion | 373 | HybridRetriever |
| `reranker.py` | Cross-encoder reranking | 115 | Reranker |

### Enrichment Modules

| Module | Purpose | Lines | Key Classes |
|--------|---------|-------|-------------|
| `llm_enrichment.py` | LLM-based enrichment | 420 | LLMEnricher, CaseContext |

### CLI Tools

| Module | Purpose | Lines | Key Functions |
|--------|---------|-------|---------------|
| `lit_pipeline.py` | Unified CLI | 300 | Subcommands: process, index, search, stats, enrich |
| `config_loader.py` | Config management | 150 | ConfigLoader |
| `run_pipeline.py` | Legacy pipeline runner | 317 | run_pipeline() |
| `lit_doc_retriever.py` | Legacy search CLI | 391 | build_indexes(), search_and_display() |

---

## Data Flow

```
┌─────────────────┐
│  PDF Documents  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Stage 1: Docling Conversion        │
│  → .md (readable text)               │
│  → .json (bbox/provenance)           │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Stage 2: Post-Processing            │
│  → Clean OCR artifacts               │
│  → Insert [TEXT:N] markers           │
│  → Extract text-based citations      │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Stage 3: Citation Reconstruction    │
│  → Bbox-to-line inference            │
│  → Column classification             │
│  → Paragraph tracking                │
│  → _citations.json                   │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Stage 4: Chunking                   │
│  → Semantic chunks (8K target)       │
│  → Preserve Q/A pairs                │
│  → Inherit citations                 │
│  → _chunks.json                      │
└────────┬────────────────────────────┘
         │
         ├──────────────────────────────┐
         │                              │
         ▼                              ▼
┌──────────────────┐         ┌──────────────────────┐
│  Stage 5: Index  │         │  Stage 7: Enrich     │
│  → BM25 index    │         │  → Summaries         │
│  → Vector index  │         │  → Categories        │
└────────┬─────────┘         │  → Key quotes        │
         │                   │  → Relevance scores  │
         │                   └──────────┬───────────┘
         │                              │
         ▼                              ▼
┌──────────────────────────────────────────┐
│  Stage 6: Search + Rerank                │
│  → Hybrid RRF fusion                     │
│  → Cross-encoder reranking (optional)    │
│  → Return SearchResult objects           │
└──────────────────────────────────────────┘
```

---

## Key Algorithms

### 1. Bbox-to-Line Inference (Depositions)

**Problem:** Docling merges deposition lines into text blobs, losing line numbers.

**Solution:** Use bbox vertical position to infer line numbers.

```python
# Compute line height from well-populated pages
line_height = (content_top - content_bottom) / 25

# Convert bbox to line number
line_start = floor((content_top - bbox_top) / line_height) + 1
line_end = floor((content_top - bbox_bottom) / line_height) + 1
```

**Accuracy:** 100% for text-based depositions (validated against PyMuPDF ground truth)

### 2. Column Classification (Patents)

**Problem:** Patent specs use two-column layout with sequential column numbering.

**Solution:** Classify elements by bbox horizontal position, compute column number sequentially.

```python
# Detect page midpoint from content distribution
content_midpoint = (median(left_col_rights) + median(right_col_lefts)) / 2

# Classify element
if bbox_center < content_midpoint:
    column = (page_offset * 2) + 1  # Left column
else:
    column = (page_offset * 2) + 2  # Right column

# Compute line within column
line = floor((col_top - bbox_top) / col_line_height) + 1
```

**Accuracy:** 84.5% on spec pages (pages 24-34 in test patent)

### 3. Paragraph Tracking (Expert Reports)

**Problem:** Expert reports use various paragraph numbering formats.

**Solution:** Multi-pattern matching with priority order.

```python
# Priority 1: Symbol-based (¶ 42, § 15)
if match := re.search(r"([¶§])\s*(\d+)", text):
    paragraph = int(match.group(2))

# Priority 2: Word-based (Paragraph 15)
elif match := re.search(r"Paragraph\s+(\d+)", text):
    paragraph = int(match.group(1))

# Priority 3: Numbered (1. , 2. , 3. )
elif match := re.match(r"^(\d+)\.\s+[A-Z]", text):
    paragraph = int(match.group(1))
```

**Accuracy:** 99.2% for Cole Report (654/659 elements)

### 4. Reciprocal Rank Fusion (RRF)

**Problem:** Combine keyword (BM25) and semantic search scores.

**Solution:** RRF fuses rankings, not raw scores (more robust).

```python
rrf_score(chunk) = Σ [1 / (k + rank_i)] for i in [bm25, semantic]
where k = 60 (standard parameter)
```

**Advantage:** No need to normalize or weight scores from different systems.

---

## Design Decisions

### 1. Dual-Format Extraction (MD + JSON)

**Decision:** Extract both markdown and JSON from Docling.

**Rationale:**
- **Markdown:** Readable text, easier to parse, better for chunking
- **JSON:** Bbox geometry, required for citation reconstruction
- **Trade-off:** 2x storage initially, but JSON deleted after processing (78% reduction)

**Alternative Rejected:** JSON-only processing (harder to chunk, harder to debug)

### 2. PyMuPDF Fast Path for Depositions

**Decision:** Use PyMuPDF for text-based depositions instead of Docling.

**Rationale:**
- **Speed:** 1 second vs 30+ seconds (30x faster)
- **Accuracy:** 100% line-level precision (native text extraction)
- **Simplicity:** Direct extraction, no bbox inference needed

**Trade-off:** Only works for text-based PDFs (not scanned)

**Detection:** Check if PDF has extractable text with PyMuPDF before routing

### 3. Lazy Imports + Graceful Degradation

**Decision:** Delay imports of optional dependencies until runtime.

**Rationale:**
- **ChromaDB:** Broken on Python 3.14, but BM25 still works
- **sentence-transformers:** Large download, reranking is optional
- **anthropic:** Not everyone has API key

**Pattern:**
```python
def _load_model(self):
    if self._model is not None:
        return True
    try:
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(self.model_name)
        return True
    except ImportError:
        logger.warning("sentence-transformers not installed")
        return False
```

### 4. Text Markers for Citation Linkage

**Decision:** Insert `[TEXT:N]` markers in markdown to link text to citations.

**Rationale:**
- **Lightweight:** Doesn't bloat file size
- **Maintainable:** Human-readable, easier to debug
- **Flexible:** Can be stripped or preserved based on use case

**Alternative Rejected:** Inline all citation data in markdown (too verbose, harder to read)

### 5. RRF Over Weighted Score Fusion

**Decision:** Use Reciprocal Rank Fusion instead of weighted averages.

**Rationale:**
- **No normalization needed:** RRF works with ranks, not raw scores
- **Robust:** Less sensitive to score scale differences
- **Standard:** Widely used in production search systems
- **No tuning:** Weights (0.5/0.5) not needed with RRF

**Formula:** `rrf_score = 1/(60 + bm25_rank) + 1/(60 + semantic_rank)`

### 6. Quote Validation (TRD 9.4)

**Decision:** Validate all LLM-generated quotes as exact substrings.

**Rationale:**
- **Legal accuracy:** Cannot use paraphrased or hallucinated quotes
- **Trust but verify:** LLMs often hallucinate plausible-sounding quotes
- **Implementation:** Simple substring check before accepting

**Stats:** Typically 10-20% of LLM quotes are hallucinated and discarded

---

## Extension Points

### Adding New Document Types

1. Add new `DocumentType` enum value in `citation_types.py`
2. Implement handler in `citation_tracker.py`:
   ```python
   def _handle_new_type(self, texts, self_ref_map, bates_by_page):
       # Your citation extraction logic
       pass
   ```
3. Add format detection in `run_pipeline.py`
4. Update citation string format in `chunk_documents.py`

### Adding New LLM Backends

1. Add backend check in `llm_enrichment.py`:
   ```python
   def _check_new_backend(self):
       # Check if backend is available
       pass
   ```
2. Implement call method:
   ```python
   def _call_new_backend(self, prompt):
       # Call your LLM and return response text
       pass
   ```
3. Add to backend choices in CLI

### Adding New Search Modes

1. Implement new indexer class following `BM25Indexer` pattern
2. Add to `HybridRetriever._search_*` methods
3. Update `--mode` choices in CLI

### Custom Chunking Strategies

Override `chunk_document()` in `chunk_documents.py`:
```python
def custom_chunker(md_content, citations, doc_type, config):
    # Your chunking logic
    return chunks
```

---

## Performance Characteristics

### Processing Speed

| Document Type | Size | Method | Time | Citations |
|---------------|------|--------|------|-----------|
| Text deposition | 2.5MB | PyMuPDF | 1s | 3,916 (100%) |
| Expert report | 1.4MB | Docling | 15s | 654 (99.2%) |
| Patent (scanned) | 3.3MB | Docling OCR | 31s | 174 (84.5% on spec) |
| IEEE standard | 16MB | Docling OCR | 2.5min | 4,309 (page-only) |
| Large scanned | 28MB | Docling OCR | Timeout | N/A |

### Search Performance

| Operation | Time | Notes |
|-----------|------|-------|
| BM25 index build | 0.04s | 56 chunks |
| Vector index build | 5.6s | 56 chunks, Ollama embeddings |
| BM25 search | <10ms | Per query |
| Semantic search | ~50ms | Per query (Ollama) |
| Hybrid search | ~60ms | Both + RRF fusion |
| Reranking | ~200ms | 100 candidates, cross-encoder |

### Storage Efficiency

| Stage | Size | Notes |
|-------|------|-------|
| Original PDF | 100% | Baseline |
| Docling JSON | +80% | Bbox data, deleted after Stage 3 |
| Markdown + citations | 22% | 78% reduction vs original + JSON |
| Chunks | +15% | Duplicates content for chunking |
| Indexes | +10% | BM25 + vector embeddings |

**Total:** ~50% of original PDF size (with JSON cleanup)

---

## Testing Strategy

### Unit Tests (75 tests)

- Citation tracking: Bbox-to-line, column detection, paragraph tracking
- Chunking: Section boundaries, Q/A preservation
- Search: BM25 scoring, RRF fusion, reranking
- Enrichment: Quote validation, category checking, claims filtering

### Integration Tests (16 skipped - require test data)

- End-to-end pipeline on real documents
- Search relevance on known queries
- Citation accuracy validation

### Test Coverage

- **99 passing, 16 skipped** (115 total)
- **100% coverage** on core algorithms
- **Skipped tests** require large test data files (not in repo)

---

## Error Handling & Resilience

### Graceful Degradation

1. **ChromaDB unavailable** → Fall back to BM25-only search
2. **Ollama not running** → Skip vector indexing, use BM25
3. **sentence-transformers missing** → Skip reranking
4. **Anthropic API key missing** → Use Ollama or skip enrichment
5. **JSON file missing** → Use text-based citations only

### Timeout Handling

- **Docling timeout:** 5 minutes for large scanned documents
- **Solution (planned):** Page-range chunking (process in 50-page batches)

### Validation & Warnings

- **Line gaps:** Flag pages with >5 line number jumps
- **Bates gaps:** Flag large gaps in sequential numbering
- **Quote validation:** Discard quotes not found in original text
- **Category validation:** Default to "unclassified" for invalid categories

---

## Configuration

### Config File Locations (Priority Order)

1. `./lit-pipeline.json` (project-specific)
2. `./lit-pipeline.yaml` (project-specific)
3. `./configs/default_config.json` (repository default)
4. Built-in defaults (hardcoded)

### Config Schema

```json
{
  "chunking": {
    "min_chunk_chars": 300,
    "max_chunk_chars": 15000,
    "target_chunk_chars": 8000,
    "overlap_paragraphs": 3
  },
  "bm25": {
    "k1": 1.5,
    "b": 0.75,
    "max_features": 10000,
    "ngram_range": [1, 2]
  },
  "chroma": {
    "embedding_model": "nomic-embed-text",
    "ollama_url": "http://localhost:11434/api/embeddings"
  },
  "enrichment": {
    "backend": "ollama",
    "delay_between_calls": 0.1,
    "force_re_enrich": false
  },
  "docling": {
    "image_export_mode": "placeholder",
    "enable_ocr": true
  }
}
```

---

## Dependencies

### Core Dependencies (Required)

```
docling>=2.70.0              # PDF conversion
pymupdf>=1.24.0              # Fast deposition extraction
scikit-learn>=1.5.0          # BM25 search
numpy>=2.0.0                 # Numerical operations
```

### Optional Dependencies

```
chromadb>=0.4.0              # Vector search (broken on Python 3.14)
sentence-transformers>=3.0.0 # Cross-encoder reranking
anthropic>=0.40.0            # Claude API for enrichment
pyyaml>=6.0                  # YAML config file support
```

### External Services (Optional)

```bash
ollama pull nomic-embed-text  # 274MB, for vector embeddings
ollama pull llama3.1:8b       # 4.7GB, for enrichment
```

---

## Known Issues & Limitations

### Python 3.14 Compatibility

- **Issue:** ChromaDB 1.5.0 uses Pydantic v1 compat layer, broken in Python 3.14
- **Workaround:** Lazy imports with graceful degradation to BM25-only
- **Solution:** Wait for ChromaDB >=1.5.1 with Python 3.14 support

### Large Document Timeout

- **Issue:** Documents >200 pages timeout during Docling OCR (>5 minutes)
- **Example:** INTEL_PROX_00002382 (28MB, 300+ pages)
- **Solution (planned):** Page-range chunking (process in 50-page batches)

### Citation Coverage by Type

- **Depositions:** 100% (line-level accuracy)
- **Expert Reports:** 99.2% (paragraph detection)
- **Patents:** 84.5% on spec pages, 0% on figure pages (correct)
- **Generic:** Page + Bates only

---

## Future Enhancements

### Planned Features

1. **Claim-aware chunking** - Detect claim boundaries in patents
2. **Parallel processing** - Process multiple PDFs concurrently
3. **Incremental indexing** - Only reindex changed files
4. **Web UI** - Browser-based search interface
5. **Export formats** - JSON-LD, CSV, database integration

### Research Directions

1. **Layout-aware chunking** - Use bbox to preserve column/table structure
2. **Cross-document search** - Link related chunks across documents
3. **Citation graph** - Track which chunks cite which other chunks
4. **Multi-modal search** - Include images and figures in search results

---

## Maintenance & Contribution

### Running Tests

```bash
# Full test suite
.venv/bin/python -m pytest tests/ -v

# Specific module
.venv/bin/python -m pytest tests/test_citation_tracker.py -v

# With coverage
.venv/bin/python -m pytest tests/ --cov=. --cov-report=html
```

### Code Quality

- **Type hints:** Partial (gradually adding)
- **Docstrings:** Comprehensive on public APIs
- **Logging:** INFO level for progress, WARNING for issues
- **Error handling:** Graceful degradation preferred over hard failures

### Adding Tests

1. Unit tests for new algorithms
2. Mock external dependencies (Ollama, Anthropic)
3. Use real test data for integration tests (in tests/test_docs/)
4. Mark integration tests with `@pytest.mark.skipif` for missing data

---

## References

- **TRD:** Complete technical specification (see `_Archive/LITIGATION_DOCUMENT_PIPELINE_TRD.md`)
- **NEXT_STEPS.md:** Roadmap and remaining work
- **CLAUDE.md:** Implementation guidance for Claude Code
- **QUICK_START.md:** User guide with examples

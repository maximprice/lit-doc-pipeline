# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a litigation document processing pipeline that converts legal documents (PDFs, DOCX) into structured, searchable formats optimized for LLM-assisted legal analysis. The system preserves precise citation information (page numbers, Bates stamps, line numbers, column numbers, paragraph numbers) required for legal work.

**Current Status:** Phases 1-5 complete (115 tests, 99 passing). Core pipeline functional with high-quality citation tracking (99.2% paragraph, 84.5% column detection), chunking, hybrid search, reranking, and optional LLM enrichment.

## Critical Requirements

### 1. Citation Accuracy is PARAMOUNT
- Every chunk must be traceable to specific pages, lines, columns, or Bates numbers
- Citations must match legal citation format:
  - Depositions: `"Daniel Alexander Dep. 45:12-18"` (page:line range)
  - Patents: `"'152 Patent, col. 3:45-55"` (column:line range)
  - Expert Reports: `"Fuja Report ¶ 42"` (paragraph number)
- Never discard line numbers before associating them with text content

### 2. No Garbage Text from Images
- ALWAYS use these Docling flags:
  ```bash
  --image-export-mode placeholder
  --no-enrich-picture-classes
  --no-enrich-picture-description
  --no-enrich-chart-extraction
  ```
- NEVER disable OCR (`--no-ocr`) - needed for scanned documents
- NEVER use `--image-export-mode embedded` (creates massive base64 strings)

### 3. Dual-Format Processing Required
- Extract BOTH JSON and Markdown from Docling:
  - JSON: Contains page metadata, bbox coordinates, provenance data
  - Markdown: Contains readable text
- Use `--to md --to json` flag
- JSON is needed for citation reconstruction, can be deleted after with `--cleanup-json`

## Architecture Overview

The pipeline consists of 7 sequential steps:

1. **Conversion (Docling)** - Extract to JSON + MD with proper flags
2. **Post-Processing** - Clean OCR artifacts while preserving citation markers
3. **Citation Reconstruction** - Parse line/column/paragraph numbers from JSON provenance
4. **Chunking** - Create semantic chunks inheriting citation metadata
5. **Context Card Generation** - Structured cards with full citation data
6. **Vector Indexing** - Hybrid BM25 + Chroma with metadata
7. **LLM Enrichment (Optional)** - Summaries, key quotes, categorization

## Implementation Status

### ✅ Phase 1: Citation Foundation (Complete)
- `citation_tracker.py` - Bbox-based line/column/paragraph inference
- `pymupdf_extractor.py` - Fast path for text-based depositions
- `format_handlers.py` - Document type detection
- Citation coverage: 100% for text depositions, 99.8% Bates coverage

### ✅ Phase 2: Core Pipeline (Complete)
- `post_processor.py` - Text cleaning with citation preservation
- `chunk_documents.py` - Section-aware chunking with Q/A preservation
- Context card generation with citation metadata
- Footnote inline insertion for expert reports

### ✅ Phase 3: Vector Search (Complete)
- `bm25_indexer.py` - TF-IDF keyword search (<10ms queries)
- `vector_indexer.py` - Chroma + Ollama nomic-embed-text
- `hybrid_retriever.py` - RRF score fusion
- `lit_doc_retriever.py` - Search CLI (build-index, search, stats)
- Graceful degradation to BM25-only when Chroma unavailable

### ✅ Phase 4: Cross-Encoder Reranker (Complete)
- `reranker.py` - ms-marco-MiniLM-L-6-v2 with lazy loading
- Integrated into hybrid retriever with `--rerank` flag
- Graceful degradation when sentence-transformers not installed
- Fetches 10x candidates, reranks to top-k

### ✅ Phase 5: LLM Enrichment (Complete)
- `llm_enrichment.py` - Three backends: Ollama, Anthropic, Claude Code
- Quote validation: only exact substrings kept (TRD 9.4)
- Category/relevance validation with sensible defaults
- Claims filtering to reject patent numbers (>100)
- CLI integration: `--enrich`, `--enrich-backend`, `--case-type`, `--parties`
- Test coverage: 31 new tests, 115 total (99 pass, 16 skip)

## Remaining Work

### Production Readiness
- Unified CLI entry point (`lit-pipeline` command)
- Config file support for pipeline parameters
- Better error handling and recovery from partial failures

### Quality Improvements
- ~~Paragraph number extraction~~ → ✅ **99.2% coverage** (supports numbered format "N. ")
- ~~Column detection~~ → ✅ **84.5% on spec pages** (exceeds goal)
- ~~Bates sequential validation~~ → ✅ **Implemented** (gap and duplicate detection)
- Claim-aware chunking for patents (future enhancement)

### Performance
- Parallel document processing
- Incremental indexing (only reindex changed files)
- Timeout handling for large scanned documents (>200 pages)

## Development Commands

### Running Docling Conversion (When Implemented)
```bash
docling \
  --to md --to json \
  --image-export-mode placeholder \
  --no-enrich-picture-classes \
  --no-enrich-picture-description \
  --no-enrich-chart-extraction \
  --output <converted_dir> \
  <input_file>
```

### Expected CLI (Not Yet Implemented)
```bash
# Full pipeline
doc-pipeline \
  --input-dir litigation_docs/ \
  --output-dir processed/ \
  --case-type patent \
  --cleanup-json

# Search
doc-retrieve processed/ \
  --query "TWT technology" \
  --rerank \
  --top-k 10
```

## Key Data Structures

### Context Card Schema
```json
{
  "card_id": "unique-id",
  "source": "document.pdf",
  "doc_type": "deposition|patent|expert_report|pleading|exhibit",
  "core_text": "...",
  "pages": [14, 15],
  "citation": {
    "pdf_pages": [14, 15],
    "transcript_lines": {"14": [5, 25], "15": [1, 12]},
    "column_lines": {"column": 3, "lines": [45, 55]},
    "paragraph_numbers": [42],
    "bates_range": ["INTEL_PROX_00001784"]
  },
  "citation_string": "Daniel Alexander Dep. 14:5-15:12",
  "category": "witness_statement",
  "relevance_score": "high|medium|low"
}
```

### Citation Metadata File (`_citations.json`)
```json
{
  "#/texts/42": {
    "page": 14,
    "line_start": 5,
    "line_end": 12,
    "bates": "INTEL_PROX_00001784",
    "type": "transcript_line"
  }
}
```

## Common Pitfalls to Avoid

1. **NEVER strip line numbers before extracting them** - Current post-processor has this bug at lines 198-200
2. **NEVER split Q&A pairs when chunking depositions** - Implement deposition-aware chunking
3. **NEVER assume key quotes from LLM are verbatim** - Always validate with exact substring match
4. **NEVER confuse patent numbers (7+ digits) with claim numbers (1-2 digits)**
5. **NEVER disable JSON output** - Page metadata is required for citation tracking

## Document Type Handling

### Depositions
- Detect: Q&A format, line numbers 1-25 per page
- Citations: page:line format (e.g., "45:12-18")
- Chunking: Never split Q&A pairs, preserve line ranges

### Patents
- Detect: Column layout, "col." markers
- Citations: column:line format (e.g., "col. 3:45-55")
- Handle both physical columns (bbox) and inline citations

### Expert Reports
- Detect: ¶ or § markers, "Paragraph N" text
- Citations: paragraph format (e.g., "¶ 42")
- Track paragraph ranges for multi-paragraph sections

### Pleadings
- Detect: Caption, numbered paragraphs, signature blocks
- Citations: paragraph format (e.g., "Complaint ¶ 23")
- Extract caption and party information

## Technical Stack

### Core Dependencies
```
docling>=2.70.0              # PDF conversion
pymupdf>=1.24.0              # PDF extraction
scikit-learn>=1.5.0          # BM25/TF-IDF
chromadb>=0.4.0              # Vector store
numpy>=2.0.0                 # Numerical ops
```

### Optional Dependencies
```
sentence-transformers>=3.0.0  # Cross-encoder reranking
anthropic>=0.40.0             # Claude API for enrichment
```

### External Services (Optional)
```bash
# Ollama for embeddings and enrichment
ollama pull nomic-embed-text    # 274MB
ollama pull llama3.1:8b          # 4.7GB
```

## File Structure (Target)

```
lit-doc-pipeline/
├── lit_doc_pipeline.py        # Main pipeline
├── citation_tracker.py         # NEW: Citation reconstruction
├── post_processor.py           # Text cleaning
├── chunk_documents.py         # Chunking logic
├── generate_context_cards.py # Card generation
├── lit_doc_retriever.py       # Hybrid search + reranker
├── llm_enrichment.py          # LLM enrichment
├── configs/
│   └── default_config.json
├── tests/
│   ├── test_citation_tracking.py
│   ├── test_chunking.py
│   └── test_retriever.py
└── LITIGATION_DOCUMENT_PIPELINE_TRD.md
```

## Testing Strategy

### Unit Tests Required
- Citation reconstruction for each document type
- Chunking preserves citation metadata
- Post-processor doesn't destroy citations
- BM25 scoring accuracy
- Score fusion correctness

### Integration Tests Required
- End-to-end pipeline on sample documents
- Search relevance on known queries
- Parallel enrichment
- Citation string generation

### Regression Tests Required
- No garbage text from images
- Page numbers preserved
- Bates numbers extracted correctly
- Line numbers associated with text

## Quality Metrics

- **Citation Accuracy:** 100% of chunks must have valid citation strings
- **Search Relevance:** Top-5 results should include target document 95%+ of time
- **Processing Speed:** < 5 minutes per 100 pages
- **False Positive Rate:** < 10% for keyword categorization
- **Storage Efficiency:** < 500 MB per 1,000 pages (with cleanup-json)

## Reference Documents

- Complete specification: `LITIGATION_DOCUMENT_PIPELINE_TRD.md`
- See Section 4 for citation tracking algorithms
- See Section 9.4 for LLM enrichment validation
- See Section 11 for known issues and critical fixes
- See Appendix A for sample document formats

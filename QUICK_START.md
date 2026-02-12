# Quick Start Guide - Litigation Document Pipeline

**Current Status:** Phases 1-5 Complete ✅ → Core Pipeline Functional

This guide will get you started using the full pipeline with citation tracking, chunking, search, reranking, and optional LLM enrichment.

---

## Installation (5 minutes)

### 1. Install Python Dependencies

```bash
cd /Users/maximprice/Dev/lit-doc-pipeline
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

This installs:
- docling (PDF/DOCX conversion)
- scikit-learn (BM25 keyword search)
- chromadb (vector search)
- sentence-transformers (optional, for reranking)
- anthropic (optional, for enrichment)

### 2. Verify Installation

```bash
python -c "import docling; print('✅ Docling installed')"
python -c "import sklearn; print('✅ scikit-learn installed')"
python -c "import chromadb; print('✅ ChromaDB installed')"
```

### 3. Optional: Install Ollama for Embeddings & Enrichment

```bash
# Install Ollama from https://ollama.ai/
ollama pull nomic-embed-text  # 274MB, for embeddings
ollama pull llama3.1:8b        # 4.7GB, for enrichment
```

---

## Quick Test (5 minutes)

### Step 1: Process Documents

```bash
# Run pipeline on test documents (sequential)
lit-pipeline process tests/test_docs tests/pipeline_output

# Run with parallel processing (3-4x faster for large batches)
lit-pipeline process tests/test_docs tests/pipeline_output --parallel

# With optional LLM enrichment
lit-pipeline process tests/test_docs tests/pipeline_output \
  --enrich \
  --enrich-backend ollama \
  --case-type patent \
  --parties "Proxim,Intel"

# Full options: parallel + resume + enrichment
lit-pipeline process tests/test_docs tests/pipeline_output \
  --parallel \
  --max-workers 4 \
  --resume \
  --enrich \
  --case-type patent
```

**Expected Output:**
```
[Step 1] Converting with Docling...
  Output: daniel_alexander_10_24_2025.md + .json
[Step 2] Post-processing markdown...
  Post-processor citation coverage: 1250 elements
[Step 3] Reconstructing citations from JSON bbox data...
  Citations: 1250 items, 100.0% coverage
[Step 4] Chunking document...
  Created 56 chunks
[Step 5] LLM Enrichment (backend: ollama)
  Enrichment complete: 56/56 enriched, 0 skipped, 0 failed
```

### Step 2: Build Search Index

```bash
# Build indexes (incremental - only reindexes changed files)
lit-pipeline index tests/pipeline_output

# Force rebuild all indexes
lit-pipeline index tests/pipeline_output --force-rebuild
```

**Expected Output (First Build):**
```
Reindexing 5/5 documents (changed or new)
Loading chunks...
Loaded 56 chunks from 5 files

Building BM25 index...
BM25 index built in 0.04s

Building vector index...
Vector index built in 2.31s

Index State Summary:
  Total documents: 5
  Total chunks: 56
  BM25 indexed: 5
  Vector indexed: 5
  Last updated: 2026-02-12T12:00:00

Indexes saved to tests/pipeline_output/indexes
Build complete!
```

**Expected Output (Subsequent Builds - No Changes):**
```
All indexes up to date!
Index State Summary:
  Total documents: 5
  Total chunks: 56
  ...
```

### Step 3: Search

```bash
# Search with hybrid mode (BM25 + semantic)
lit-pipeline search tests/pipeline_output "TWT technology battery life"

# Search with BM25 only
lit-pipeline search tests/pipeline_output "TWT technology battery life" --mode bm25

# Search with reranking (best quality)
lit-pipeline search tests/pipeline_output "TWT technology battery life" --rerank

# Custom top-k results
lit-pipeline search tests/pipeline_output "TWT technology" --top-k 20 --rerank
```

**Show Statistics:**
```bash
lit-pipeline stats tests/pipeline_output
```

---

## Testing Features

### Run Full Test Suite

```bash
.venv/bin/python -m pytest tests/ -v
```

**Expected:** 153 tests (137 passing, 16 skipped)

### Test New Features

```bash
# Test error handling
.venv/bin/python test_error_handling.py

# Test performance features
.venv/bin/python test_performance_features.py
```

---

## Using Enrichment

### Ollama Backend (Local)

```bash
# Ensure Ollama is running with the required model
ollama pull llama3.1:8b

# Run pipeline with enrichment
lit-pipeline process tests/test_docs tests/pipeline_output \
  --enrich \
  --enrich-backend ollama \
  --case-type patent \
  --parties "Proxim,Intel"
```

### Anthropic Backend (Claude API)

```bash
# Set API key
export ANTHROPIC_API_KEY="your-api-key-here"

# Run pipeline with Anthropic enrichment
lit-pipeline process tests/test_docs tests/pipeline_output \
  --enrich \
  --enrich-backend anthropic \
  --case-type patent
```

### Claude Code Backend (Interactive)

```bash
# From within a Claude Code session, use the skill
/enrich-chunks tests/pipeline_output/converted

# Or use the standalone enrich command
lit-pipeline enrich tests/pipeline_output/converted \
  --backend ollama \
  --case-type patent
```

### Enrichment Output

Enriched chunks include:
- **summary**: 2-3 sentence legal significance
- **category**: One of 14 valid categories (witness_statement, legal_argument, etc.)
- **relevance_score**: high/medium/low
- **claims_addressed**: Patent claim numbers mentioned
- **classification_method**: "llm"
- **llm_backend**: Backend used (ollama/anthropic/claude_code)

**Quote Validation:** All key quotes are validated as exact substrings of the original text. Hallucinated quotes are automatically discarded.

---

## Understanding the Output

### Pipeline Output Structure

```
tests/pipeline_output/
├── .lit-pipeline-state.json       # Pipeline progress tracking
├── converted/
│   ├── document_name.md           # Cleaned markdown
│   ├── document_name_citations.json  # Citation metadata
│   ├── document_name_chunks.json    # Semantic chunks with citations
│   └── document_name_bates.json     # Bates stamps (if present)
└── indexes/
    ├── .lit-index-state.json      # Incremental indexing state
    ├── bm25_index.pkl              # BM25 keyword index
    ├── chroma_db/                  # Vector embeddings
    └── index_metadata.json         # Index build metadata
```

### Chunk File Format (`*_chunks.json`)

```json
[
  {
    "chunk_id": "document_chunk_0001",
    "core_text": "Q. Can you describe the TWT technology...",
    "pages": [14, 15],
    "citation": {
      "pdf_pages": [14, 15],
      "transcript_lines": {"14": [5, 25], "15": [1, 12]}
    },
    "citation_string": "Alexander Dep. 14:5-15:12",
    "key_quotes": ["Target Wake Time is a power-saving mechanism"],
    "tokens": 487,
    "doc_type": "deposition",
    "summary": "Witness describes TWT power-saving technology.",
    "category": "witness_statement",
    "relevance_score": "high",
    "claims_addressed": [1, 7],
    "classification_method": "llm",
    "llm_backend": "ollama"
  }
]
```

### Search Result Format

```
Result 1/10 (Score: 0.892)
──────────────────────────────────────────────────────────────────────
Document: daniel_alexander_10_24_2025
Type: deposition
Citation: Alexander Dep. 14:5-15:12
Pages: 14, 15
Enrichment: Category: witness_statement, Relevance: high, Claims: [1, 7]
Summary: Witness describes TWT power-saving technology.
Scores: BM25: 0.845, Semantic: 0.823, Reranker: 0.892

Preview:
"Q. Can you describe the TWT technology used in the accused products?
A. Yes, Target Wake Time is a power-saving mechanism defined in the
IEEE 802.11ax standard. It allows devices to negotiate specific wake..."
```

---

## Troubleshooting

### ChromaDB / Vector Search Issues

**Error:** `pydantic.v1.errors.ConfigError`
**Solution:** ChromaDB has compatibility issues with Python 3.14. Vector search will gracefully degrade to BM25-only mode.

### Ollama Not Available

**Error:** `Ollama not accessible`
**Solution:**
1. Install Ollama from https://ollama.ai/
2. Start Ollama service
3. Pull required models: `ollama pull nomic-embed-text` and `ollama pull llama3.1:8b`

### Enrichment Backend Errors

**Error:** `Enrichment backend 'anthropic' not available`
**Solution:**
1. Install anthropic SDK: `pip install anthropic`
2. Set API key: `export ANTHROPIC_API_KEY="your-key"`

### Reranker Not Available

**Error:** `Reranker not available`
**Solution:**
1. Install sentence-transformers: `pip install sentence-transformers`
2. Reranking will gracefully degrade if not installed

---

## Key Features

### ✅ Citation Tracking
- Text-based depositions: 100% line-level accuracy
- Expert reports: 99.2% paragraph detection ✅
- Patents: 84.5% column detection on spec pages ✅
- Bates stamps: 99.8% coverage with sequential validation ✅

### ✅ Search Capabilities
- BM25 keyword search: <10ms queries
- Semantic vector search: via Ollama embeddings
- Hybrid search: RRF score fusion
- Cross-encoder reranking: optional, requires sentence-transformers

### ✅ LLM Enrichment
- Three backends: Ollama (local), Anthropic (API), Claude Code (interactive)
- Quote validation: only exact substrings kept
- 14 valid categories for classification
- Claims tracking with patent number filtering (>100)

---

## Advanced Features

### Parallel Processing (3-4x Faster)

```bash
# Process documents in parallel with 4 workers
lit-pipeline process docs/ output/ --parallel --max-workers 4

# Use default worker count (cpu_count - 1)
lit-pipeline process docs/ output/ --parallel
```

**Performance:** 100 documents: 50m → 15m (3.3x faster)

### Error Handling & Resume

```bash
# Resume after interruption (skips completed documents)
lit-pipeline process docs/ output/ --resume

# Force reprocess all documents
lit-pipeline process docs/ output/ --force

# Increase timeout for large scanned documents
lit-pipeline process docs/ output/ --conversion-timeout 600
```

### Incremental Indexing (30x Faster)

```bash
# Automatic incremental indexing (only reindexes changed files)
lit-pipeline index output/

# Force rebuild all indexes
lit-pipeline index output/ --force-rebuild
```

**Performance:** No changes: 60s → 2s (30x faster!)

## Next Steps

See documentation for more:
- [PERFORMANCE.md](PERFORMANCE.md) - Parallel processing & incremental indexing
- [ERROR_HANDLING.md](ERROR_HANDLING.md) - Checkpoint/resume & error recovery
- [NEXT_STEPS.md](NEXT_STEPS.md) - Roadmap and future enhancements
- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical design

---

## Getting Help

**Documentation:**
- [NEXT_STEPS.md](NEXT_STEPS.md) - Roadmap and remaining work
- [CLAUDE.md](CLAUDE.md) - Implementation guidance
- [README.md](README.md) - Project overview

**Key Files:**
- `run_pipeline.py` - Main pipeline runner
- `lit_doc_retriever.py` - Search CLI
- `llm_enrichment.py` - Enrichment module
- `citation_tracker.py` - Citation extraction
- `chunk_documents.py` - Chunking logic

**Tests:**
```bash
.venv/bin/python -m pytest tests/ -v
```

# Litigation Document Processing Pipeline

A Python-based system for converting legal documents (PDFs, DOCX, etc.) into structured, searchable formats optimized for LLM-assisted legal analysis. The pipeline preserves precise citation information (page numbers, Bates stamps, line numbers) required for legal work.

## Features

- **Accurate Citation Tracking**: Preserves page:line, column:line, and paragraph citations
- **Multi-Format Support**: PDF, DOCX, XLSX, EML, MSG, PPTX
- **Hybrid Search**: BM25 + semantic vector search with optional cross-encoder reranking
- **Incremental Indexing**: Add new documents without rebuilding entire corpus
- **LLM Enrichment**: Optional summaries and categorization

## Installation

```bash
# Install core dependencies
pip install -r requirements.txt

# Install Docling (requires Python 3.10+)
pip install docling

# Optional: Install Ollama for embeddings
# https://ollama.ai/
ollama pull nomic-embed-text    # 274MB
```

## Quick Start

```bash
# Process documents
lit-pipeline process tests/test_docs output/

# Process with enrichment
lit-pipeline process tests/test_docs output/ --enrich --case-type patent

# Build search index
lit-pipeline index output/

# Search with reranking
lit-pipeline search output/ "witness testimony about TWT technology" --rerank

# Show statistics
lit-pipeline stats output/
```

**Note:** Use `.venv/bin/python lit_pipeline.py` if not installed globally.

## Project Status

**Current Status:** Phases 1-5 Complete ✅

- ✅ **Phase 1-2:** Citation tracking, chunking, post-processing
- ✅ **Phase 3:** Hybrid search (BM25 + semantic vector)
- ✅ **Phase 4:** Cross-encoder reranker
- ✅ **Phase 5:** LLM enrichment (Ollama/Anthropic/Claude Code backends)

**Test Coverage:** 115 tests (99 passing, 16 skipped)

**Citation Quality:**
- Depositions: 100% line-level accuracy
- Expert Reports: 99.2% paragraph detection
- Patents: 84.5% column detection on spec pages
- Bates stamps: Sequential validation with gap detection

**Documentation:**
- [QUICK_START.md](QUICK_START.md) - Getting started guide
- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical architecture and design
- [NEXT_STEPS.md](NEXT_STEPS.md) - Roadmap and remaining work
- [CLAUDE.md](CLAUDE.md) - Implementation guidance

## License

MIT

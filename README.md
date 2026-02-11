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
# Process documents (with optional enrichment)
python run_pipeline.py --input-dir tests/test_docs --output-dir output --enrich

# Build search index
python lit_doc_retriever.py --index-dir output --build-index

# Search with reranking
python lit_doc_retriever.py --index-dir output \
  --query "witness testimony about TWT technology" \
  --rerank --top-k 10
```

## Project Status

**Current Status:** Phases 1-5 Complete ✅

- ✅ **Phase 1-2:** Citation tracking, chunking, post-processing
- ✅ **Phase 3:** Hybrid search (BM25 + semantic vector)
- ✅ **Phase 4:** Cross-encoder reranker
- ✅ **Phase 5:** LLM enrichment (Ollama/Anthropic/Claude Code backends)

**Test Coverage:** 109 tests (93 passing, 16 skipped)

See [NEXT_STEPS.md](NEXT_STEPS.md) for remaining work and [CLAUDE.md](CLAUDE.md) for implementation guidance.

## License

MIT

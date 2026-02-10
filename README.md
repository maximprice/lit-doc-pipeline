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
doc-pipeline --input-dir litigation_docs/ --output-dir processed/

# Search
doc-retrieve processed/ --query "witness testimony about TWT technology" --top-k 10
```

## Project Status

**Current Phase:** Implementation Phase 1 - Conversion & Citation Extraction

See [CLAUDE.md](CLAUDE.md) for implementation guidance and [LITIGATION_DOCUMENT_PIPELINE_TRD.md](LITIGATION_DOCUMENT_PIPELINE_TRD.md) for complete technical specification.

## License

MIT

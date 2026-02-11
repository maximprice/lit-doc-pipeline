#!/usr/bin/env python3
"""
Unified CLI for the litigation document processing pipeline.

Usage:
    lit-pipeline process <input-dir> <output-dir> [options]
    lit-pipeline index <output-dir> [options]
    lit-pipeline search <index-dir> <query> [options]
    lit-pipeline stats <index-dir>
    lit-pipeline enrich <chunks-dir> [options]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_process(args):
    """Run the document processing pipeline (Steps 1-3)."""
    from run_pipeline import run_pipeline

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    use_existing = Path(args.use_existing) if args.use_existing else None

    if not input_dir.exists():
        logger.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    run_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        use_existing=use_existing,
        cleanup_json=args.cleanup_json,
        enrich=args.enrich,
        enrich_backend=args.enrich_backend,
        case_type=args.case_type,
        parties=args.parties,
    )


def cmd_index(args):
    """Build search indexes from processed chunks."""
    from lit_doc_retriever import build_indexes

    output_dir = args.output_dir
    config_path = args.config

    build_indexes(output_dir, config_path)


def cmd_search(args):
    """Search indexed documents."""
    from lit_doc_retriever import search_and_display

    index_dir = args.index_dir
    query = args.query
    top_k = args.top_k
    mode = args.mode
    rerank = args.rerank

    search_and_display(index_dir, query, top_k, mode, rerank)


def cmd_stats(args):
    """Show index statistics."""
    from lit_doc_retriever import show_stats

    show_stats(args.index_dir)


def cmd_enrich(args):
    """Run LLM enrichment on chunk files."""
    from llm_enrichment import LLMEnricher, CaseContext

    chunks_dir = Path(args.chunks_dir)
    if not chunks_dir.exists():
        logger.error("Chunks directory not found: %s", chunks_dir)
        sys.exit(1)

    case_context = CaseContext(
        case_type=args.case_type,
        parties=[p.strip() for p in args.parties.split(",") if p.strip()] if args.parties else [],
    )

    enricher = LLMEnricher(backend=args.backend, delay_between_calls=args.delay)

    if not enricher.is_available():
        logger.error("Enrichment backend '%s' not available", args.backend)
        sys.exit(1)

    stats = enricher.enrich_directory(str(chunks_dir), case_context, force=args.force)
    print(f"\n{stats.summary()}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="lit-pipeline",
        description="Litigation document processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process documents
  lit-pipeline process tests/test_docs output/

  # Process with enrichment
  lit-pipeline process tests/test_docs output/ --enrich --case-type patent

  # Use config file
  lit-pipeline --config my-config.json process tests/test_docs output/

  # Build indexes
  lit-pipeline index output/

  # Search
  lit-pipeline search output/ "TWT technology" --rerank

  # Show stats
  lit-pipeline stats output/

  # Enrich existing chunks
  lit-pipeline enrich output/converted/ --backend ollama
        """
    )

    # Global options (before subcommand)
    parser.add_argument(
        "--config",
        help="Path to config file (JSON or YAML)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── Process command ──────────────────────────────────────────────────
    process_parser = subparsers.add_parser(
        "process",
        help="Process PDF documents through the pipeline",
        description="Convert PDFs to markdown, extract citations, and create chunks"
    )
    process_parser.add_argument(
        "input_dir",
        help="Directory containing PDF files"
    )
    process_parser.add_argument(
        "output_dir",
        help="Directory for pipeline output"
    )
    process_parser.add_argument(
        "--use-existing",
        help="Path to existing converted files (skip Docling conversion)"
    )
    process_parser.add_argument(
        "--cleanup-json",
        action="store_true",
        default=True,
        help="Delete Docling JSON files after processing (default: True)"
    )
    process_parser.add_argument(
        "--no-cleanup-json",
        dest="cleanup_json",
        action="store_false",
        help="Keep Docling JSON files"
    )
    process_parser.add_argument(
        "--enrich",
        action="store_true",
        help="Run LLM enrichment after processing"
    )
    process_parser.add_argument(
        "--enrich-backend",
        choices=["ollama", "anthropic"],
        default="ollama",
        help="LLM backend for enrichment (default: ollama)"
    )
    process_parser.add_argument(
        "--case-type",
        default="patent",
        help="Case type for enrichment context (default: patent)"
    )
    process_parser.add_argument(
        "--parties",
        default="",
        help="Comma-separated party names for enrichment"
    )

    # ── Index command ────────────────────────────────────────────────────
    index_parser = subparsers.add_parser(
        "index",
        help="Build search indexes from processed chunks",
        description="Build BM25 and vector indexes for fast search"
    )
    index_parser.add_argument(
        "output_dir",
        help="Directory containing pipeline output (with converted/ subdirectory)"
    )
    index_parser.add_argument(
        "--config",
        help="Path to retrieval config JSON"
    )

    # ── Search command ───────────────────────────────────────────────────
    search_parser = subparsers.add_parser(
        "search",
        help="Search indexed documents",
        description="Search using BM25, semantic, or hybrid mode with optional reranking"
    )
    search_parser.add_argument(
        "index_dir",
        help="Directory containing indexes"
    )
    search_parser.add_argument(
        "query",
        help="Search query"
    )
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to return (default: 10)"
    )
    search_parser.add_argument(
        "--mode",
        choices=["bm25", "semantic", "hybrid"],
        default="hybrid",
        help="Search mode (default: hybrid)"
    )
    search_parser.add_argument(
        "--rerank",
        action="store_true",
        help="Apply cross-encoder reranking"
    )

    # ── Stats command ────────────────────────────────────────────────────
    stats_parser = subparsers.add_parser(
        "stats",
        help="Show index statistics",
        description="Display index statistics and document breakdown"
    )
    stats_parser.add_argument(
        "index_dir",
        help="Directory containing indexes"
    )

    # ── Enrich command ───────────────────────────────────────────────────
    enrich_parser = subparsers.add_parser(
        "enrich",
        help="Run LLM enrichment on chunks",
        description="Add summaries, categories, and relevance scores to existing chunks"
    )
    enrich_parser.add_argument(
        "chunks_dir",
        help="Directory containing *_chunks.json files"
    )
    enrich_parser.add_argument(
        "--backend",
        choices=["ollama", "anthropic"],
        default="ollama",
        help="LLM backend (default: ollama)"
    )
    enrich_parser.add_argument(
        "--case-type",
        default="patent",
        help="Case type for context (default: patent)"
    )
    enrich_parser.add_argument(
        "--parties",
        default="",
        help="Comma-separated party names"
    )
    enrich_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-enrich already-enriched chunks"
    )
    enrich_parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between API calls in seconds (default: 0.1)"
    )

    # Parse args
    args = parser.parse_args()

    # Set logging level
    if hasattr(args, 'verbose') and args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config if specified
    if hasattr(args, 'config') and args.config:
        from config_loader import ConfigLoader
        config = ConfigLoader(args.config)
        # Make config available to command handlers
        args._config = config
    else:
        args._config = None

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Route to command handlers
    if args.command == "process":
        cmd_process(args)
    elif args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "enrich":
        cmd_enrich(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

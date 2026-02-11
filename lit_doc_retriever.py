#!/usr/bin/env python3
"""
CLI tool for searching litigation documents.

This tool provides command-line access to hybrid search (BM25 + semantic)
over processed litigation document chunks.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from citation_types import Chunk, DocumentType

# Delay imports to avoid chromadb import issues
def _import_indexers():
    """Lazy import of indexers to avoid chromadb import-time errors."""
    from bm25_indexer import BM25Indexer
    from vector_indexer import VectorIndexer
    from hybrid_retriever import HybridRetriever
    return BM25Indexer, VectorIndexer, HybridRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def build_indexes(output_dir: str, config_path: Optional[str] = None) -> None:
    """
    Build BM25 and vector indexes from chunk files.

    Args:
        output_dir: Directory containing chunk JSON files
        config_path: Path to retrieval config (optional)
    """
    BM25Indexer, VectorIndexer, _ = _import_indexers()

    output_path = Path(output_dir)
    converted_dir = output_path / "converted"
    index_dir = output_path / "indexes"

    if not converted_dir.exists():
        logger.error(f"Chunks directory not found: {converted_dir}")
        sys.exit(1)

    # Load config if provided
    config = {}
    if config_path:
        with open(config_path, 'r') as f:
            config = json.load(f)

    # Load chunks
    logger.info("Loading chunks...")
    chunks = []
    chunk_files = list(converted_dir.glob("*_chunks.json"))

    if not chunk_files:
        logger.error(f"No chunk files found in {converted_dir}")
        sys.exit(1)

    for chunk_file in chunk_files:
        try:
            with open(chunk_file, 'r') as f:
                chunks_data = json.load(f)

            for chunk_data in chunks_data:
                chunk = Chunk(
                    chunk_id=chunk_data["chunk_id"],
                    core_text=chunk_data["core_text"],
                    pages=chunk_data["pages"],
                    citation=chunk_data["citation"],
                    citation_string=chunk_data["citation_string"],
                    key_quotes=chunk_data.get("key_quotes", []),
                    tokens=chunk_data.get("tokens", 0),
                    doc_type=DocumentType(chunk_data.get("doc_type", "unknown")),
                    summary=chunk_data.get("summary"),
                    category=chunk_data.get("category"),
                    relevance_score=chunk_data.get("relevance_score"),
                    claims_addressed=chunk_data.get("claims_addressed"),
                    classification_method=chunk_data.get("classification_method"),
                    llm_backend=chunk_data.get("llm_backend"),
                )
                chunks.append(chunk)

        except Exception as e:
            logger.error(f"Error loading {chunk_file}: {e}")
            continue

    logger.info(f"Loaded {len(chunks)} chunks from {len(chunk_files)} files")

    # Build BM25 index
    logger.info("\nBuilding BM25 index...")
    bm25_config = config.get("bm25", {})
    bm25_indexer = BM25Indexer(
        index_dir=str(index_dir),
        k1=bm25_config.get("k1", 1.5),
        b=bm25_config.get("b", 0.75),
        max_features=bm25_config.get("max_features", 10000),
        ngram_range=tuple(bm25_config.get("ngram_range", [1, 2]))
    )

    start_time = time.time()
    bm25_indexer.build_index(chunks)
    bm25_time = time.time() - start_time
    logger.info(f"BM25 index built in {bm25_time:.2f}s")

    # Build vector index (if Ollama available)
    logger.info("\nBuilding vector index...")
    chroma_config = config.get("chroma", {})
    vector_indexer = VectorIndexer(
        persist_directory=str(index_dir / "chroma_db"),
        embedding_model=chroma_config.get("embedding_model", "nomic-embed-text"),
        ollama_url=chroma_config.get("ollama_url", "http://localhost:11434/api/embeddings")
    )

    if vector_indexer.is_available():
        start_time = time.time()
        vector_indexer.build_index(chunks)
        vector_time = time.time() - start_time
        logger.info(f"Vector index built in {vector_time:.2f}s")
    else:
        logger.warning("Ollama not available, skipping vector index")
        logger.warning("Semantic search will not be available")

    # Save metadata
    metadata = {
        "num_chunks": len(chunks),
        "num_documents": len(chunk_files),
        "bm25_build_time": bm25_time,
        "vector_build_time": vector_time if vector_indexer.is_available() else None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    metadata_path = index_dir / "index_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\nIndexes saved to {index_dir}")
    logger.info("Build complete!")


def show_stats(output_dir: str) -> None:
    """
    Show index statistics.

    Args:
        output_dir: Directory containing indexes
    """
    output_path = Path(output_dir)
    index_dir = output_path / "indexes"
    converted_dir = output_path / "converted"

    if not index_dir.exists():
        logger.error(f"Index directory not found: {index_dir}")
        logger.info(f"Run with --build-index first")
        sys.exit(1)

    # Load metadata
    metadata_path = index_dir / "index_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        print("\n=== Index Statistics ===")
        print(f"Total chunks: {metadata['num_chunks']}")
        print(f"Total documents: {metadata['num_documents']}")
        print(f"BM25 build time: {metadata['bm25_build_time']:.2f}s")
        if metadata.get('vector_build_time'):
            print(f"Vector build time: {metadata['vector_build_time']:.2f}s")
        print(f"Last updated: {metadata['timestamp']}")
    else:
        print("Index metadata not found")

    # Show per-document stats
    try:
        _, _, HybridRetriever = _import_indexers()
        retriever = HybridRetriever(
            index_dir=str(index_dir),
            chunks_dir=str(converted_dir)
        )
        stats = retriever.get_stats()

        print(f"\n=== Document Breakdown ===")
        for doc_name, count in sorted(stats["documents"].items()):
            print(f"  {doc_name}: {count} chunks")

        print(f"\n=== Search Capabilities ===")
        print(f"BM25 search: {'✓' if stats['bm25_available'] else '✗'}")
        print(f"Semantic search: {'✓' if stats['semantic_available'] else '✗'}")
        if not stats['semantic_available']:
            print("  (Ollama not running - install and start Ollama for semantic search)")

    except Exception as e:
        logger.error(f"Error loading retriever: {e}")


def search_and_display(
    output_dir: str,
    query: str,
    top_k: int,
    mode: str,
    rerank: bool = False
) -> None:
    """
    Search and display results.

    Args:
        output_dir: Directory containing indexes
        query: Search query
        top_k: Number of results to return
        mode: Search mode (bm25, semantic, hybrid)
        rerank: Whether to apply cross-encoder reranking
    """
    output_path = Path(output_dir)
    index_dir = output_path / "indexes"
    converted_dir = output_path / "converted"

    if not index_dir.exists():
        logger.error(f"Index directory not found: {index_dir}")
        logger.info(f"Run with --build-index first")
        sys.exit(1)

    # Initialize retriever
    try:
        _, _, HybridRetriever = _import_indexers()
        retriever = HybridRetriever(
            index_dir=str(index_dir),
            chunks_dir=str(converted_dir)
        )
    except Exception as e:
        logger.error(f"Error loading retriever: {e}")
        sys.exit(1)

    # Execute search
    start_time = time.time()
    try:
        results = retriever.search(query, top_k=top_k, mode=mode, rerank=rerank)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        sys.exit(1)

    search_time = time.time() - start_time

    # Display results
    print(f"\n{'=' * 70}")
    print(f"Search Results for \"{query}\"")
    print(f"{'=' * 70}")
    print(f"Mode: {mode}")
    if mode == "hybrid":
        print(f"  (BM25 + semantic vector search)")
    print(f"Found {len(results)} results in {search_time:.2f}s")
    print()

    if not results:
        print("No results found. Try:")
        print("  - Using different keywords")
        print("  - Using broader terms")
        print("  - Checking spelling")
        return

    for result in results:
        print(f"{'\u2500' * 70}")
        print(f"Result {result.rank}/{len(results)} (Score: {result.score:.3f})")
        print(f"{'\u2500' * 70}")

        # Document info
        chunk = result.chunk
        doc_name = "_".join(chunk.chunk_id.split("_")[:-2])
        print(f"Document: {doc_name}")
        print(f"Type: {chunk.doc_type.value if hasattr(chunk.doc_type, 'value') else chunk.doc_type}")
        print(f"Citation: {chunk.citation_string}")
        print(f"Pages: {', '.join(map(str, chunk.pages))}")

        # Enrichment metadata
        if chunk.category or chunk.relevance_score or chunk.summary:
            enrichment_parts = []
            if chunk.category:
                enrichment_parts.append(f"Category: {chunk.category}")
            if chunk.relevance_score:
                enrichment_parts.append(f"Relevance: {chunk.relevance_score}")
            if chunk.claims_addressed:
                enrichment_parts.append(f"Claims: {chunk.claims_addressed}")
            print(f"Enrichment: {', '.join(enrichment_parts)}")
            if chunk.summary:
                print(f"Summary: {chunk.summary}")

        # Scores
        if result.bm25_score is not None or result.semantic_score is not None or result.reranker_score is not None:
            score_parts = []
            if result.bm25_score is not None:
                score_parts.append(f"BM25: {result.bm25_score:.3f}")
            if result.semantic_score is not None:
                score_parts.append(f"Semantic: {result.semantic_score:.3f}")
            if result.reranker_score is not None:
                score_parts.append(f"Reranker: {result.reranker_score:.3f}")
            print(f"Scores: {', '.join(score_parts)}")

        # Preview
        print(f"\nPreview:")
        preview = chunk.core_text[:400]
        if len(chunk.core_text) > 400:
            preview += "..."
        # Clean up whitespace
        preview = " ".join(preview.split())
        print(f'"{preview}"')
        print()

    print(f"{'=' * 70}\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Search litigation documents using hybrid BM25 + semantic search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build indexes
  python lit_doc_retriever.py --index-dir tests/pipeline_output --build-index

  # Search with hybrid mode (default)
  python lit_doc_retriever.py --index-dir tests/pipeline_output \\
      --query "TWT technology battery life"

  # Search with BM25 only
  python lit_doc_retriever.py --index-dir tests/pipeline_output \\
      --query "designated corporate representative" --mode bm25

  # Search with cross-encoder reranking
  python lit_doc_retriever.py --index-dir tests/pipeline_output \\
      --query "TWT technology battery life" --rerank

  # Show statistics
  python lit_doc_retriever.py --index-dir tests/pipeline_output --stats
        """
    )

    parser.add_argument(
        "--index-dir",
        required=True,
        help="Directory containing pipeline output (converted/ and indexes/)"
    )

    parser.add_argument(
        "--query",
        help="Search query"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to return (default: 10)"
    )

    parser.add_argument(
        "--mode",
        choices=["bm25", "semantic", "hybrid"],
        default="hybrid",
        help="Search mode (default: hybrid)"
    )

    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Apply cross-encoder reranking (requires sentence-transformers)"
    )

    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build indexes from chunk files"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show index statistics"
    )

    parser.add_argument(
        "--config",
        help="Path to retrieval config JSON (for --build-index)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.build_index:
        build_indexes(args.index_dir, args.config)
    elif args.stats:
        show_stats(args.index_dir)
    elif args.query:
        search_and_display(args.index_dir, args.query, args.top_k, args.mode, rerank=args.rerank)
    else:
        parser.print_help()
        print("\nError: Must specify --build-index, --stats, or --query")
        sys.exit(1)


if __name__ == "__main__":
    main()

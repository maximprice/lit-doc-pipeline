"""
Parallel document processing for the litigation pipeline.

Enables concurrent processing of multiple documents while maintaining
thread-safe state tracking and error handling.
"""

import logging
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from citation_tracker import CitationTracker
from citation_types import DocumentType
from docling_converter import DoclingConverter
from pipeline_state import PipelineState
from post_processor import PostProcessor
from pymupdf_extractor import is_text_based_pdf, extract_deposition

logger = logging.getLogger(__name__)


def process_single_document(
    pdf_path: Path,
    output_dir: Path,
    normalized: str,
    known_doc_types: Dict[str, DocumentType],
    conversion_timeout: int,
    cleanup_json: bool,
    use_existing: Optional[Path] = None,
) -> Dict:
    """
    Process a single document through the pipeline.

    This function is designed to be called in a separate process.

    Args:
        pdf_path: Path to PDF file
        output_dir: Output directory
        normalized: Normalized document stem
        known_doc_types: Mapping of document stems to types
        conversion_timeout: Timeout for conversion
        cleanup_json: Whether to cleanup JSON files
        use_existing: Path to existing converted files

    Returns:
        Result dictionary with processing status
    """
    converted_dir = output_dir / "converted"
    stem = pdf_path.stem

    try:
        logger.info("Processing: %s (worker PID: %d)", pdf_path.name, os.getpid())

        # ── PyMuPDF fast path for text-based depositions ─────────────
        known_type = known_doc_types.get(normalized)
        if known_type == DocumentType.DEPOSITION and is_text_based_pdf(str(pdf_path)):
            logger.info("[PyMuPDF] Text-based deposition detected")
            pymupdf_result = extract_deposition(str(pdf_path), str(converted_dir))
            logger.info("  Extracted %d lines, %d citations",
                       pymupdf_result["line_count"],
                       pymupdf_result["citation_count"])

            return {
                "file": pdf_path.name,
                "stem": normalized,
                "status": "OK",
                "doc_type": DocumentType.DEPOSITION.value,
                "md_file": Path(pymupdf_result["md_path"]).name,
                "json_file": None,
                "citations_count": pymupdf_result["citation_count"],
                "coverage_pct": 100.0,
                "type_distribution": {"transcript_line": pymupdf_result["citation_count"]},
                "extraction_method": "pymupdf",
            }

        # ── Step 1: Conversion ───────────────────────────────────────
        logger.info("[Step 1] Converting with Docling...")

        # Check for existing files
        existing_json = None
        existing_md = None
        if use_existing:
            existing_json = use_existing / f"{normalized}.json"
            existing_md = use_existing / f"{normalized}.md"

        if use_existing and existing_json and existing_json.exists() and existing_md.exists():
            logger.info("  Using existing converted files")
            # Copy files (handled in main process to avoid race conditions)
            final_md = converted_dir / f"{normalized}.md"
            final_json = converted_dir / f"{normalized}.json"
            conversion_citations = {}
        else:
            converter = DoclingConverter(timeout=conversion_timeout)
            result = converter.convert_document(str(pdf_path), str(converted_dir))

            if result.errors and not result.md_path:
                error_msg = result.errors[0] if result.errors else "Conversion failed"
                logger.error("  FAILED: %s", error_msg)
                return {
                    "file": pdf_path.name,
                    "stem": normalized,
                    "status": "FAILED",
                    "error": error_msg,
                }

            conversion_citations = result.citations_found

            # Rename if needed
            docling_md = converted_dir / f"{stem}.md"
            docling_json = converted_dir / f"{stem}.json"
            target_md = converted_dir / f"{normalized}.md"
            target_json = converted_dir / f"{normalized}.json"

            if docling_md.exists() and docling_md != target_md:
                docling_md.rename(target_md)
            if docling_json.exists() and docling_json != target_json:
                docling_json.rename(target_json)

            final_md = target_md
            final_json = target_json

        # Verify output
        if not final_md.exists():
            error_msg = f"No .md file at {final_md}"
            logger.error("  %s", error_msg)
            return {
                "file": pdf_path.name,
                "stem": normalized,
                "status": "FAILED",
                "error": error_msg,
            }

        has_json = final_json.exists()
        logger.info("  Output: %s%s", final_md.name,
                   f" + {final_json.name}" if has_json else " (no JSON)")

        # ── Step 2: Post-processing ──────────────────────────────────
        logger.info("[Step 2] Post-processing markdown...")

        # Detect document type
        has_lines = len(conversion_citations.get("line_markers", [])) > 0
        has_columns = len(conversion_citations.get("column_markers", [])) > 0
        has_paragraphs = len(conversion_citations.get("paragraph_markers", [])) > 0

        if known_type:
            doc_type = known_type
        elif has_lines:
            doc_type = DocumentType.DEPOSITION
        elif has_columns:
            doc_type = DocumentType.PATENT
        elif has_paragraphs:
            doc_type = DocumentType.EXPERT_REPORT
        else:
            doc_type = DocumentType.UNKNOWN

        logger.info("  Document type: %s", doc_type.value)

        processor = PostProcessor()
        proc_result = processor.process(str(final_md), doc_type)
        logger.info("  Post-processor citation coverage: %d elements",
                   proc_result.citation_coverage)

        # ── Step 3: Citation tracking (bbox-based) ───────────────────
        citations = {}
        metrics = None

        if has_json:
            logger.info("[Step 3] Reconstructing citations from JSON bbox data...")
            try:
                tracker = CitationTracker(
                    converted_dir=str(converted_dir),
                    doc_type=doc_type,
                )
                citations = tracker.reconstruct_citations(normalized)
                metrics = tracker.validate(citations)

                logger.info("  Citations: %d items, %.1f%% coverage",
                           metrics.total_items, metrics.coverage_pct)
                logger.info("  Types: %s", metrics.type_distribution)

                # Cleanup JSON
                if cleanup_json and final_json.exists():
                    json_size = final_json.stat().st_size
                    final_json.unlink()
                    logger.info("  Cleaned up Docling JSON (%d bytes freed)", json_size)

            except Exception as e:
                logger.error("Citation tracking error: %s", e)
                logger.warning("  Continuing without bbox-based citations")
        else:
            logger.warning("[Step 3] Skipped — no JSON file")

        # Return result
        return {
            "file": pdf_path.name,
            "stem": normalized,
            "status": "OK",
            "doc_type": doc_type.value,
            "md_file": final_md.name,
            "json_file": final_json.name if has_json and final_json.exists() else None,
            "citations_count": len(citations),
            "coverage_pct": metrics.coverage_pct if metrics else 0.0,
            "type_distribution": metrics.type_distribution if metrics else {},
        }

    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "file": pdf_path.name,
            "stem": normalized,
            "status": "FAILED",
            "error": error_msg,
        }


def process_documents_parallel(
    pdfs: List[Path],
    output_dir: Path,
    normalized_stems: Dict[str, str],
    known_doc_types: Dict[str, DocumentType],
    state: PipelineState,
    conversion_timeout: int = 300,
    cleanup_json: bool = True,
    use_existing: Optional[Path] = None,
    max_workers: Optional[int] = None,
    force: bool = False,
    skip_failed: bool = True,
) -> List[Dict]:
    """
    Process multiple documents in parallel.

    Args:
        pdfs: List of PDF paths to process
        output_dir: Output directory
        normalized_stems: Mapping of original stems to normalized stems
        known_doc_types: Mapping of normalized stems to document types
        state: Pipeline state for tracking progress
        conversion_timeout: Timeout for conversion
        cleanup_json: Whether to cleanup JSON files
        use_existing: Path to existing converted files
        max_workers: Number of parallel workers (default: cpu_count - 1)
        force: Force reprocessing
        skip_failed: Skip documents that have failed multiple times

    Returns:
        List of result dictionaries
    """
    if max_workers is None:
        max_workers = max(1, mp.cpu_count() - 1)

    logger.info("Processing %d documents with %d workers", len(pdfs), max_workers)

    results = []

    # Filter documents based on state
    pdfs_to_process = []
    for pdf_path in pdfs:
        stem = pdf_path.stem
        normalized = normalized_stems.get(stem, stem.lower())
        doc_state = state.get_document(normalized, pdf_path.name)

        # Skip if completed and not forcing
        if not force and doc_state.status == "completed":
            logger.info("Skipping %s (already completed)", pdf_path.name)
            results.append({
                "file": pdf_path.name,
                "stem": normalized,
                "status": "SKIPPED",
                "reason": "Already completed",
            })
            continue

        # Skip if failed too many times
        if skip_failed and doc_state.status == "failed" and doc_state.retry_count >= 3:
            logger.info("Skipping %s (failed %d times)", pdf_path.name, doc_state.retry_count)
            results.append({
                "file": pdf_path.name,
                "stem": normalized,
                "status": "SKIPPED",
                "reason": f"Failed {doc_state.retry_count} times",
            })
            continue

        pdfs_to_process.append(pdf_path)

    if not pdfs_to_process:
        logger.info("No documents to process")
        return results

    logger.info("Processing %d documents (skipped %d)",
               len(pdfs_to_process), len(pdfs) - len(pdfs_to_process))

    # Process documents in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_pdf = {}
        for pdf_path in pdfs_to_process:
            stem = pdf_path.stem
            normalized = normalized_stems.get(stem, stem.lower())

            future = executor.submit(
                process_single_document,
                pdf_path,
                output_dir,
                normalized,
                known_doc_types,
                conversion_timeout,
                cleanup_json,
                use_existing,
            )
            future_to_pdf[future] = (pdf_path, normalized)

        # Collect results as they complete
        for future in as_completed(future_to_pdf):
            pdf_path, normalized = future_to_pdf[future]

            try:
                result = future.result()
                results.append(result)

                # Update state (thread-safe)
                doc_state = state.get_document(normalized, pdf_path.name)

                if result["status"] == "OK":
                    doc_state.mark_stage_complete("conversion")
                    doc_state.mark_stage_complete("post_processing")
                    doc_state.mark_stage_complete("citation_tracking")
                    doc_state.mark_completed()
                    logger.info("✓ Completed: %s", pdf_path.name)
                else:
                    doc_state.mark_failed(result.get("error", "Unknown error"))
                    logger.error("✗ Failed: %s - %s", pdf_path.name, result.get("error"))

                # Save state after each document
                state.save()

            except Exception as e:
                error_msg = f"Worker error: {str(e)}"
                logger.error(error_msg, exc_info=True)

                doc_state = state.get_document(normalized, pdf_path.name)
                doc_state.mark_failed(error_msg)
                state.save()

                results.append({
                    "file": pdf_path.name,
                    "stem": normalized,
                    "status": "FAILED",
                    "error": error_msg,
                })

    return results


def get_optimal_worker_count() -> int:
    """
    Get optimal number of workers based on system resources.

    Returns:
        Recommended worker count
    """
    cpu_count = mp.cpu_count()

    # Leave one core for system
    optimal = max(1, cpu_count - 1)

    # Cap at 8 workers to avoid overwhelming the system
    return min(optimal, 8)

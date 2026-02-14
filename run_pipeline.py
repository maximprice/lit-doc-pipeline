#!/usr/bin/env python3
"""
Run pipeline steps 1-5 on PDF documents:
  1. Convert (Docling) → .md + .json
  2. Post-process → cleaned .md + initial _citations.json
  3. Citation tracking → bbox-based _citations.json (overwrites step 2's)
  4. Chunking → *_chunks.json files
  5. LLM enrichment (optional) → enriched chunks

Usage:
    python run_pipeline.py --input-dir tests/test_docs --output-dir tests/pipeline_output
"""

import argparse
import json
import logging
import os
import shutil
import sys
import traceback
from pathlib import Path

from tqdm import tqdm

from citation_tracker import CitationTracker
from citation_types import DocumentType
from chunk_documents import chunk_all_documents
from docling_converter import DoclingConverter
from parallel_processor import process_documents_parallel, get_optimal_worker_count
from pipeline_state import PipelineState
from post_processor import PostProcessor
from pymupdf_extractor import is_text_based_pdf, extract_deposition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def should_disable_tqdm():
    """Check if progress bars should be disabled."""
    return (os.environ.get('TQDM_DISABLE', '0') == '1' or
            os.environ.get('PYTEST_CURRENT_TEST') is not None or
            os.environ.get('CI', '').lower() == 'true')


# Map known document stems to their types
KNOWN_DOC_TYPES = {
    "daniel_alexander_10_24_2025": DocumentType.DEPOSITION,
    "intel_prox_00006214": DocumentType.PATENT,
    # IEEE standards — not traditional patent layout, use generic
    "intel_prox_00001770": DocumentType.UNKNOWN,
    "intel_prox_00002058": DocumentType.UNKNOWN,
    "intel_prox_00002382": DocumentType.UNKNOWN,
}


def normalize_stem(name: str) -> str:
    """Normalize a filename stem to lowercase with underscores."""
    import re
    result = name.lower().replace(" ", "_").replace("-", "_")
    result = re.sub(r"_+", "_", result)  # collapse multiple underscores
    return result.strip("_")


def detect_doc_type(stem: str, conversion_citations: dict) -> DocumentType:
    """Detect document type from known map or conversion citations."""
    normalized = normalize_stem(stem)
    if normalized in KNOWN_DOC_TYPES:
        return KNOWN_DOC_TYPES[normalized]

    # Fall back to heuristics from conversion
    has_lines = len(conversion_citations.get("line_markers", [])) > 0
    has_columns = len(conversion_citations.get("column_markers", [])) > 0
    has_paragraphs = len(conversion_citations.get("paragraph_markers", [])) > 0

    if has_lines:
        return DocumentType.DEPOSITION
    elif has_columns:
        return DocumentType.PATENT
    elif has_paragraphs:
        return DocumentType.EXPERT_REPORT
    return DocumentType.UNKNOWN


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    use_existing: Path = None,
    cleanup_json: bool = True,
    enrich: bool = False,
    enrich_backend: str = "ollama",
    case_type: str = "patent",
    parties: str = "",
    resume: bool = False,
    force: bool = False,
    skip_failed: bool = True,
    conversion_timeout: int = 300,
    parallel: bool = False,
    max_workers: int = None,
):
    """Run steps 1-3 (+ optional enrichment) on all PDFs in input_dir.

    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory for pipeline output
        use_existing: Path to existing converted files (skip Docling)
        cleanup_json: Delete large Docling JSON files after processing (default: True)
        enrich: Whether to run LLM enrichment after chunking
        enrich_backend: LLM backend for enrichment (ollama or anthropic)
        case_type: Case type for enrichment context
        parties: Comma-separated party names for enrichment context
        resume: Resume from previous run using saved state
        force: Force reprocessing even if stage already complete
        skip_failed: Skip documents that have failed multiple times
        conversion_timeout: Timeout in seconds for document conversion (default: 300)
        parallel: Enable parallel processing of documents
        max_workers: Number of parallel workers (default: cpu_count - 1)
    """
    converted_dir = output_dir / "converted"
    converted_dir.mkdir(parents=True, exist_ok=True)

    # Initialize pipeline state for checkpoint/resume
    state = PipelineState(output_dir)

    if resume:
        logger.info("Resuming from previous run")
        logger.info(state.summary())

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        logger.error("No PDF files found in %s", input_dir)
        sys.exit(1)

    logger.info("Found %d PDF files in %s", len(pdfs), input_dir)

    # Check if parallel processing is requested
    if parallel:
        if max_workers is None:
            max_workers = get_optimal_worker_count()

        logger.info("Using parallel processing with %d workers", max_workers)

        # Build normalized stems map
        normalized_stems = {pdf.stem: normalize_stem(pdf.stem) for pdf in pdfs}

        # Process documents in parallel
        results = process_documents_parallel(
            pdfs=pdfs,
            output_dir=output_dir,
            normalized_stems=normalized_stems,
            known_doc_types=KNOWN_DOC_TYPES,
            state=state,
            conversion_timeout=conversion_timeout,
            cleanup_json=cleanup_json,
            use_existing=use_existing,
            max_workers=max_workers,
            force=force,
            skip_failed=skip_failed,
        )

        # Skip to enrichment step
        logger.info("Parallel processing complete")

    else:
        # Sequential processing (original code)
        results = []
        disable_progress = should_disable_tqdm()
        pbar = tqdm(pdfs, desc="Processing documents", unit="doc", disable=disable_progress)

        for pdf_path in pbar:
            stem = pdf_path.stem
            normalized = normalize_stem(stem)

            # Get document state
            doc_state = state.get_document(normalized, pdf_path.name)

            # Skip if document completed and not forcing
            if not force and doc_state.status == "completed":
                logger.info("Skipping %s (already completed)", pdf_path.name)
                results.append({
                    "file": pdf_path.name,
                    "stem": normalized,
                    "status": "SKIPPED",
                    "reason": "Already completed",
                })
                continue

            # Skip if document failed too many times
            if skip_failed and doc_state.status == "failed" and doc_state.retry_count >= 3:
                logger.info("Skipping %s (failed %d times: %s)",
                           pdf_path.name, doc_state.retry_count, doc_state.error)
                results.append({
                    "file": pdf_path.name,
                    "stem": normalized,
                    "status": "SKIPPED",
                    "reason": f"Failed {doc_state.retry_count} times",
                })
                continue

            logger.info("=" * 60)
            logger.info("Processing: %s", pdf_path.name)
            pbar.set_postfix_str(f"{pdf_path.name[:30]}...")

            try:
                # ── PyMuPDF fast path for text-based depositions ─────────────
                known_type = KNOWN_DOC_TYPES.get(normalized)
                if (known_type == DocumentType.DEPOSITION and
                    is_text_based_pdf(str(pdf_path)) and
                    state.should_process_document(normalized, "conversion", force)):

                    logger.info("[PyMuPDF] Text-based deposition detected, using direct extraction")
                    try:
                        pymupdf_result = extract_deposition(str(pdf_path), str(converted_dir))
                        logger.info(
                            "  Extracted %d lines, %d citations",
                            pymupdf_result["line_count"],
                            pymupdf_result["citation_count"],
                        )

                        # Mark all stages complete for PyMuPDF extraction
                        doc_state.mark_stage_complete("conversion")
                        doc_state.mark_stage_complete("post_processing")
                        doc_state.mark_stage_complete("citation_tracking")
                        doc_state.mark_completed()
                        state.save()

                        results.append({
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
                        })
                        continue

                    except Exception as e:
                        logger.error("PyMuPDF extraction failed: %s", e)
                        logger.info("Falling back to Docling conversion")
                        # Fall through to Docling conversion

            except Exception as e:
                error_msg = f"Pre-processing error: {str(e)}"
                logger.error(error_msg)
                doc_state.mark_failed(error_msg)
                state.save()
                results.append({
                    "file": pdf_path.name,
                    "stem": normalized,
                    "status": "FAILED",
                    "error": error_msg,
                })
                continue

            # ── Step 1: Conversion ───────────────────────────────────────
            if state.should_process_document(normalized, "conversion", force):
                logger.info("[Step 1] Converting with Docling...")

                try:
                    # Check if we should use pre-existing converted files
                    existing_json = None
                    existing_md = None
                    if use_existing:
                        existing_json = use_existing / f"{normalized}.json"
                        existing_md = use_existing / f"{normalized}.md"

                    if use_existing and existing_json.exists() and existing_md.exists():
                        logger.info("  Using existing converted files from %s", use_existing)
                        # Copy to our output dir
                        target_json = converted_dir / f"{normalized}.json"
                        target_md = converted_dir / f"{normalized}.md"
                        if not target_json.exists():
                            shutil.copy2(existing_json, target_json)
                        if not target_md.exists():
                            shutil.copy2(existing_md, target_md)
                        # Also copy bates sidecar if present
                        existing_bates = use_existing / f"{normalized}_bates.json"
                        if existing_bates.exists():
                            target_bates = converted_dir / f"{normalized}_bates.json"
                            if not target_bates.exists():
                                shutil.copy2(existing_bates, target_bates)

                        md_path = str(target_md)
                        conversion_citations = {}
                        conversion_errors = []
                    else:
                        converter = DoclingConverter(timeout=conversion_timeout)
                        result = converter.convert_document(str(pdf_path), str(converted_dir))
                        md_path = result.md_path
                        conversion_citations = result.citations_found
                        conversion_errors = result.errors

                        if conversion_errors:
                            logger.warning("  Conversion errors: %s", conversion_errors)
                            if not md_path:
                                error_msg = conversion_errors[0] if conversion_errors else "No output produced"
                                logger.error("  FAILED: %s", error_msg)
                                doc_state.mark_failed(error_msg)
                                state.save()
                                results.append({
                                    "file": pdf_path.name,
                                    "stem": normalized,
                                    "status": "FAILED",
                                    "error": error_msg,
                                })
                                continue

                        # Docling may produce files with original stem; rename if needed
                        docling_md = converted_dir / f"{stem}.md"
                        docling_json = converted_dir / f"{stem}.json"
                        target_md = converted_dir / f"{normalized}.md"
                        target_json = converted_dir / f"{normalized}.json"

                        if docling_md.exists() and docling_md != target_md:
                            docling_md.rename(target_md)
                            md_path = str(target_md)
                        if docling_json.exists() and docling_json != target_json:
                            docling_json.rename(target_json)

                    # Verify we have the files we need
                    final_md = converted_dir / f"{normalized}.md"
                    final_json = converted_dir / f"{normalized}.json"

                    if not final_md.exists():
                        error_msg = f"No .md file at {final_md}"
                        logger.error("  %s", error_msg)
                        doc_state.mark_failed(error_msg)
                        state.save()
                        results.append({
                            "file": pdf_path.name,
                            "stem": normalized,
                            "status": "FAILED",
                            "error": error_msg,
                        })
                        continue

                    has_json = final_json.exists()
                    logger.info("  Output: %s%s",
                               final_md.name,
                               f" + {final_json.name}" if has_json else " (no JSON)")

                    # Mark conversion stage complete
                    doc_state.mark_stage_complete("conversion")
                    state.save()

                except Exception as e:
                    error_msg = f"Conversion error: {str(e)}"
                    logger.error(error_msg)
                    logger.debug(traceback.format_exc())
                    doc_state.mark_failed(error_msg)
                    state.save()
                    results.append({
                        "file": pdf_path.name,
                        "stem": normalized,
                        "status": "FAILED",
                        "error": error_msg,
                    })
                    continue
            else:
                logger.info("[Step 1] Skipping conversion (already complete)")
                final_md = converted_dir / f"{normalized}.md"
                final_json = converted_dir / f"{normalized}.json"
                has_json = final_json.exists()
                conversion_citations = {}

            # ── Step 2: Post-processing ──────────────────────────────────
            if state.should_process_document(normalized, "post_processing", force):
                logger.info("[Step 2] Post-processing markdown...")

                try:
                    doc_type = detect_doc_type(normalized, conversion_citations)
                    logger.info("  Document type: %s", doc_type.value)

                    processor = PostProcessor()
                    proc_result = processor.process(str(final_md), doc_type)
                    logger.info("  Post-processor citation coverage: %d elements", proc_result.citation_coverage)

                    # Mark post-processing stage complete
                    doc_state.mark_stage_complete("post_processing")
                    state.save()

                except Exception as e:
                    error_msg = f"Post-processing error: {str(e)}"
                    logger.error(error_msg)
                    logger.debug(traceback.format_exc())
                    doc_state.mark_failed(error_msg)
                    state.save()
                    results.append({
                        "file": pdf_path.name,
                        "stem": normalized,
                        "status": "FAILED",
                        "error": error_msg,
                    })
                    continue
            else:
                logger.info("[Step 2] Skipping post-processing (already complete)")
                # Detect doc type even if skipping
                doc_type = detect_doc_type(normalized, conversion_citations)

            # ── Step 3: Citation tracking (bbox-based) ───────────────────
            if has_json and state.should_process_document(normalized, "citation_tracking", force):
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
                    if metrics.line_gaps:
                        logger.warning("  Line gaps: %s", metrics.line_gaps[:3])

                    # Mark citation tracking stage complete
                    doc_state.mark_stage_complete("citation_tracking")
                    state.save()

                    # Cleanup: Delete large Docling JSON file (keep _citations.json)
                    if cleanup_json and final_json.exists():
                        json_size = final_json.stat().st_size
                        final_json.unlink()
                        logger.info("  Cleaned up Docling JSON (%d bytes freed)", json_size)

                except Exception as e:
                    error_msg = f"Citation tracking error: {str(e)}"
                    logger.error(error_msg)
                    logger.debug(traceback.format_exc())
                    # Don't fail the document, just log warning
                    logger.warning("  Continuing without bbox-based citations")
                    citations = {}
                    metrics = None
                    # Still mark stage complete (partial success)
                    doc_state.mark_stage_complete("citation_tracking")
                    state.save()

            elif not has_json:
                logger.warning("[Step 3] Skipped — no JSON file for bbox-based reconstruction")
                citations = {}
                metrics = None
                doc_state.mark_stage_complete("citation_tracking")
                state.save()
            else:
                logger.info("[Step 3] Skipping citation tracking (already complete)")
                # Load existing citations if available
                citations_file = converted_dir / f"{normalized}_citations.json"
                if citations_file.exists():
                    with open(citations_file) as f:
                        citations = json.load(f)
                    metrics = None  # Don't recompute metrics
                else:
                    citations = {}
                    metrics = None

            # Mark document as completed (all main stages done)
            doc_state.mark_completed()
            state.save()

            # Collect results
            results.append({
                "file": pdf_path.name,
                "stem": normalized,
                "status": "OK",
                "doc_type": doc_type.value,
                "md_file": final_md.name,
                "json_file": final_json.name if has_json and final_json.exists() else None,
                "citations_count": len(citations),
                "coverage_pct": metrics.coverage_pct if metrics else 0.0,
                "type_distribution": metrics.type_distribution if metrics else {},
            })

    # ── Step 4: Chunking ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("[Step 4] Chunking documents...")

    try:
        all_chunks = chunk_all_documents(str(converted_dir))

        # Save chunks to individual files
        for stem, chunks in all_chunks.items():
            chunks_file = converted_dir / f"{stem}_chunks.json"
            chunk_dicts = [chunk.to_dict() for chunk in chunks]

            with open(chunks_file, 'w') as f:
                json.dump(chunk_dicts, f, indent=2)

            logger.info("  Saved %d chunks to %s", len(chunks), chunks_file.name)

        logger.info("  Total: %d documents, %d chunks",
                   len(all_chunks), sum(len(chunks) for chunks in all_chunks.values()))
    except Exception as e:
        logger.error("Chunking failed: %s", str(e))
        logger.debug(traceback.format_exc())

    # ── Step 5 (optional): LLM Enrichment ────────────────────────────
    if enrich:
        logger.info("=" * 60)
        logger.info("[Step 5] LLM Enrichment (backend: %s)", enrich_backend)

        try:
            from llm_enrichment import LLMEnricher, CaseContext

            case_context = CaseContext(
                case_type=case_type,
                parties=[p.strip() for p in parties.split(",") if p.strip()] if parties else [],
            )

            enricher = LLMEnricher(backend=enrich_backend, delay_between_calls=0.1)
            if enricher.is_available():
                enrich_stats = enricher.enrich_directory(str(converted_dir), case_context)
                logger.info("Enrichment complete: %s", enrich_stats.summary())

                # Mark enrichment complete for all successfully processed documents
                for result in results:
                    if result["status"] == "OK":
                        doc_state = state.get_document(result["stem"], result["file"])
                        doc_state.mark_stage_complete("enrichment")
                state.save()
            else:
                logger.warning("Enrichment backend '%s' not available, skipping.", enrich_backend)

        except Exception as e:
            logger.error("Enrichment error: %s", e)
            logger.debug(traceback.format_exc())
            logger.warning("Continuing without enrichment")

    # ── Final state save ──────────────────────────────────────────────
    state.save()
    logger.info("\n" + state.summary())

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    for r in results:
        status = r["status"]
        if status == "FAILED":
            print(f"  FAIL  {r['file']}: {r.get('error', 'Unknown error')}")
        else:
            print(f"  OK    {r['file']}")
            print(f"        Type: {r['doc_type']}, Citations: {r['citations_count']}, "
                  f"Coverage: {r['coverage_pct']:.1f}%")
            if r.get("type_distribution"):
                print(f"        Types: {r['type_distribution']}")
    print("=" * 60)

    # Verify output files
    print("\nOutput files:")
    total_size = 0
    for f in sorted(converted_dir.iterdir()):
        size = f.stat().st_size
        total_size += size
        print(f"  {f.name:50s} {size:>10,} bytes")

    print(f"\nTotal output size: {total_size:,} bytes ({total_size / 1024 / 1024:.1f} MB)")
    if cleanup_json:
        print("(Docling JSON files cleaned up to save disk space)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run pipeline steps 1-3 on PDF documents",
        epilog="""
Examples:
  # Process all documents
  python run_pipeline.py --input-dir tests/test_docs --output-dir output/

  # Resume from previous run
  python run_pipeline.py --input-dir tests/test_docs --output-dir output/ --resume

  # Force reprocess all documents
  python run_pipeline.py --input-dir tests/test_docs --output-dir output/ --force

  # Increase timeout for large documents
  python run_pipeline.py --input-dir docs/ --output-dir output/ --conversion-timeout 600
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--output-dir", required=True, help="Directory for pipeline output")
    parser.add_argument("--use-existing", default=None,
                        help="Path to existing converted files (skip Docling conversion)")
    parser.add_argument("--cleanup-json", action="store_true", default=True,
                        help="Delete Docling JSON files after processing (default: True)")
    parser.add_argument("--no-cleanup-json", dest="cleanup_json", action="store_false",
                        help="Keep Docling JSON files after processing")
    parser.add_argument("--enrich", action="store_true", default=False,
                        help="Run LLM enrichment on chunks after processing")
    parser.add_argument("--enrich-backend", choices=["ollama", "anthropic"], default="ollama",
                        help="LLM backend for enrichment (default: ollama)")
    parser.add_argument("--case-type", default="patent",
                        help="Case type for enrichment context (default: patent)")
    parser.add_argument("--parties", default="",
                        help="Comma-separated party names for enrichment context")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume from previous run (skip already-completed documents)")
    parser.add_argument("--force", action="store_true", default=False,
                        help="Force reprocessing even if stages already complete")
    parser.add_argument("--no-skip-failed", dest="skip_failed", action="store_false", default=True,
                        help="Retry documents that have failed multiple times")
    parser.add_argument("--conversion-timeout", type=int, default=300,
                        help="Timeout in seconds for document conversion (default: 300)")
    parser.add_argument("--parallel", action="store_true", default=False,
                        help="Enable parallel processing of documents")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count - 1)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    use_existing = Path(args.use_existing) if args.use_existing else None

    if not input_dir.exists():
        logger.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    run_pipeline(
        input_dir, output_dir, use_existing,
        cleanup_json=args.cleanup_json,
        enrich=args.enrich,
        enrich_backend=args.enrich_backend,
        case_type=args.case_type,
        parties=args.parties,
        resume=args.resume,
        force=args.force,
        skip_failed=args.skip_failed,
        conversion_timeout=args.conversion_timeout,
        parallel=args.parallel,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()

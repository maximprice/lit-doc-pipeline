#!/usr/bin/env python3
"""
Run pipeline steps 1-3 on PDF documents:
  1. Convert (Docling) → .md + .json
  2. Post-process → cleaned .md + initial _citations.json
  3. Citation tracking → bbox-based _citations.json (overwrites step 2's)

Usage:
    python run_pipeline.py --input-dir tests/test_docs --output-dir tests/pipeline_output
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from citation_tracker import CitationTracker
from citation_types import DocumentType
from docling_converter import DoclingConverter
from post_processor import PostProcessor
from pymupdf_extractor import is_text_based_pdf, extract_deposition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


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


def run_pipeline(input_dir: Path, output_dir: Path, use_existing: Path = None, cleanup_json: bool = True):
    """Run steps 1-3 on all PDFs in input_dir.

    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory for pipeline output
        use_existing: Path to existing converted files (skip Docling)
        cleanup_json: Delete large Docling JSON files after processing (default: True)
    """
    converted_dir = output_dir / "converted"
    converted_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        logger.error("No PDF files found in %s", input_dir)
        sys.exit(1)

    logger.info("Found %d PDF files in %s", len(pdfs), input_dir)
    results = []

    for pdf_path in pdfs:
        stem = pdf_path.stem
        normalized = normalize_stem(stem)
        logger.info("=" * 60)
        logger.info("Processing: %s", pdf_path.name)

        # ── PyMuPDF fast path for text-based depositions ─────────────
        known_type = KNOWN_DOC_TYPES.get(normalized)
        if known_type == DocumentType.DEPOSITION and is_text_based_pdf(str(pdf_path)):
            logger.info("[PyMuPDF] Text-based deposition detected, using direct extraction")
            pymupdf_result = extract_deposition(str(pdf_path), str(converted_dir))
            logger.info(
                "  Extracted %d lines, %d citations",
                pymupdf_result["line_count"],
                pymupdf_result["citation_count"],
            )
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

        # ── Step 1: Conversion ───────────────────────────────────────
        logger.info("[Step 1] Converting with Docling...")

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
            converter = DoclingConverter()
            result = converter.convert_document(str(pdf_path), str(converted_dir))
            md_path = result.md_path
            conversion_citations = result.citations_found
            conversion_errors = result.errors

            if conversion_errors:
                logger.warning("  Conversion errors: %s", conversion_errors)
                if not md_path:
                    logger.error("  FAILED: No output produced, skipping.")
                    results.append({
                        "file": pdf_path.name,
                        "status": "FAILED",
                        "error": conversion_errors[0],
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
            logger.error("  No .md file at %s, skipping.", final_md)
            results.append({
                "file": pdf_path.name,
                "status": "FAILED",
                "error": "No .md file produced",
            })
            continue

        has_json = final_json.exists()
        logger.info("  Output: %s%s",
                     final_md.name,
                     f" + {final_json.name}" if has_json else " (no JSON)")

        # ── Step 2: Post-processing ──────────────────────────────────
        logger.info("[Step 2] Post-processing markdown...")
        doc_type = detect_doc_type(normalized, conversion_citations)
        logger.info("  Document type: %s", doc_type.value)

        processor = PostProcessor()
        proc_result = processor.process(str(final_md), doc_type)
        logger.info("  Post-processor citation coverage: %d elements", proc_result.citation_coverage)

        # ── Step 3: Citation tracking (bbox-based) ───────────────────
        if has_json:
            logger.info("[Step 3] Reconstructing citations from JSON bbox data...")
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

            # Cleanup: Delete large Docling JSON file (keep _citations.json)
            if cleanup_json:
                json_size = final_json.stat().st_size
                final_json.unlink()
                logger.info("  Cleaned up Docling JSON (%d bytes freed)", json_size)
        else:
            logger.warning("[Step 3] Skipped — no JSON file for bbox-based reconstruction")
            citations = {}
            metrics = None

        # Collect results
        results.append({
            "file": pdf_path.name,
            "stem": normalized,
            "status": "OK",
            "doc_type": doc_type.value,
            "md_file": final_md.name,
            "json_file": final_json.name if has_json else None,
            "citations_count": len(citations),
            "coverage_pct": metrics.coverage_pct if metrics else 0.0,
            "type_distribution": metrics.type_distribution if metrics else {},
        })

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
    parser = argparse.ArgumentParser(description="Run pipeline steps 1-3 on PDF documents")
    parser.add_argument("--input-dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--output-dir", required=True, help="Directory for pipeline output")
    parser.add_argument("--use-existing", default=None,
                        help="Path to existing converted files (skip Docling conversion)")
    parser.add_argument("--cleanup-json", action="store_true", default=True,
                        help="Delete Docling JSON files after processing (default: True)")
    parser.add_argument("--no-cleanup-json", dest="cleanup_json", action="store_false",
                        help="Keep Docling JSON files after processing")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    use_existing = Path(args.use_existing) if args.use_existing else None

    if not input_dir.exists():
        logger.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    run_pipeline(input_dir, output_dir, use_existing, cleanup_json=args.cleanup_json)


if __name__ == "__main__":
    main()

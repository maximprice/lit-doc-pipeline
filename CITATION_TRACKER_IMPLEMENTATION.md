# Citation Tracker Implementation Guide

## Overview

This document provides detailed instructions for implementing `citation_tracker.py`, the Phase 1 BLOCKER module that reconstructs line-level citations for depositions and column-level citations for patents. This module must run AFTER post-processing but BEFORE chunking to ensure all downstream components (chunks, context cards) inherit accurate citation metadata.

## Current State Analysis

### What Works (Implemented in lit_doc_pipeline.py)

✅ **Bates number extraction** - Post-processor extracts Bates stamps from headers/footers, creates sidecar files
✅ **Page number tracking** - All cards and chunks have accurate PDF page numbers
✅ **Paragraph number extraction** - Expert reports capture ¶ markers
✅ **Transcript page detection** - "Page N" markers identified in text
✅ **Citation string generation** - `build_citation_string()` exists and formats output correctly

### What's Broken (Not Implemented)

❌ **Deposition line number reconstruction** - `transcript_lines` field is empty on all cards
❌ **Patent column number tracking** - No column detection implemented
❌ **Patent line-in-column tracking** - Not implemented
❌ **Dedicated citation_tracker.py module** - Code is scattered in lit_doc_pipeline.py
❌ **Citations sidecar file** - No `{stem}_citations.json` output file created

### Root Cause: Docling Extraction Defeats Current Algorithm

The current `extract_transcript_lines()` function in `lit_doc_pipeline.py:1051` searches for:
```python
r'^\s*(\d{1,2})\s{2,}'  # Line number at start of line with 2+ spaces
```

This regex **finds 0 matches** because docling merges deposition line numbers into text blobs:

**Expected structure:**
```
1  DANIEL ALEXANDER
2       Do you know what the H
3  processor line designation means?
```

**Actual docling output:**
```
"1 DANIEL ALEXANDER 2 Do you know what the H 3 processor line designation means?"
```

## Implementation Architecture

### Module Location and Entry Point

**File:** `/Users/maximprice/Dev/lit-doc-pipeline/citation_tracker.py` (create new)

**Entry point in pipeline:**
```python
# In lit_doc_pipeline.py, after post-processing, before chunking:
if not skip_citation_tracking:
    from citation_tracker import CitationTracker
    tracker = CitationTracker(
        converted_dir=converted_dir,
        doc_type=detected_doc_type  # deposition|patent|expert_report|pleading
    )
    citations_data = tracker.reconstruct_citations(stem)
    # Save to {stem}_citations.json
    citations_file = converted_dir / f"{stem}_citations.json"
    with open(citations_file, 'w') as f:
        json.dump(citations_data, f, indent=2)
```

### Data Structures

#### Input: Docling JSON Structure
```python
{
  "texts": [
    {
      "text": "1 DANIEL ALEXANDER 2 Do you know...",
      "prov": [{
        "page_no": 47,
        "bbox": {
          "l": 65.4,        # Left edge (pixels from left)
          "t": 679.013,     # Top edge (pixels from BOTTOM in BOTTOMLEFT coord system)
          "r": 430.95,      # Right edge
          "b": 664.284,     # Bottom edge
          "coord_origin": "BOTTOMLEFT"
        }
      }]
    }
  ]
}
```

#### Output: Citations Sidecar File
```python
# {stem}_citations.json
{
  "#/texts/42": {
    "page": 14,
    "line_start": 5,
    "line_end": 12,
    "bates": "INTEL_PROX_00001784",
    "type": "transcript_line"
  },
  "#/texts/543": {
    "page": 89,
    "column": 3,
    "line_start": 45,
    "line_end": 55,
    "type": "patent_column"
  }
}
```

## Implementation Strategy

### Strategy 1: Embedded Line Number Parsing (RECOMMENDED FOR DEPOSITIONS)

Since docling merges line numbers into text, parse them from the embedded content.

#### Algorithm: Split-on-Number Pattern

```python
def extract_embedded_line_numbers(text: str, page_no: int) -> list[tuple[int, str]]:
    """
    Parse line numbers embedded in merged text blobs.

    Handles patterns like:
    "1 DANIEL ALEXANDER 2 Do you know what the H 3 processor line"

    Returns: [(line_num, text_fragment), ...]
    """
    # Pattern: number 1-25 followed by at least one space, not mid-word
    pattern = r'(?:^|\s)(\d{1,2})(?=\s+[A-Z]|\s+[a-z]|\s*$)'

    matches = []
    last_end = 0

    for m in re.finditer(pattern, text):
        num = int(m.group(1))
        if 1 <= num <= 30:  # Allow up to 30 for some formats
            start = m.end()
            # Extract text until next line number or end
            next_match = re.search(pattern, text[start:])
            if next_match:
                fragment = text[start:start + next_match.start()].strip()
            else:
                fragment = text[start:].strip()

            matches.append((num, fragment))

    return matches
```

#### Line-to-Page Association

Use the "Page N" markers in the markdown to assign line numbers to transcript pages:

```python
def associate_lines_with_pages(texts: list[dict], page_markers: dict[int, int]) -> dict:
    """
    Map each text item to its transcript page based on Page markers.

    Args:
        texts: Docling JSON texts array
        page_markers: {text_item_index: transcript_page_number}

    Returns:
        {text_item_index: {
            "pdf_page": int,
            "transcript_page": int,
            "line_numbers": [int, ...],
            "line_start": int,
            "line_end": int
        }}
    """
    result = {}
    current_transcript_page = None

    for i, item in enumerate(texts):
        # Check if this item is a page marker
        text = item.get('text', '').strip()
        if text.startswith('Page ') and text[5:].strip().isdigit():
            current_transcript_page = int(text[5:].strip())
            continue

        # Extract line numbers from this text item
        line_nums = extract_embedded_line_numbers(text, current_transcript_page)

        if line_nums and current_transcript_page:
            nums_only = [num for num, _ in line_nums]
            prov = item.get('prov', [])
            pdf_page = prov[0].get('page_no', 0) if prov else 0

            result[f"#/texts/{i}"] = {
                "pdf_page": pdf_page,
                "transcript_page": current_transcript_page,
                "line_numbers": nums_only,
                "line_start": min(nums_only),
                "line_end": max(nums_only),
                "type": "transcript_line"
            }

    return result
```

### Strategy 2: Bbox-Based Line Inference (FALLBACK)

When line numbers are not extractable from text, infer lines from vertical position.

#### Algorithm: Vertical Band Assignment

```python
def infer_lines_from_bbox(texts: list[dict], lines_per_page: int = 25) -> dict:
    """
    Infer line numbers from bbox vertical positions.

    Assumes uniform line spacing on transcript pages.
    """
    # Group texts by page
    by_page = defaultdict(list)
    for i, item in enumerate(texts):
        prov = item.get('prov', [])
        if prov:
            page = prov[0].get('page_no', 0)
            bbox = prov[0].get('bbox', {})
            by_page[page].append((i, item, bbox))

    result = {}
    for page, items in by_page.items():
        # Get vertical range of text on this page
        tops = [bbox.get('t', 0) for _, _, bbox in items if bbox]
        bottoms = [bbox.get('b', 0) for _, _, bbox in items if bbox]

        if not tops:
            continue

        # Note: BOTTOMLEFT origin means t > b (top is higher value)
        min_y = min(bottoms)
        max_y = max(tops)
        page_height = max_y - min_y
        line_height = page_height / lines_per_page

        for idx, item, bbox in items:
            if bbox:
                # Calculate which line band this text falls in
                y_center = (bbox.get('t', 0) + bbox.get('b', 0)) / 2
                line_num = int((max_y - y_center) / line_height) + 1
                line_num = max(1, min(lines_per_page, line_num))

                result[f"#/texts/{idx}"] = {
                    "pdf_page": page,
                    "line_start": line_num,
                    "line_end": line_num,
                    "type": "inferred_line"
                }

    return result
```

### Strategy 3: Patent Column Detection

For patents, detect column layout using bbox horizontal positions.

#### Algorithm: Column Boundary Detection

```python
def detect_patent_columns(texts: list[dict]) -> dict:
    """
    Detect column numbers for patent specifications.

    Typical patent layout:
    - Single column: bbox.l < 300, bbox.r > 400 (full width)
    - Left column: bbox.l < 300, bbox.r < 400
    - Right column: bbox.l > 300
    """
    result = {}

    for i, item in enumerate(texts):
        prov = item.get('prov', [])
        if not prov:
            continue

        bbox = prov[0].get('bbox', {})
        l = bbox.get('l', 0)
        r = bbox.get('r', 0)
        page = prov[0].get('page_no', 0)

        # Detect column based on horizontal position
        if r - l > 400:  # Full-width (single column or title)
            column = None
        elif l < 300:
            column = 1  # Left column
        else:
            column = 2  # Right column

        if column:
            # Look for column:line format in text (e.g., "col. 3, lines 45-55")
            text = item.get('text', '')
            col_match = re.search(r'col(?:umn)?\.\s*(\d+)', text, re.IGNORECASE)
            line_match = re.search(r'lines?\s*(\d+)(?:\s*-\s*(\d+))?', text, re.IGNORECASE)

            if col_match:
                column = int(col_match.group(1))

            line_start = None
            line_end = None
            if line_match:
                line_start = int(line_match.group(1))
                line_end = int(line_match.group(2)) if line_match.group(2) else line_start

            result[f"#/texts/{i}"] = {
                "page": page,
                "column": column,
                "line_start": line_start,
                "line_end": line_end,
                "type": "patent_column"
            }

    return result
```

## Implementation Steps

### Step 1: Create Module Structure

```python
# citation_tracker.py

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import re
from collections import defaultdict

@dataclass
class DocumentType:
    """Document type enumeration."""
    DEPOSITION = "deposition"
    PATENT = "patent"
    EXPERT_REPORT = "expert_report"
    PLEADING = "pleading"
    GENERAL = "general"

class CitationTracker:
    """
    Reconstructs citation metadata from docling JSON output.

    Handles:
    - Deposition line numbers (page:line format)
    - Patent column numbers (col:line format)
    - Expert report paragraph numbers
    - Bates number association
    """

    def __init__(self, converted_dir: Path, doc_type: str):
        self.converted_dir = Path(converted_dir)
        self.doc_type = doc_type

    def reconstruct_citations(self, stem: str) -> dict:
        """
        Main entry point: reconstruct citations for a document.

        Returns citations dict suitable for saving as {stem}_citations.json
        """
        json_path = self.converted_dir / f"{stem}.json"
        md_path = self.converted_dir / f"{stem}.md"
        bates_path = self.converted_dir / f"{stem}_bates.json"

        # Load inputs
        with open(json_path) as f:
            docling_data = json.load(f)

        bates_by_page = {}
        if bates_path.exists():
            with open(bates_path) as f:
                bates_by_page = json.load(f)

        # Route to appropriate handler
        if self.doc_type == DocumentType.DEPOSITION:
            return self._reconstruct_deposition(docling_data, bates_by_page)
        elif self.doc_type == DocumentType.PATENT:
            return self._reconstruct_patent(docling_data, bates_by_page)
        elif self.doc_type == DocumentType.EXPERT_REPORT:
            return self._reconstruct_expert_report(docling_data, bates_by_page)
        else:
            return self._reconstruct_generic(docling_data, bates_by_page)

    def _reconstruct_deposition(self, docling_data: dict, bates_by_page: dict) -> dict:
        """Reconstruct deposition citations with line numbers."""
        # Implementation: Strategy 1 + Strategy 2 fallback
        pass

    def _reconstruct_patent(self, docling_data: dict, bates_by_page: dict) -> dict:
        """Reconstruct patent citations with column numbers."""
        # Implementation: Strategy 3
        pass

    def _reconstruct_expert_report(self, docling_data: dict, bates_by_page: dict) -> dict:
        """Reconstruct expert report citations with paragraph numbers."""
        # Already implemented in extract_paragraph_numbers(), adapt here
        pass

    def _reconstruct_generic(self, docling_data: dict, bates_by_page: dict) -> dict:
        """Generic citation tracking (page + Bates only)."""
        pass
```

### Step 2: Implement Deposition Handler

```python
def _reconstruct_deposition(self, docling_data: dict, bates_by_page: dict) -> dict:
    """
    Reconstruct deposition line-level citations.

    Uses Strategy 1 (embedded line parsing) with Strategy 2 (bbox) as fallback.
    """
    texts = docling_data.get('texts', [])
    citations = {}

    # Step 1: Find all "Page N" markers
    page_markers = {}  # {text_index: transcript_page_number}
    for i, item in enumerate(texts):
        if isinstance(item, dict):
            text = item.get('text', '').strip()
            if text.startswith('Page ') and text[5:].strip().isdigit():
                page_markers[i] = int(text[5:].strip())

    # Step 2: Track current transcript page and extract line numbers
    current_transcript_page = None

    for i, item in enumerate(texts):
        if not isinstance(item, dict):
            continue

        # Update current transcript page if this is a marker
        if i in page_markers:
            current_transcript_page = page_markers[i]
            continue

        # Get PDF page from provenance
        prov = item.get('prov', [])
        if not prov:
            continue
        pdf_page = prov[0].get('page_no', 0)

        # Extract line numbers from text
        text = item.get('text', '')
        line_nums = self._extract_embedded_line_numbers(text)

        if line_nums and current_transcript_page:
            citations[f"#/texts/{i}"] = {
                "pdf_page": pdf_page,
                "transcript_page": current_transcript_page,
                "line_start": min(line_nums),
                "line_end": max(line_nums),
                "line_numbers": line_nums,
                "bates": bates_by_page.get(str(pdf_page)),
                "type": "transcript_line"
            }
        elif current_transcript_page:
            # Fallback: infer line from bbox if no embedded numbers
            bbox = prov[0].get('bbox', {})
            if bbox:
                line_num = self._infer_line_from_bbox(bbox, pdf_page, texts)
                if line_num:
                    citations[f"#/texts/{i}"] = {
                        "pdf_page": pdf_page,
                        "transcript_page": current_transcript_page,
                        "line_start": line_num,
                        "line_end": line_num,
                        "bates": bates_by_page.get(str(pdf_page)),
                        "type": "inferred_line"
                    }

    return citations

def _extract_embedded_line_numbers(self, text: str) -> list[int]:
    """
    Extract line numbers (1-30) embedded in text.

    Handles: "1 DANIEL ALEXANDER 2 Do you know what..."
    """
    # Pattern: standalone 1-2 digit number at word boundaries
    pattern = r'(?:^|\s)(\d{1,2})(?=\s+[A-Z\']|\s+[a-z]|\s*$)'

    line_nums = []
    for m in re.finditer(pattern, text):
        num = int(m.group(1))
        if 1 <= num <= 30:
            line_nums.append(num)

    return sorted(set(line_nums))

def _infer_line_from_bbox(self, bbox: dict, page: int, all_texts: list) -> Optional[int]:
    """Fallback: infer line number from vertical position."""
    # Implementation: Strategy 2
    pass
```

### Step 3: Implement Patent Handler

```python
def _reconstruct_patent(self, docling_data: dict, bates_by_page: dict) -> dict:
    """
    Reconstruct patent column:line citations.

    Uses Strategy 3 (bbox-based column detection).
    """
    texts = docling_data.get('texts', [])
    citations = {}

    for i, item in enumerate(texts):
        if not isinstance(item, dict):
            continue

        prov = item.get('prov', [])
        if not prov:
            continue

        bbox = prov[0].get('bbox', {})
        page = prov[0].get('page_no', 0)
        text = item.get('text', '')

        # Detect column from bbox horizontal position
        l = bbox.get('l', 0)
        r = bbox.get('r', 0)
        width = r - l

        # Heuristic: full-width vs columnar
        if width > 400:
            column = None  # Full-width text (title, abstract, etc.)
        elif l < 300:
            column = 1  # Left column
        else:
            column = 2  # Right column

        # Look for explicit "col. X, lines Y-Z" references in text
        col_match = re.search(r'col(?:umn)?\.\s*(\d+)', text, re.IGNORECASE)
        line_match = re.search(r'lines?\s*(\d+)(?:\s*[-–]\s*(\d+))?', text, re.IGNORECASE)

        if col_match:
            column = int(col_match.group(1))

        line_start = None
        line_end = None
        if line_match:
            line_start = int(line_match.group(1))
            line_end = int(line_match.group(2)) if line_match.group(2) else line_start

        if column or line_start:
            citations[f"#/texts/{i}"] = {
                "page": page,
                "column": column,
                "line_start": line_start,
                "line_end": line_end,
                "bates": bates_by_page.get(str(page)),
                "type": "patent_column"
            }

    return citations
```

### Step 4: Integration with Pipeline

Modify `lit_doc_pipeline.py` to call citation tracker:

```python
# In main pipeline flow, after post-processing:

def run_citation_tracking(converted_dir: Path, stem: str, doc_type: str) -> Optional[Path]:
    """Run citation tracker and save results."""
    try:
        from citation_tracker import CitationTracker

        tracker = CitationTracker(converted_dir=converted_dir, doc_type=doc_type)
        citations = tracker.reconstruct_citations(stem)

        citations_file = converted_dir / f"{stem}_citations.json"
        with open(citations_file, 'w') as f:
            json.dump(citations, f, indent=2)

        print(f"  Citation tracking: {len(citations)} text items mapped")
        return citations_file
    except Exception as e:
        print(f"  Warning: Citation tracking failed: {e}")
        return None

# Call after docling conversion and post-processing:
citations_file = run_citation_tracking(
    converted_dir=output_dir / "converted",
    stem=stem,
    doc_type=detected_doc_type
)
```

### Step 5: Update Chunking to Use Citations

Modify `chunk_sections_with_overlap()` to load and use citations:

```python
def load_citations_for_chunk(chunk_text_indices: list[int], citations_data: dict) -> dict:
    """
    Load citation metadata for text items in a chunk.

    Returns aggregated citation metadata.
    """
    transcript_pages = set()
    transcript_lines = defaultdict(lambda: [999, 0])  # [min, max]
    pdf_pages = set()
    bates = set()

    for idx in chunk_text_indices:
        key = f"#/texts/{idx}"
        if key in citations_data:
            cit = citations_data[key]

            if 'transcript_page' in cit:
                transcript_pages.add(cit['transcript_page'])
                page_key = str(cit['transcript_page'])
                transcript_lines[page_key][0] = min(transcript_lines[page_key][0], cit['line_start'])
                transcript_lines[page_key][1] = max(transcript_lines[page_key][1], cit['line_end'])

            if 'pdf_page' in cit:
                pdf_pages.add(cit['pdf_page'])

            if cit.get('bates'):
                bates.add(cit['bates'])

    return {
        "pdf_pages": sorted(pdf_pages),
        "transcript_pages": sorted(transcript_pages),
        "transcript_lines": {k: v for k, v in transcript_lines.items()},
        "bates_range": sorted(bates)
    }
```

## Testing Strategy

### Unit Tests

Create `tests/test_citation_tracker.py`:

```python
import pytest
from citation_tracker import CitationTracker

def test_embedded_line_extraction():
    """Test parsing line numbers from merged text."""
    tracker = CitationTracker(converted_dir=".", doc_type="deposition")

    text = "1 DANIEL ALEXANDER 2 Do you know what 3 the processor means?"
    line_nums = tracker._extract_embedded_line_numbers(text)

    assert line_nums == [1, 2, 3]

def test_page_association():
    """Test associating line numbers with transcript pages."""
    # Mock docling JSON with Page markers
    texts = [
        {"text": "Page 45"},
        {"text": "1 Q. What happened? 2 A. I don't know.", "prov": [{"page_no": 45}]},
        {"text": "Page 46"},
        {"text": "1 Q. And then? 2 A. Nothing.", "prov": [{"page_no": 46}]},
    ]

    tracker = CitationTracker(converted_dir=".", doc_type="deposition")
    result = tracker._associate_lines_with_pages(texts, {})

    assert "#/texts/1" in result
    assert result["#/texts/1"]["transcript_page"] == 45
    assert result["#/texts/1"]["line_start"] == 1
    assert result["#/texts/1"]["line_end"] == 2

def test_patent_column_detection():
    """Test patent column detection from bbox."""
    tracker = CitationTracker(converted_dir=".", doc_type="patent")

    # Left column bbox
    bbox_left = {"l": 100, "r": 280, "t": 500, "b": 480}
    assert tracker._detect_column(bbox_left) == 1

    # Right column bbox
    bbox_right = {"l": 320, "r": 500, "t": 500, "b": 480}
    assert tracker._detect_column(bbox_right) == 2
```

### Integration Tests

Create sample deposition JSON and verify end-to-end:

```python
def test_alexander_deposition_integration():
    """Test on actual Alexander deposition output."""
    tracker = CitationTracker(
        converted_dir="/path/to/processed_20260209_232538/converted",
        doc_type="deposition"
    )

    citations = tracker.reconstruct_citations("daniel_alexander_10_24_2025")

    # Verify we got line number data
    assert len(citations) > 0

    # Check a known card has proper citation
    sample = citations.get("#/texts/550")
    if sample:
        assert "line_start" in sample
        assert "line_end" in sample
        assert sample["type"] in ["transcript_line", "inferred_line"]
```

## Validation and Quality Checks

### Post-Processing Validation

After citation tracking runs, validate output:

```python
def validate_citations(citations_file: Path, doc_type: str) -> dict:
    """
    Validate citation tracking quality.

    Returns metrics dict.
    """
    with open(citations_file) as f:
        citations = json.load(f)

    metrics = {
        "total_items": len(citations),
        "has_line_numbers": 0,
        "has_column_numbers": 0,
        "has_bates": 0,
        "missing_data": []
    }

    for key, cit in citations.items():
        if cit.get("line_start"):
            metrics["has_line_numbers"] += 1
        if cit.get("column"):
            metrics["has_column_numbers"] += 1
        if cit.get("bates"):
            metrics["has_bates"] += 1

        # Check for expected fields based on type
        if doc_type == "deposition":
            if not cit.get("transcript_page"):
                metrics["missing_data"].append(f"{key}: missing transcript_page")
            if not cit.get("line_start"):
                metrics["missing_data"].append(f"{key}: missing line_start")

    return metrics
```

### Citation String Verification

Test that generated citation strings are correct:

```python
def test_citation_string_generation():
    """Verify citation strings match legal format."""
    # Mock chunk with citation metadata
    chunk = {
        "transcript_pages": [45, 46],
        "transcript_lines": {
            "45": [12, 25],
            "46": [1, 8]
        }
    }

    citation_string = build_citation_string(
        source="Alexander Dep.",
        citation=chunk
    )

    assert citation_string == "Alexander Dep. 45:12-46:8"
```

## Debugging and Troubleshooting

### Common Issues

**Issue 1: Line numbers not extracted**
- **Symptom:** `transcript_lines` still empty after running tracker
- **Debug:** Check if line numbers appear in markdown text (grep for `^\s*\d{1,2}\s`)
- **Fix:** Adjust regex pattern in `_extract_embedded_line_numbers()` to match actual format

**Issue 2: Wrong transcript pages assigned**
- **Symptom:** Line numbers mapped to wrong pages
- **Debug:** Verify "Page N" markers are detected correctly
- **Fix:** Check page marker regex and ensure page state tracking works

**Issue 3: Bbox inference produces wrong line numbers**
- **Symptom:** Line numbers don't match transcript
- **Debug:** Check bbox coordinate system (BOTTOMLEFT vs TOPLEFT)
- **Fix:** Adjust vertical position calculations

### Debug Output

Add verbose logging:

```python
def _reconstruct_deposition(self, docling_data: dict, bates_by_page: dict) -> dict:
    """..."""
    DEBUG = os.getenv("CITATION_TRACKER_DEBUG") == "1"

    if DEBUG:
        print(f"[DEBUG] Processing {len(texts)} text items")
        print(f"[DEBUG] Found {len(page_markers)} page markers")

    for i, item in enumerate(texts):
        # ...
        if DEBUG and line_nums:
            print(f"[DEBUG] Item {i}: page={current_transcript_page}, lines={line_nums}")
```

Run with:
```bash
CITATION_TRACKER_DEBUG=1 doc-pipeline --input-dir test_docs --output-dir output
```

## Performance Considerations

- Citation tracking adds ~5-10% to total pipeline time
- Regex matching on large text blobs can be slow; consider compiling patterns
- Bbox calculations are fast; use as primary strategy for patents
- Cache page marker detection to avoid re-scanning

## Future Enhancements

1. **Machine learning line detection** - Train model to recognize line boundaries from PDF images
2. **OCR fallback** - If docling fails, use pytesseract to extract line-by-line
3. **Cross-reference validation** - Verify extracted line numbers match Q&A structure
4. **Multi-column patent support** - Handle 3+ column layouts
5. **Exhibit markers** - Track "Exhibit A" references across documents

## Success Metrics

Citation tracker is successful when:

✅ ≥95% of deposition cards have non-empty `transcript_lines`
✅ ≥90% of patent cards have column numbers
✅ Citation strings match legal citation format
✅ Line number ranges are continuous (no gaps like 5-7, missing 6)
✅ Page:line cites can be used to locate exact text in source PDFs

## References

- CLAUDE.md Phase 1 specifications (lines 53-76)
- TRD Section 4: Citation tracking algorithms
- Docling JSON schema documentation
- Legal citation formats: Bluebook Rule 10.8 (depositions), Rule 14 (patents)

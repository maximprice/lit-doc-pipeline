"""
Regression tests for pipeline quality assurance.

These tests validate critical properties of the pipeline output:
1. No garbage text from images (base64, OCR artifacts)
2. Page numbers preserved correctly
3. Line numbers accurate for text-based depositions
4. Footnotes included in relevant chunks
5. Citation coverage meets minimum thresholds
6. Chunk structure integrity

Tests run against pre-generated pipeline output in tests/pipeline_output/.
"""

import json
import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Test data paths
# ---------------------------------------------------------------------------

PIPELINE_OUTPUT = Path("tests/pipeline_output/converted")
WEDNESDAY_OUTPUT = Path("tests/wednesday_morning_test_2026-02-11/converted")

ALEXANDER_CHUNKS = PIPELINE_OUTPUT / "daniel_alexander_10_24_2025_chunks.json"
ALEXANDER_CITATIONS = PIPELINE_OUTPUT / "daniel_alexander_10_24_2025_citations.json"
ALEXANDER_MD = PIPELINE_OUTPUT / "daniel_alexander_10_24_2025.md"

COLE_CHUNKS = PIPELINE_OUTPUT / "2025_12_11_cole_report_chunks.json"
COLE_CITATIONS = PIPELINE_OUTPUT / "2025_12_11_cole_report_citations.json"
COLE_MD = PIPELINE_OUTPUT / "2025_12_11_cole_report.md"

PATENT_CITATIONS = PIPELINE_OUTPUT / "intel_prox_00006214_citations.json"
PATENT_MD = PIPELINE_OUTPUT / "intel_prox_00006214.md"


def _require(path: Path):
    if not path.exists():
        pytest.skip(f"Test data not available: {path}")


# ---------------------------------------------------------------------------
# 1. No garbage text from images
# ---------------------------------------------------------------------------

class TestNoImageGarbage:
    """Verify no base64 image data or OCR garbage in output."""

    GARBAGE_PATTERNS = [
        (r"data:image/[^;]+;base64", "base64 image data"),
        (r"<picture>", "HTML picture element"),
        (r"ImageElement|ImageRef", "Docling image reference"),
        (r"iVBOR|/9j/|R0lGOD", "raw base64 image bytes"),
    ]

    @pytest.mark.parametrize("md_file", [
        ALEXANDER_MD,
        COLE_MD,
        PATENT_MD,
    ])
    def test_no_base64_in_markdown(self, md_file):
        """Markdown files must not contain base64 image data."""
        _require(md_file)
        content = md_file.read_text()

        for pattern, description in self.GARBAGE_PATTERNS:
            matches = re.findall(pattern, content)
            assert len(matches) == 0, (
                f"Found {description} in {md_file.name}: {len(matches)} occurrences"
            )

    def test_image_placeholders_are_clean(self):
        """Image placeholders should be simple comments, not embedded data."""
        _require(COLE_MD)
        content = COLE_MD.read_text()

        for match in re.finditer(r"<!-- image[^>]*-->", content):
            placeholder = match.group()
            # Should be short (just a comment, no data)
            assert len(placeholder) < 100, (
                f"Image placeholder too long ({len(placeholder)} chars), may contain data"
            )

    @pytest.mark.parametrize("chunks_file", [
        ALEXANDER_CHUNKS,
        COLE_CHUNKS,
    ])
    def test_no_base64_in_chunks(self, chunks_file):
        """Chunk text must not contain base64 image data."""
        _require(chunks_file)
        with open(chunks_file) as f:
            chunks = json.load(f)

        for chunk in chunks:
            text = chunk["core_text"]
            for pattern, description in self.GARBAGE_PATTERNS:
                assert not re.search(pattern, text), (
                    f"Found {description} in chunk {chunk['chunk_id']}"
                )


# ---------------------------------------------------------------------------
# 2. Page numbers preserved correctly
# ---------------------------------------------------------------------------

class TestPageNumberPreservation:
    """Verify page numbers are correctly tracked in citations and chunks."""

    def test_deposition_citations_have_pages(self):
        """Every Alexander deposition citation must have a page number."""
        _require(ALEXANDER_CITATIONS)
        with open(ALEXANDER_CITATIONS) as f:
            citations = json.load(f)

        no_page = [ref for ref, c in citations.items() if c.get("page") is None]
        assert len(no_page) == 0, (
            f"{len(no_page)} citations missing page number (of {len(citations)} total)"
        )

    def test_expert_report_citations_have_pages(self):
        """Every Cole Report citation must have a page number."""
        _require(COLE_CITATIONS)
        with open(COLE_CITATIONS) as f:
            citations = json.load(f)

        no_page = [ref for ref, c in citations.items() if c.get("page") is None]
        assert len(no_page) == 0, (
            f"{len(no_page)} citations missing page number (of {len(citations)} total)"
        )

    def test_patent_citations_have_pages(self):
        """Every patent citation must have a page number."""
        _require(PATENT_CITATIONS)
        with open(PATENT_CITATIONS) as f:
            citations = json.load(f)

        no_page = [ref for ref, c in citations.items() if c.get("page") is None]
        assert len(no_page) == 0, (
            f"{len(no_page)} citations missing page number (of {len(citations)} total)"
        )

    def test_deposition_chunks_have_pages(self):
        """Every deposition chunk must list page numbers."""
        _require(ALEXANDER_CHUNKS)
        with open(ALEXANDER_CHUNKS) as f:
            chunks = json.load(f)

        for chunk in chunks:
            assert len(chunk["pages"]) > 0, (
                f"Chunk {chunk['chunk_id']} has no page numbers"
            )

    def test_deposition_page_range_reasonable(self):
        """Deposition pages should be within document range (1-165)."""
        _require(ALEXANDER_CITATIONS)
        with open(ALEXANDER_CITATIONS) as f:
            citations = json.load(f)

        pages = [c["page"] for c in citations.values() if c.get("page") is not None]
        assert min(pages) >= 1, f"Page number below 1: {min(pages)}"
        assert max(pages) <= 200, f"Page number above 200: {max(pages)}"

    def test_patent_page_range_reasonable(self):
        """Patent pages should be within document range."""
        _require(PATENT_CITATIONS)
        with open(PATENT_CITATIONS) as f:
            citations = json.load(f)

        pages = [c["page"] for c in citations.values() if c.get("page") is not None]
        assert min(pages) >= 1, f"Page number below 1: {min(pages)}"
        assert max(pages) <= 100, f"Page number unexpectedly high: {max(pages)}"


# ---------------------------------------------------------------------------
# 3. Line numbers accurate for depositions
# ---------------------------------------------------------------------------

class TestDepositionLineAccuracy:
    """Verify line number extraction for text-based depositions."""

    def test_100_percent_line_coverage(self):
        """Alexander deposition must have 100% line-level citation coverage."""
        _require(ALEXANDER_CITATIONS)
        with open(ALEXANDER_CITATIONS) as f:
            citations = json.load(f)

        total = len(citations)
        with_lines = sum(
            1 for c in citations.values() if c.get("line_start") is not None
        )

        coverage = with_lines / total * 100
        assert coverage == 100.0, (
            f"Expected 100% line coverage, got {coverage:.1f}% "
            f"({with_lines}/{total})"
        )

    def test_line_numbers_within_range(self):
        """Line numbers must be 1-25 (standard deposition page)."""
        _require(ALEXANDER_CITATIONS)
        with open(ALEXANDER_CITATIONS) as f:
            citations = json.load(f)

        for ref, cit in citations.items():
            ls = cit.get("line_start")
            le = cit.get("line_end")
            if ls is not None:
                assert 1 <= ls <= 25, (
                    f"line_start={ls} out of range in {ref}"
                )
            if le is not None:
                assert 1 <= le <= 25, (
                    f"line_end={le} out of range in {ref}"
                )
            if ls is not None and le is not None:
                assert ls <= le, (
                    f"line_start={ls} > line_end={le} in {ref}"
                )

    def test_deposition_chunks_have_transcript_lines(self):
        """Every deposition chunk must have transcript_lines in citation."""
        _require(ALEXANDER_CHUNKS)
        with open(ALEXANDER_CHUNKS) as f:
            chunks = json.load(f)

        for chunk in chunks:
            citation = chunk.get("citation", {})
            assert citation.get("transcript_lines"), (
                f"Chunk {chunk['chunk_id']} missing transcript_lines in citation"
            )

    def test_deposition_citation_string_format(self):
        """Deposition citation strings should follow page:line format."""
        _require(ALEXANDER_CHUNKS)
        with open(ALEXANDER_CHUNKS) as f:
            chunks = json.load(f)

        for chunk in chunks:
            cs = chunk.get("citation_string", "")
            assert len(cs) > 0, (
                f"Chunk {chunk['chunk_id']} has empty citation_string"
            )
            # Should contain "Dep." for deposition format
            assert "Dep." in cs, (
                f"Chunk {chunk['chunk_id']} citation_string missing 'Dep.': {cs}"
            )


# ---------------------------------------------------------------------------
# 4. Footnotes included in relevant chunks
# ---------------------------------------------------------------------------

class TestFootnoteInclusion:
    """Verify footnotes are inline in expert report chunks."""

    def test_cole_report_has_footnotes_in_chunks(self):
        """Cole Report chunks should contain inline footnotes."""
        _require(COLE_CHUNKS)
        with open(COLE_CHUNKS) as f:
            chunks = json.load(f)

        chunks_with_fn = [
            c for c in chunks if "[FOOTNOTE" in c["core_text"]
        ]

        # Cole Report has 12+ footnotes, some chunks should contain them
        assert len(chunks_with_fn) >= 3, (
            f"Expected >= 3 chunks with footnotes, got {len(chunks_with_fn)}"
        )

    def test_footnote_format_correct(self):
        """Footnotes should use [FOOTNOTE N: ...] format."""
        _require(COLE_CHUNKS)
        with open(COLE_CHUNKS) as f:
            chunks = json.load(f)

        all_text = " ".join(c["core_text"] for c in chunks)
        footnotes = re.findall(r"\[FOOTNOTE (\d+):", all_text)

        assert len(footnotes) >= 5, (
            f"Expected >= 5 inline footnotes, found {len(footnotes)}"
        )

        # Footnote numbers should be sequential/reasonable
        fn_numbers = sorted(set(int(n) for n in footnotes))
        assert fn_numbers[0] >= 1, "Footnote numbering should start at 1"
        assert fn_numbers[-1] <= 20, (
            f"Footnote number {fn_numbers[-1]} seems too high"
        )

    def test_footnote_content_not_empty(self):
        """Inline footnotes should contain actual text (legal abbreviations like 'Id.' are OK)."""
        _require(COLE_MD)
        content = COLE_MD.read_text()

        for match in re.finditer(r"\[FOOTNOTE \d+: (.*?)\]", content, re.DOTALL):
            fn_content = match.group(1).strip()
            # Legal abbreviations like "Id." and "Id. at 5" are valid short footnotes
            assert len(fn_content) > 2, (
                f"Footnote content too short: {fn_content[:50]}"
            )

    def test_deposition_no_footnotes(self):
        """Depositions should not have inline footnotes (they don't have footnotes)."""
        _require(ALEXANDER_CHUNKS)
        with open(ALEXANDER_CHUNKS) as f:
            chunks = json.load(f)

        for chunk in chunks:
            assert "[FOOTNOTE" not in chunk["core_text"], (
                f"Unexpected footnote in deposition chunk {chunk['chunk_id']}"
            )


# ---------------------------------------------------------------------------
# 5. Citation coverage thresholds
# ---------------------------------------------------------------------------

class TestCitationCoverage:
    """Verify citation extraction meets quality thresholds."""

    def test_deposition_minimum_citations(self):
        """Alexander deposition should produce >= 3000 citations."""
        _require(ALEXANDER_CITATIONS)
        with open(ALEXANDER_CITATIONS) as f:
            citations = json.load(f)

        assert len(citations) >= 3000, (
            f"Expected >= 3000 citations for Alexander deposition, got {len(citations)}"
        )

    def test_expert_report_minimum_citations(self):
        """Cole Report should produce >= 500 citations."""
        _require(COLE_CITATIONS)
        with open(COLE_CITATIONS) as f:
            citations = json.load(f)

        assert len(citations) >= 500, (
            f"Expected >= 500 citations for Cole Report, got {len(citations)}"
        )

    def test_patent_minimum_citations(self):
        """Patent should produce >= 1000 citations."""
        _require(PATENT_CITATIONS)
        with open(PATENT_CITATIONS) as f:
            citations = json.load(f)

        assert len(citations) >= 1000, (
            f"Expected >= 1000 citations for patent, got {len(citations)}"
        )

    def test_patent_column_detection_on_spec_pages(self):
        """Patent spec pages (24-34) should have >= 80% column detection."""
        _require(PATENT_CITATIONS)
        with open(PATENT_CITATIONS) as f:
            citations = json.load(f)

        spec_pages = set(range(24, 35))
        spec_cits = [
            c for c in citations.values()
            if c.get("page") in spec_pages
        ]
        col_cits = [c for c in spec_cits if c.get("column") is not None]

        coverage = len(col_cits) / len(spec_cits) * 100
        assert coverage >= 80.0, (
            f"Expected >= 80% column detection on spec pages, got {coverage:.1f}% "
            f"({len(col_cits)}/{len(spec_cits)})"
        )


# ---------------------------------------------------------------------------
# 6. Chunk structure integrity
# ---------------------------------------------------------------------------

class TestChunkIntegrity:
    """Verify chunk structure and metadata completeness."""

    @pytest.mark.parametrize("chunks_file,expected_min", [
        (ALEXANDER_CHUNKS, 30),
        (COLE_CHUNKS, 10),
    ])
    def test_minimum_chunk_count(self, chunks_file, expected_min):
        """Document should produce a minimum number of chunks."""
        _require(chunks_file)
        with open(chunks_file) as f:
            chunks = json.load(f)

        assert len(chunks) >= expected_min, (
            f"Expected >= {expected_min} chunks, got {len(chunks)}"
        )

    @pytest.mark.parametrize("chunks_file", [ALEXANDER_CHUNKS, COLE_CHUNKS])
    def test_all_chunks_have_required_fields(self, chunks_file):
        """Every chunk must have all required fields."""
        _require(chunks_file)
        with open(chunks_file) as f:
            chunks = json.load(f)

        required = ["chunk_id", "core_text", "pages", "citation", "citation_string"]
        for chunk in chunks:
            for field in required:
                assert field in chunk, (
                    f"Chunk {chunk.get('chunk_id', '?')} missing required field: {field}"
                )

    @pytest.mark.parametrize("chunks_file", [ALEXANDER_CHUNKS, COLE_CHUNKS])
    def test_no_empty_chunks(self, chunks_file):
        """No chunk should have empty core_text."""
        _require(chunks_file)
        with open(chunks_file) as f:
            chunks = json.load(f)

        for chunk in chunks:
            assert len(chunk["core_text"].strip()) > 0, (
                f"Chunk {chunk['chunk_id']} has empty core_text"
            )

    @pytest.mark.parametrize("chunks_file", [ALEXANDER_CHUNKS, COLE_CHUNKS])
    def test_chunk_ids_unique(self, chunks_file):
        """All chunk IDs must be unique within a document."""
        _require(chunks_file)
        with open(chunks_file) as f:
            chunks = json.load(f)

        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), (
            f"Duplicate chunk IDs found: {len(ids)} total, {len(set(ids))} unique"
        )

    @pytest.mark.parametrize("chunks_file", [ALEXANDER_CHUNKS, COLE_CHUNKS])
    def test_chunk_ids_sequential(self, chunks_file):
        """Chunk IDs should end with sequential numbers."""
        _require(chunks_file)
        with open(chunks_file) as f:
            chunks = json.load(f)

        for i, chunk in enumerate(chunks):
            expected_suffix = f"_chunk_{i:04d}"
            assert chunk["chunk_id"].endswith(expected_suffix), (
                f"Chunk {i} ID '{chunk['chunk_id']}' should end with '{expected_suffix}'"
            )

    @pytest.mark.parametrize("chunks_file", [ALEXANDER_CHUNKS, COLE_CHUNKS])
    def test_every_chunk_has_citation_string(self, chunks_file):
        """Every chunk must have a non-empty citation_string."""
        _require(chunks_file)
        with open(chunks_file) as f:
            chunks = json.load(f)

        for chunk in chunks:
            assert len(chunk["citation_string"].strip()) > 0, (
                f"Chunk {chunk['chunk_id']} has empty citation_string"
            )

    def test_deposition_doc_type_correct(self):
        """Alexander chunks should have doc_type 'deposition'."""
        _require(ALEXANDER_CHUNKS)
        with open(ALEXANDER_CHUNKS) as f:
            chunks = json.load(f)

        for chunk in chunks:
            assert chunk.get("doc_type") == "deposition", (
                f"Chunk {chunk['chunk_id']} has doc_type '{chunk.get('doc_type')}', "
                f"expected 'deposition'"
            )

    def test_expert_report_doc_type_correct(self):
        """Cole Report chunks should have doc_type 'expert_report'."""
        _require(COLE_CHUNKS)
        with open(COLE_CHUNKS) as f:
            chunks = json.load(f)

        for chunk in chunks:
            assert chunk.get("doc_type") == "expert_report", (
                f"Chunk {chunk['chunk_id']} has doc_type '{chunk.get('doc_type')}', "
                f"expected 'expert_report'"
            )

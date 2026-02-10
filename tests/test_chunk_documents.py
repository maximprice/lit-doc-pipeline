"""Tests for document chunking."""

import json
from pathlib import Path

import pytest

from chunk_documents import DocumentChunker
from citation_types import DocumentType


CONVERTED_DIR = Path(__file__).parent.parent / "tests" / "pipeline_output" / "converted"


@pytest.fixture
def chunker():
    return DocumentChunker(str(CONVERTED_DIR))


class TestDepositionChunking:
    def test_chunks_created(self, chunker):
        """Verify chunks are created for Alexander deposition."""
        if not (CONVERTED_DIR / "daniel_alexander_10_24_2025.md").exists():
            pytest.skip("Alexander deposition not available")

        chunks = chunker.chunk_document(
            "daniel_alexander_10_24_2025",
            DocumentType.DEPOSITION,
            "Daniel Alexander - 10-24-2025.pdf"
        )

        assert len(chunks) > 10, "Should create multiple chunks"

    def test_chunk_has_citation_metadata(self, chunker):
        """Verify chunks have complete citation data."""
        if not (CONVERTED_DIR / "daniel_alexander_10_24_2025.md").exists():
            pytest.skip("Alexander deposition not available")

        chunks = chunker.chunk_document(
            "daniel_alexander_10_24_2025",
            DocumentType.DEPOSITION,
            "Daniel Alexander - 10-24-2025.pdf"
        )

        chunk = chunks[0]
        assert chunk.chunk_id
        assert chunk.pages
        assert chunk.citation_string
        assert "Dep." in chunk.citation_string

    def test_deposition_citation_string_format(self, chunker):
        """Verify deposition citation strings have page:line format."""
        if not (CONVERTED_DIR / "daniel_alexander_10_24_2025.md").exists():
            pytest.skip("Alexander deposition not available")

        chunks = chunker.chunk_document(
            "daniel_alexander_10_24_2025",
            DocumentType.DEPOSITION,
            "Daniel Alexander - 10-24-2025.pdf"
        )

        # Check a chunk with line ranges
        for chunk in chunks:
            if chunk.citation.get("transcript_lines"):
                assert "Dep." in chunk.citation_string
                # Should have format like "14:5" or "14:5-12"
                assert re.search(r"\d+:\d+", chunk.citation_string)
                break


class TestExpertReportChunking:
    def test_cole_report_chunks_created(self, chunker):
        """Verify chunks created for Cole Report."""
        if not (CONVERTED_DIR / "2025_12_11_cole_report.md").exists():
            pytest.skip("Cole Report not available")

        chunks = chunker.chunk_document(
            "2025_12_11_cole_report",
            DocumentType.EXPERT_REPORT,
            "2025-12-11 - Cole Report.pdf"
        )

        assert len(chunks) > 5, "Should create multiple chunks"

    def test_footnotes_in_chunks(self, chunker):
        """Verify footnotes are included in chunks."""
        if not (CONVERTED_DIR / "2025_12_11_cole_report.md").exists():
            pytest.skip("Cole Report not available")

        chunks = chunker.chunk_document(
            "2025_12_11_cole_report",
            DocumentType.EXPERT_REPORT,
            "2025-12-11 - Cole Report.pdf"
        )

        # Find chunk with paragraph 25
        para_25_chunks = [c for c in chunks if "TWT effective" in c.core_text]
        assert len(para_25_chunks) > 0, "Should find chunk with paragraph 25"

        chunk = para_25_chunks[0]
        assert "[FOOTNOTE 1:" in chunk.core_text, "Should include footnote 1"
        assert "[FOOTNOTE 2:" in chunk.core_text, "Should include footnote 2"

    def test_expert_report_has_page_metadata(self, chunker):
        """Verify expert report chunks have page numbers."""
        if not (CONVERTED_DIR / "2025_12_11_cole_report.md").exists():
            pytest.skip("Cole Report not available")

        chunks = chunker.chunk_document(
            "2025_12_11_cole_report",
            DocumentType.EXPERT_REPORT,
            "2025-12-11 - Cole Report.pdf"
        )

        assert all(chunk.pages for chunk in chunks), "All chunks should have page numbers"


class TestChunkMetadata:
    def test_chunk_has_required_fields(self, chunker):
        """Verify all chunks have required fields."""
        if not (CONVERTED_DIR / "daniel_alexander_10_24_2025.md").exists():
            pytest.skip("Test files not available")

        chunks = chunker.chunk_document(
            "daniel_alexander_10_24_2025",
            DocumentType.DEPOSITION,
            "test.pdf"
        )

        for chunk in chunks:
            assert chunk.chunk_id
            assert chunk.core_text
            assert chunk.citation_string
            assert chunk.tokens > 0
            assert isinstance(chunk.citation, dict)

    def test_chunks_saved_to_json(self, chunker):
        """Verify chunks are saved to JSON file."""
        if not (CONVERTED_DIR / "daniel_alexander_10_24_2025.md").exists():
            pytest.skip("Test files not available")

        chunker.chunk_document(
            "daniel_alexander_10_24_2025",
            DocumentType.DEPOSITION,
            "test.pdf"
        )

        chunks_file = CONVERTED_DIR / "daniel_alexander_10_24_2025_chunks.json"
        assert chunks_file.exists(), "Chunks file should be created"

        # Verify it's valid JSON
        with open(chunks_file) as f:
            data = json.load(f)
            assert isinstance(data, list)
            assert len(data) > 0


import re

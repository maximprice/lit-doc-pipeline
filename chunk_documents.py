"""
Section-aware chunking for litigation documents.

Creates semantic chunks that preserve document structure and citation metadata:
- Depositions: Never split Q/A pairs
- Expert reports: Preserve paragraph boundaries with inline footnotes
- Patents: Preserve claim structure and column formatting
- All types: Attach complete citation metadata (page, Bates, line numbers)

Output: Context cards ready for vector indexing.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from citation_types import DocumentType, Chunk

logger = logging.getLogger(__name__)

# Chunking parameters
DEFAULT_TARGET_TOKENS = 800  # Target chunk size
DEFAULT_MAX_TOKENS = 1200    # Hard limit before forcing split
DEFAULT_OVERLAP_TOKENS = 100  # Overlap between chunks
CHARS_PER_TOKEN = 4          # Rough estimate: 1 token ≈ 4 characters


@dataclass
class ChunkMetadata:
    """Metadata collected while building a chunk."""
    text_ids: List[str] = field(default_factory=list)
    pages: List[int] = field(default_factory=list)
    bates_stamps: List[str] = field(default_factory=list)
    line_ranges: Dict[int, Tuple[int, int]] = field(default_factory=dict)  # page -> (line_start, line_end)
    paragraph_numbers: List[int] = field(default_factory=list)
    columns: List[int] = field(default_factory=list)
    transcript_pages: List[int] = field(default_factory=list)


class DocumentChunker:
    """
    Create semantic chunks from processed markdown with citation metadata.

    Reads:
    - {stem}.md: Markdown with [TEXT:N], [PAGE:N], [FOOTNOTE:...] markers
    - {stem}_citations.json: Citation metadata keyed by #/texts/N or line_P*_L*

    Outputs:
    - {stem}_chunks.json: Array of context cards with full citation data
    """

    def __init__(
        self,
        converted_dir: str,
        target_tokens: int = DEFAULT_TARGET_TOKENS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    ):
        self.converted_dir = Path(converted_dir)
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    # Type sets for handler routing
    TRANSCRIPT_TYPES = {DocumentType.DEPOSITION, DocumentType.HEARING_TRANSCRIPT}
    PARAGRAPH_TYPES = {
        DocumentType.EXPERT_REPORT, DocumentType.PLEADING,
        DocumentType.DECLARATION, DocumentType.MOTION,
        DocumentType.BRIEF, DocumentType.WITNESS_STATEMENT,
        DocumentType.AGREEMENT,
    }
    PATENT_TYPES = {DocumentType.PATENT}

    def chunk_document(
        self,
        stem: str,
        doc_type: DocumentType = DocumentType.UNKNOWN,
        source_file: str = "",
        source_path: str = "",
    ) -> List[Chunk]:
        """
        Create chunks from a processed document.

        Args:
            stem: Document stem (e.g., "daniel_alexander_10_24_2025")
            doc_type: Document type for type-specific handling
            source_file: Original PDF filename for metadata

        Returns:
            List of Chunk objects with complete citation data
        """
        # Load inputs
        md_content, citations = self._load_inputs(stem)
        if not md_content or not citations:
            logger.warning("Missing inputs for %s", stem)
            return []

        # Parse markdown into sections
        sections = self._parse_markdown(md_content, doc_type)
        logger.info("Parsed %d sections from markdown", len(sections))

        # Create chunks from sections - route by type sets
        chunks = []
        if doc_type in self.TRANSCRIPT_TYPES:
            chunks = self._chunk_deposition(sections, citations, stem, source_file, source_path=source_path)
        elif doc_type in self.PARAGRAPH_TYPES:
            chunks = self._chunk_expert_report(sections, citations, stem, source_file, source_path=source_path)
        elif doc_type in self.PATENT_TYPES:
            chunks = self._chunk_patent(sections, citations, stem, source_file, source_path=source_path)
        else:
            chunks = self._chunk_generic(sections, citations, stem, source_file, source_path=source_path)

        logger.info("Created %d chunks from %s", len(chunks), stem)

        # Save chunks
        output_path = self.converted_dir / f"{stem}_chunks.json"
        with open(output_path, "w") as f:
            json.dump([c.to_dict() for c in chunks], f, indent=2)

        logger.info("Saved chunks to %s", output_path)
        return chunks

    # ── Loading ──────────────────────────────────────────────────────

    def _load_inputs(self, stem: str) -> Tuple[Optional[str], Optional[dict]]:
        """Load markdown and citations JSON."""
        md_path = self.converted_dir / f"{stem}.md"
        citations_path = self.converted_dir / f"{stem}_citations.json"

        if not md_path.exists():
            logger.error("Markdown file not found: %s", md_path)
            return None, None

        if not citations_path.exists():
            logger.error("Citations file not found: %s", citations_path)
            return None, None

        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        with open(citations_path, "r", encoding="utf-8") as f:
            citations = json.load(f)

        return md_content, citations

    # ── Markdown Parsing ─────────────────────────────────────────────

    def _parse_markdown(self, content: str, doc_type: DocumentType) -> List[dict]:
        """
        Parse markdown into logical sections.

        Returns:
            List of section dicts with 'lines', 'text_markers', 'type'.
            Each entry in 'lines' is a tuple (line_text, text_id_or_None)
            where text_id is the most recent [TEXT:N] marker that applies
            to this line.
        """
        lines = content.split("\n")
        sections = []
        current_section = {
            "lines": [],
            "text_markers": [],
            "type": "content"
        }
        active_text_id = None  # Track the most recent [TEXT:N]

        for line in lines:
            # Detect section boundaries
            if line.startswith("## ") or line.startswith("# "):
                # Save current section and start new one
                if current_section["lines"]:
                    sections.append(current_section)
                current_section = {
                    "lines": [(line, active_text_id)],
                    "text_markers": [],
                    "type": "header"
                }
                continue

            # Track [TEXT:N] markers
            if line.startswith("[TEXT:"):
                match = re.match(r"\[TEXT:(\d+)\]", line)
                if match:
                    active_text_id = match.group(1)
                    current_section["text_markers"].append(active_text_id)
                # Don't add the marker line to content
                continue

            # Skip other markers (will be parsed when needed)
            if line.startswith("[PAGE:") or line.startswith("[BATES:"):
                current_section["lines"].append((line, active_text_id))
                continue

            current_section["lines"].append((line, active_text_id))

        # Add final section
        if current_section["lines"]:
            sections.append(current_section)

        return sections

    # ── Deposition Chunking ──────────────────────────────────────────

    def _chunk_deposition(
        self,
        sections: List[dict],
        citations: dict,
        stem: str,
        source_file: str,
        source_path: str = "",
    ) -> List[Chunk]:
        """
        Chunk deposition preserving Q/A pairs.

        CRITICAL: Never split a Q from its A.
        """
        chunks = []
        current_chunk_lines = []
        # Per-line tracking: (page, bates) for each line in current_chunk_lines
        current_line_attrs = []
        current_metadata = ChunkMetadata()

        for section in sections:
            for line, _text_id in section["lines"]:
                # Track page markers
                if line.startswith("[PAGE:"):
                    match = re.match(r"\[PAGE:(\d+)\]", line)
                    if match:
                        page = int(match.group(1))
                        if page not in current_metadata.transcript_pages:
                            current_metadata.transcript_pages.append(page)
                    continue

                # Parse line number and Q/A marker
                match = re.match(r"^\s*(\d{1,2})\s+([QA])\s+(.+)$", line)
                if match:
                    line_num = int(match.group(1))
                    qa_marker = match.group(2)
                    text = match.group(3)

                    # Look up citation for this line (PyMuPDF key first, then Docling fallback)
                    current_page = current_metadata.transcript_pages[-1] if current_metadata.transcript_pages else 1
                    line_page = current_page
                    line_bates = None
                    cit = self._find_deposition_citation(citations, current_page, line_num)
                    if cit:
                        page = cit.get("page", current_page)
                        line_page = page
                        if page not in current_metadata.pages:
                            current_metadata.pages.append(page)

                        # Track line range for this page
                        if current_page not in current_metadata.line_ranges:
                            current_metadata.line_ranges[current_page] = (line_num, line_num)
                        else:
                            start, end = current_metadata.line_ranges[current_page]
                            current_metadata.line_ranges[current_page] = (min(start, line_num), max(end, line_num))

                        bates = cit.get("bates")
                        line_bates = bates
                        if bates and bates not in current_metadata.bates_stamps:
                            current_metadata.bates_stamps.append(bates)

                    current_chunk_lines.append(line)
                    current_line_attrs.append((line_page, line_bates))

                    # Check if we should start a new chunk
                    # Rule 1: If this is a Q, check if adding the expected A would exceed max_tokens
                    # Rule 2: Never split before an A
                    chunk_text = "\n".join(current_chunk_lines)
                    tokens = len(chunk_text) // CHARS_PER_TOKEN

                    if qa_marker == "A" and tokens >= self.target_tokens:
                        # Build per-line maps from tracked attrs
                        page_map = [a[0] for a in current_line_attrs]
                        bates_map = [a[1] for a in current_line_attrs]
                        # Complete chunk after this A
                        chunk = self._create_chunk(
                            chunk_text, current_metadata, stem, source_file,
                            len(chunks), DocumentType.DEPOSITION,
                            page_map=page_map, bates_map=bates_map,
                            source_path=source_path,
                        )
                        chunks.append(chunk)

                        # Start new chunk with overlap (last few lines)
                        overlap_count = min(3, len(current_chunk_lines))
                        overlap_lines = current_chunk_lines[-overlap_count:] if overlap_count else []
                        overlap_attrs = current_line_attrs[-overlap_count:] if overlap_count else []
                        current_chunk_lines = overlap_lines
                        current_line_attrs = overlap_attrs
                        current_metadata = ChunkMetadata()
                        # Preserve page/Bates from overlap
                        if chunk.pages:
                            current_metadata.pages = [chunk.pages[-1]]
                            current_metadata.transcript_pages = [chunk.citation.get("transcript_pages", [])[-1]] if chunk.citation.get("transcript_pages") else []
                        if chunk.citation.get("bates_range"):
                            current_metadata.bates_stamps = [chunk.citation["bates_range"][-1]]

                else:
                    current_chunk_lines.append(line)
                    # Non-Q/A lines: inherit last known page/bates
                    last_page = current_line_attrs[-1][0] if current_line_attrs else None
                    last_bates = current_line_attrs[-1][1] if current_line_attrs else None
                    current_line_attrs.append((last_page, last_bates))

        # Add final chunk
        if current_chunk_lines:
            chunk_text = "\n".join(current_chunk_lines)
            page_map = [a[0] for a in current_line_attrs]
            bates_map = [a[1] for a in current_line_attrs]
            chunk = self._create_chunk(
                chunk_text, current_metadata, stem, source_file,
                len(chunks), DocumentType.DEPOSITION,
                page_map=page_map, bates_map=bates_map,
                source_path=source_path,
            )
            chunks.append(chunk)

        return chunks

    def _find_deposition_citation(
        self,
        citations: dict,
        page: int,
        line_num: int,
    ) -> Optional[dict]:
        """Look up citation data for a deposition line.

        Tries the PyMuPDF key format first (line_P{page}_L{line}), then
        falls back to scanning all citations for a matching page + line_start
        (Docling #/texts/N key format).

        Returns the citation dict, or None if not found.
        """
        # Fast path: PyMuPDF key format
        pymupdf_key = f"line_P{page}_L{line_num}"
        if pymupdf_key in citations:
            return citations[pymupdf_key]

        # Fallback: scan for Docling-style keys with matching page + line_start
        for key, cit in citations.items():
            if (cit.get("transcript_page") == page
                    and cit.get("line_start") == line_num):
                return cit
            # Also check page field for non-transcript citations
            if (cit.get("page") == page
                    and cit.get("line_start") == line_num
                    and cit.get("type") == "transcript_line"):
                return cit

        return None

    # ── Expert Report Chunking ───────────────────────────────────────

    def _chunk_expert_report(
        self,
        sections: List[dict],
        citations: dict,
        stem: str,
        source_file: str,
        source_path: str = "",
    ) -> List[Chunk]:
        """
        Chunk expert report by paragraphs with inline footnotes.

        Strategy: Preserve paragraph boundaries, include footnotes with their paragraphs.
        """
        chunks = []
        current_chunk_entries = []  # (line_text, text_id) tuples
        current_metadata = ChunkMetadata()

        for section in sections:
            for line_text, text_id in section["lines"]:
                # Skip empty lines
                if not line_text.strip():
                    current_chunk_entries.append((line_text, text_id))
                    continue

                # Detect paragraph start (numbered paragraphs)
                para_match = re.match(r"^(\d+)\.\s+", line_text)

                # Check if we should start a new chunk
                chunk_text = "\n".join(t for t, _ in current_chunk_entries)
                tokens = len(chunk_text) // CHARS_PER_TOKEN

                # Split at paragraph boundaries when target size reached
                if para_match and tokens >= self.target_tokens and current_chunk_entries:
                    page_map, bates_map = self._build_line_maps(current_chunk_entries, citations)
                    chunk = self._create_chunk(
                        chunk_text, current_metadata, stem, source_file,
                        len(chunks), DocumentType.EXPERT_REPORT,
                        page_map=page_map, bates_map=bates_map,
                        source_path=source_path,
                    )
                    chunks.append(chunk)

                    # Start new chunk (no overlap for expert reports - paragraphs are self-contained)
                    current_chunk_entries = []
                    current_metadata = ChunkMetadata()

                current_chunk_entries.append((line_text, text_id))

                # Apply citation metadata for this line's text_id
                if text_id and text_id not in current_metadata.text_ids:
                    current_metadata.text_ids.append(text_id)
                    cite_key = f"#/texts/{text_id}"
                    if cite_key in citations:
                        self._update_metadata(current_metadata, citations[cite_key])

        # Add final chunk
        if current_chunk_entries:
            chunk_text = "\n".join(t for t, _ in current_chunk_entries)
            page_map, bates_map = self._build_line_maps(current_chunk_entries, citations)
            chunk = self._create_chunk(
                chunk_text, current_metadata, stem, source_file,
                len(chunks), DocumentType.EXPERT_REPORT,
                page_map=page_map, bates_map=bates_map,
                source_path=source_path,
            )
            chunks.append(chunk)

        return chunks

    # ── Patent Chunking ──────────────────────────────────────────────

    def _chunk_patent(
        self,
        sections: List[dict],
        citations: dict,
        stem: str,
        source_file: str,
        source_path: str = "",
    ) -> List[Chunk]:
        """Chunk patent preserving claim structure."""
        # For now, use generic chunking
        # TODO: Implement claim-aware chunking
        return self._chunk_generic(sections, citations, stem, source_file, source_path=source_path)

    # ── Generic Chunking ─────────────────────────────────────────────

    def _chunk_generic(
        self,
        sections: List[dict],
        citations: dict,
        stem: str,
        source_file: str,
        source_path: str = "",
    ) -> List[Chunk]:
        """Generic chunking by token count."""
        chunks = []
        # Each entry is (line_text, text_id_or_None)
        current_chunk_entries = []
        current_metadata = ChunkMetadata()

        for section in sections:
            for line_text, text_id in section["lines"]:
                # Apply citation metadata for this line's text_id
                if text_id and text_id not in current_metadata.text_ids:
                    current_metadata.text_ids.append(text_id)
                    cite_key = f"#/texts/{text_id}"
                    if cite_key in citations:
                        self._update_metadata(current_metadata, citations[cite_key])

                current_chunk_entries.append((line_text, text_id))

                # Check chunk size
                chunk_text = "\n".join(t for t, _ in current_chunk_entries)
                tokens = len(chunk_text) // CHARS_PER_TOKEN

                if tokens >= self.target_tokens:
                    page_map, bates_map = self._build_line_maps(current_chunk_entries, citations)
                    chunk = self._create_chunk(
                        chunk_text, current_metadata, stem, source_file,
                        len(chunks), DocumentType.UNKNOWN,
                        page_map=page_map, bates_map=bates_map,
                        source_path=source_path,
                    )
                    chunks.append(chunk)

                    # Start new chunk with overlap
                    overlap_entries = current_chunk_entries[-5:] if len(current_chunk_entries) > 5 else []
                    current_chunk_entries = overlap_entries
                    current_metadata = ChunkMetadata()
                    # Re-apply metadata for overlap lines
                    for _, oid in overlap_entries:
                        if oid and oid not in current_metadata.text_ids:
                            current_metadata.text_ids.append(oid)
                            cite_key = f"#/texts/{oid}"
                            if cite_key in citations:
                                self._update_metadata(current_metadata, citations[cite_key])

        # Add final chunk
        if current_chunk_entries:
            chunk_text = "\n".join(t for t, _ in current_chunk_entries)
            page_map, bates_map = self._build_line_maps(current_chunk_entries, citations)
            chunk = self._create_chunk(
                chunk_text, current_metadata, stem, source_file,
                len(chunks), DocumentType.UNKNOWN,
                page_map=page_map, bates_map=bates_map,
                source_path=source_path,
            )
            chunks.append(chunk)

        return chunks

    # ── Metadata Helpers ─────────────────────────────────────────────

    def _build_line_maps(
        self,
        entries: List[Tuple[str, Optional[str]]],
        citations: dict,
    ) -> Tuple[List[Optional[int]], List[Optional[str]]]:
        """
        Build per-line page_map and bates_map from chunk entries.

        Each entry is (line_text, text_id). For each line in the joined
        core_text, look up the page and bates from the citation keyed by
        text_id. Forward-fill None gaps from the last known value.

        Returns:
            (page_map, bates_map) — parallel lists, one entry per line of core_text.
        """
        page_map: List[Optional[int]] = []
        bates_map: List[Optional[str]] = []

        for _line_text, text_id in entries:
            page = None
            bates = None
            if text_id:
                cite_key = f"#/texts/{text_id}"
                cit = citations.get(cite_key, {})
                page = cit.get("page")
                bates = cit.get("bates")
            page_map.append(page)
            bates_map.append(bates)

        # Forward-fill None gaps
        last_page = None
        last_bates = None
        for i in range(len(page_map)):
            if page_map[i] is not None:
                last_page = page_map[i]
            else:
                page_map[i] = last_page
            if bates_map[i] is not None:
                last_bates = bates_map[i]
            else:
                bates_map[i] = last_bates

        return page_map, bates_map

    def _update_metadata(self, metadata: ChunkMetadata, citation: dict):
        """Update chunk metadata from a citation dict."""
        page = citation.get("page")
        if page and page not in metadata.pages:
            metadata.pages.append(page)

        bates = citation.get("bates")
        if bates and bates not in metadata.bates_stamps:
            metadata.bates_stamps.append(bates)

        line_start = citation.get("line_start")
        line_end = citation.get("line_end")
        transcript_page = citation.get("transcript_page")

        if transcript_page and line_start:
            if transcript_page not in metadata.line_ranges:
                metadata.line_ranges[transcript_page] = (line_start, line_end or line_start)
            else:
                start, end = metadata.line_ranges[transcript_page]
                metadata.line_ranges[transcript_page] = (
                    min(start, line_start),
                    max(end, line_end or line_start)
                )

        para = citation.get("paragraph_number")
        if para and para not in metadata.paragraph_numbers:
            metadata.paragraph_numbers.append(para)

        col = citation.get("column")
        if col and col not in metadata.columns:
            metadata.columns.append(col)

        tp = citation.get("transcript_page")
        if tp and tp not in metadata.transcript_pages:
            metadata.transcript_pages.append(tp)

    def _create_chunk(
        self,
        text: str,
        metadata: ChunkMetadata,
        stem: str,
        source_file: str,
        chunk_idx: int,
        doc_type: DocumentType,
        page_map: Optional[List[Optional[int]]] = None,
        bates_map: Optional[List[Optional[str]]] = None,
        source_path: str = "",
    ) -> Chunk:
        """Create a Chunk object with complete citation metadata."""
        # Generate chunk ID
        chunk_id = f"{stem}_chunk_{chunk_idx:04d}"

        # Build citation dict
        citation = {
            "pdf_pages": sorted(set(metadata.pages)),
            "bates_range": metadata.bates_stamps,
        }

        # Store per-line maps for precise search attribution
        if page_map:
            citation["page_map"] = page_map
        if bates_map and any(b is not None for b in bates_map):
            citation["bates_map"] = bates_map

        # Add type-specific fields
        if metadata.line_ranges:
            citation["transcript_lines"] = {
                str(pg): list(rng) for pg, rng in metadata.line_ranges.items()
            }
            citation["transcript_pages"] = sorted(set(metadata.transcript_pages))

        if metadata.paragraph_numbers:
            citation["paragraph_numbers"] = sorted(set(metadata.paragraph_numbers))

        if metadata.columns:
            citation["column_lines"] = {
                "columns": sorted(set(metadata.columns))
            }

        # Generate citation string
        citation_string = self._generate_citation_string(
            stem, doc_type, metadata
        )

        # Calculate tokens
        tokens = len(text) // CHARS_PER_TOKEN

        return Chunk(
            chunk_id=chunk_id,
            core_text=text,
            pages=sorted(set(metadata.pages)),
            citation=citation,
            citation_string=citation_string,
            tokens=tokens,
            doc_type=doc_type,
            source_path=source_path or None,
        )

    def _generate_citation_string(
        self,
        stem: str,
        doc_type: DocumentType,
        metadata: ChunkMetadata,
    ) -> str:
        """Generate human-readable citation string."""
        # Extract document name from stem
        doc_name = stem.replace("_", " ").title()

        # Citation format by type suffix
        TYPE_SUFFIX = {
            DocumentType.DEPOSITION: "Dep.",
            DocumentType.HEARING_TRANSCRIPT: "Tr.",
        }

        if doc_type in self.TRANSCRIPT_TYPES:
            suffix = TYPE_SUFFIX.get(doc_type, "Tr.")
            if metadata.line_ranges:
                ranges = []
                for pg in sorted(metadata.line_ranges.keys()):
                    start, end = metadata.line_ranges[pg]
                    if start == end:
                        ranges.append(f"{pg}:{start}")
                    else:
                        ranges.append(f"{pg}:{start}-{end}")
                return f"{doc_name} {suffix} {', '.join(ranges)}"
            return f"{doc_name} {suffix}"

        elif doc_type in self.PARAGRAPH_TYPES:
            if metadata.paragraph_numbers:
                paras = sorted(set(metadata.paragraph_numbers))
                if len(paras) == 1:
                    return f"{doc_name} ¶{paras[0]}"
                else:
                    return f"{doc_name} ¶¶{paras[0]}-{paras[-1]}"
            return f"{doc_name}"

        elif doc_type in self.PATENT_TYPES:
            if metadata.columns:
                cols = sorted(set(metadata.columns))
                return f"{doc_name}, col. {cols[0]}"
            return f"{doc_name}"

        else:
            # Generic: "Document Name, p. 14 [BATES_001]"
            bates_suffix = ""
            if metadata.bates_stamps:
                bates_suffix = f" [{metadata.bates_stamps[0]}]"
            if metadata.pages:
                pages = sorted(set(metadata.pages))
                if len(pages) == 1:
                    return f"{doc_name}, p. {pages[0]}{bates_suffix}"
                else:
                    return f"{doc_name}, pp. {pages[0]}-{pages[-1]}{bates_suffix}"
            if bates_suffix:
                return f"{doc_name}{bates_suffix}"
            return f"{doc_name}"


def chunk_all_documents(
    converted_dir: str,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    doc_type_map: Optional[Dict[str, DocumentType]] = None,
    source_path_map: Optional[Dict[str, str]] = None,
) -> Dict[str, List[Chunk]]:
    """
    Chunk all documents in a converted directory.

    Args:
        converted_dir: Directory containing .md and _citations.json files
        target_tokens: Target chunk size in tokens
        doc_type_map: Pre-computed mapping of stem -> DocumentType from classifier
        source_path_map: Pre-computed mapping of stem -> original relative path

    Returns:
        Dict mapping stem to list of chunks
    """
    converted_dir = Path(converted_dir)
    chunker = DocumentChunker(str(converted_dir), target_tokens=target_tokens)
    if doc_type_map is None:
        doc_type_map = {}
    if source_path_map is None:
        source_path_map = {}

    results = {}
    for md_file in sorted(converted_dir.glob("*.md")):
        stem = md_file.stem

        # Skip if citations file doesn't exist
        citations_file = converted_dir / f"{stem}_citations.json"
        if not citations_file.exists():
            logger.warning("No citations file for %s, skipping", stem)
            continue

        # Look up doc type from classifier map, fall back to inferring from citations
        doc_type = doc_type_map.get(stem, DocumentType.UNKNOWN)
        if doc_type == DocumentType.UNKNOWN:
            doc_type = _infer_type_from_citations(citations_file)

        source_path = source_path_map.get(stem, "")
        chunks = chunker.chunk_document(stem, doc_type, md_file.name, source_path=source_path)
        results[stem] = chunks

    return results


def _infer_type_from_citations(citations_path: Path) -> DocumentType:
    """Infer document type from citation type fields in _citations.json."""
    try:
        with open(citations_path) as f:
            citations = json.load(f)
    except (json.JSONDecodeError, OSError):
        return DocumentType.UNKNOWN

    type_counts: Dict[str, int] = {}
    for cit in citations.values():
        t = cit.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    if "transcript_line" in type_counts:
        return DocumentType.DEPOSITION
    if "patent_column" in type_counts:
        return DocumentType.PATENT
    if "paragraph" in type_counts and type_counts["paragraph"] > 3:
        return DocumentType.EXPERT_REPORT
    return DocumentType.UNKNOWN

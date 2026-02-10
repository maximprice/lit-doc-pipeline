"""
Post-processor for cleaning markdown and enhancing citation markers.

Strategy:
1. Parse markdown structure
2. Identify sections with citation context
3. Insert structured citation markers
4. Create citation map: text_position → citation_data
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from citation_types import ProcessingResult, CitationData, DocumentType


class PostProcessor:
    """
    Clean markdown while enhancing citation markers.

    CRITICAL: Must preserve citation information while cleaning text.
    Never strip line numbers, page markers, or Bates stamps before
    associating them with text content.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize post-processor.

        Args:
            config: Configuration dict with bates patterns, etc.
        """
        self.config = config or {}
        self.bates_patterns = self._load_bates_patterns()

    def _load_bates_patterns(self) -> List[re.Pattern]:
        """Load Bates stamp patterns from config."""
        pattern_strs = self.config.get('bates_patterns', [
            r'INTEL_PROX_\d{5,11}',
            r'PROX_INTEL-\d{5,11}',
            r'[A-Z]{2,}[-_][A-Z]{2,}[-_]\d{5,}'
        ])
        return [re.compile(p) for p in pattern_strs]

    def process(self, md_path: str, doc_type: DocumentType = DocumentType.UNKNOWN) -> ProcessingResult:
        """
        Clean markdown and enhance with structured citation markers.

        Args:
            md_path: Path to markdown file from conversion
            doc_type: Detected document type

        Returns:
            ProcessingResult with paths and coverage info
        """
        md_path = Path(md_path)

        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Clean content
        cleaned = self._clean_markdown(content)

        # Enhance with structured citation markers
        enhanced, citation_map = self._add_citation_markers(cleaned, doc_type)

        # Save enhanced markdown
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(enhanced)

        # Save citation map
        citation_path = str(md_path).replace('.md', '_citations.json')
        with open(citation_path, 'w', encoding='utf-8') as f:
            json.dump(citation_map, f, indent=2)

        return ProcessingResult(
            cleaned_path=str(md_path),
            citations_path=citation_path,
            citation_coverage=len(citation_map)
        )

    def _clean_markdown(self, content: str) -> str:
        """
        Clean markdown content while preserving citation markers.

        Removes:
        - Concordances (alphabetical word lists)
        - Table of contents
        - OCR artifacts (excessive whitespace, garbled characters)
        - Boilerplate headers/footers

        Preserves:
        - Line numbers (1-25 in depositions)
        - Page markers
        - Bates stamps
        - Q/A structure
        - Paragraph numbers
        """
        lines = content.split('\n')
        cleaned_lines = []

        # Track state
        in_concordance = False
        in_toc = False

        for line in lines:
            # Detect concordance sections (alphabetical word lists)
            if re.match(r'^[A-Z]\s*$', line) and len(cleaned_lines) > 0:
                # Single letter lines might indicate concordance
                if self._is_concordance_section(lines, lines.index(line)):
                    in_concordance = True
                    continue

            # Detect table of contents
            if re.match(r'^(table\s+of\s+contents|contents)', line, re.IGNORECASE):
                in_toc = True
                continue

            # Exit TOC when we hit substantive content
            if in_toc and (
                re.match(r'^\d{1,2}\s+[QA]\s+', line) or  # Deposition Q/A
                re.match(r'^[¶§]\s*\d+', line) or  # Paragraph marker
                re.match(r'^CLAIM \d+', line, re.IGNORECASE)  # Patent claim
            ):
                in_toc = False

            # Skip if in concordance or TOC
            if in_concordance or in_toc:
                continue

            # Skip excessive whitespace
            if re.match(r'^\s*$', line) and (
                not cleaned_lines or re.match(r'^\s*$', cleaned_lines[-1])
            ):
                continue

            # Skip OCR artifacts (lines with excessive underscores or dots)
            if re.match(r'^[_\.\s]{10,}$', line):
                continue

            # Clean excessive whitespace within line
            line = re.sub(r'\s{3,}', '  ', line)

            cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _is_concordance_section(self, lines: List[str], start_idx: int) -> bool:
        """
        Detect if we're at the start of a concordance section.

        Concordances are alphabetical word lists, typically at end of document.
        Heuristic: Next 5-10 lines are single words in alphabetical order.
        """
        if start_idx + 10 >= len(lines):
            return False

        words = []
        for i in range(start_idx, min(start_idx + 10, len(lines))):
            line = lines[i].strip()
            if re.match(r'^[A-Za-z]+$', line):
                words.append(line.lower())
            else:
                break

        # Check if words are in alphabetical order
        if len(words) >= 5:
            return words == sorted(words)

        return False

    def _add_citation_markers(
        self,
        content: str,
        doc_type: DocumentType
    ) -> Tuple[str, Dict[str, Dict]]:
        """
        Add structured citation markers to markdown.

        Strategy:
        1. Parse content for citation elements (pages, lines, Bates, etc.)
        2. Insert structured markers: [PAGE:14], [LINE:5], [BATES:...]
        3. Build citation map: line_index → CitationData

        Args:
            content: Cleaned markdown content
            doc_type: Detected document type

        Returns:
            Tuple of (enhanced_content, citation_map)
        """
        lines = content.split('\n')
        enhanced_lines = []
        citation_map = {}

        current_page = None
        current_bates = None
        current_line_num = None

        for idx, line in enumerate(lines):
            # Detect page markers
            page_match = re.search(r'(?:Page|p\.)\s+(\d+)', line, re.IGNORECASE)
            if page_match:
                current_page = int(page_match.group(1))
                enhanced_lines.append(f"[PAGE:{current_page}]")
                enhanced_lines.append(line)
                continue

            # Detect Bates stamps
            bates_match = None
            for pattern in self.bates_patterns:
                bates_match = pattern.search(line)
                if bates_match:
                    current_bates = bates_match.group(0)
                    enhanced_lines.append(f"[BATES:{current_bates}]")
                    break

            # Detect deposition line numbers
            if doc_type == DocumentType.DEPOSITION:
                line_match = re.match(r'^\s*(\d{1,2})\s+([QA])\s+(.*)$', line)
                if line_match:
                    current_line_num = int(line_match.group(1))
                    qa_marker = line_match.group(2)
                    text = line_match.group(3)

                    # Create citation entry
                    citation_data = CitationData(
                        page=current_page,
                        line_start=current_line_num,
                        line_end=current_line_num,
                        bates=current_bates,
                        type="transcript_line"
                    )
                    citation_map[f"line_{idx}"] = citation_data.to_dict()

                    # Add line marker and text
                    enhanced_lines.append(f"[LINE:{current_line_num}]")
                    enhanced_lines.append(f"{current_line_num}  {qa_marker}  {text}")
                    continue

            # Detect patent column markers
            if doc_type == DocumentType.PATENT:
                col_match = re.search(r'col(?:umn)?\.?\s*(\d+)', line, re.IGNORECASE)
                if col_match:
                    column = int(col_match.group(1))
                    enhanced_lines.append(f"[COLUMN:{column}]")

            # Detect paragraph markers
            para_match = re.match(r'^([¶§]\s*\d+)', line)
            if para_match:
                para_num = int(re.search(r'\d+', para_match.group(1)).group(0))
                enhanced_lines.append(f"[PARA:{para_num}]")

                # Create citation entry
                citation_data = CitationData(
                    page=current_page,
                    paragraph_number=para_num,
                    bates=current_bates,
                    type="paragraph"
                )
                citation_map[f"line_{idx}"] = citation_data.to_dict()

            # Add original line
            enhanced_lines.append(line)

        enhanced_content = '\n'.join(enhanced_lines)
        return enhanced_content, citation_map


def main():
    """Test post-processor on a sample markdown file."""
    import sys
    if len(sys.argv) < 2:
        print("Usage: python post_processor.py <markdown_file>")
        sys.exit(1)

    processor = PostProcessor()
    result = processor.process(sys.argv[1])

    print(result)


if __name__ == "__main__":
    main()

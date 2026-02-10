# Litigation Document Processing Pipeline - Technical Requirements Document

**Version:** 2.0
**Date:** February 9, 2026
**Purpose:** Complete technical specification for rebuilding the litigation document processing pipeline from scratch with proper citation tracking

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Document Conversion Pipeline](#document-conversion-pipeline)
4. [Citation Tracking System](#citation-tracking-system)
5. [Chunking & Semantic Processing](#chunking--semantic-processing)
6. [Context Card Generation](#context-card-generation)
7. [Vector-Based Retrieval System](#vector-based-retrieval-system)
8. [Cross-Encoder Reranker](#cross-encoder-reranker)
9. [LLM Enrichment Pipeline](#llm-enrichment-pipeline)
10. [Document Type Specifications](#document-type-specifications)
11. [Known Issues & Critical Fixes](#known-issues--critical-fixes)
12. [Technical Stack](#technical-stack)

---

## 1. Executive Summary

### 1.1 Purpose

This system converts litigation documents (PDFs, DOCX, etc.) into structured, searchable formats optimized for LLM-assisted legal analysis. The pipeline must preserve **precise citation information** (page numbers, Bates stamps, line numbers, column numbers, paragraph numbers) required for legal work while efficiently processing large document sets.

### 1.2 Core Requirements

1. **Citation Accuracy** - Every extracted chunk must be traceable to specific pages, lines, columns, or Bates numbers
2. **No Garbage Text** - Image processing must not generate massive amounts of meaningless text
3. **Hybrid Search** - Combine BM25 keyword search with semantic vector search
4. **Scalability** - Handle 100+ documents, 10,000+ pages efficiently
5. **Quality Validation** - Automated quality checks for each processed document

### 1.3 Key Innovations

- **Dual-format processing**: Extract from both Docling JSON (page metadata) and cleaned Markdown (readable text)
- **Citation reconstruction**: Associate line numbers, page markers, and Bates stamps with text chunks before post-processing
- **Selective OCR**: Enable OCR for scanned documents while disabling image enrichment models that generate garbage
- **Hybrid retrieval**: BM25 + vector embeddings + cross-encoder reranker for optimal relevance

---

## 2. System Architecture

### 2.1 Pipeline Overview

```
┌─────────────────┐
│  Input Docs     │ PDF, DOCX, XLSX, EML
│  (Litigation)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 1: CONVERSION (Docling)                            │
│ - Extract to JSON (structure + page metadata)           │
│ - Extract to Markdown (readable text)                   │
│ - Bates extraction from headers/footers                 │
│ - Page number tracking via provenance data              │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 2: POST-PROCESSING                                 │
│ - Remove concordances, boilerplate, OCR artifacts       │
│ - Save Bates sidecar files                              │
│ - Preserve page/line markers in text                    │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 3: CITATION RECONSTRUCTION (NEW!)                  │
│ - Parse line numbers from JSON provenance               │
│ - Associate line ranges with text elements              │
│ - Build page:line mappings for depositions              │
│ - Extract column:line for patents                       │
│ - Track paragraph numbers for expert reports            │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 4: CHUNKING                                        │
│ - Section-aware chunking with overlap                   │
│ - Inherit citation metadata from source elements        │
│ - Calculate line/page ranges for each chunk             │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 5: CONTEXT CARDS                                   │
│ - Generate structured cards with full citation data     │
│ - Keyword-based categorization                          │
│ - Entity extraction                                     │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 6: VECTOR INDEXING                                 │
│ - BM25 keyword index (scikit-learn TfidfVectorizer)     │
│ - Chroma vector store (nomic-embed-text embeddings)     │
│ - Metadata indexing for filtered search                 │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 7: OPTIONAL LLM ENRICHMENT                         │
│ - Summary generation                                    │
│ - Key quote extraction (verbatim from core_text)        │
│ - Enhanced categorization                               │
│ - Relevance scoring                                     │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ RETRIEVAL API   │ Hybrid Search + Reranker
│ - Query         │
│ - Results       │
└─────────────────┘
```

### 2.2 Data Flow

**Input:** Raw litigation documents
**Intermediate:**
- Docling JSON (structure + page provenance)
- Cleaned Markdown (readable text)
- Bates sidecar JSON (page → Bates mapping)
- Citation metadata JSON (page → line number mapping)

**Output:**
- Context cards JSON (chunks with full citation data)
- BM25 index (keyword search)
- Chroma vector store (semantic search)
- Manifest JSON (processing metadata)

---

## 3. Document Conversion Pipeline

### 3.1 Docling Configuration

**Purpose:** Extract text and structure from PDFs while preserving page metadata WITHOUT generating garbage text from images.

#### 3.1.1 Critical Docling Flags

```bash
docling \
  --to md --to json \
  --image-export-mode placeholder \
  --no-enrich-picture-classes \
  --no-enrich-picture-description \
  --no-enrich-chart-extraction \
  --output <converted_dir> \
  <input_file>
```

**Why Each Flag Matters:**

| Flag | Purpose | Problem it Solves |
|------|---------|-------------------|
| `--to md --to json` | **Both** formats needed | Markdown for readable text, JSON for page metadata |
| `--image-export-mode placeholder` | Don't embed base64 images | Prevents 10MB+ garbage base64 strings in markdown |
| `--no-enrich-picture-classes` | Disable image classification | Prevents "person, indoor, laptop" spam text |
| `--no-enrich-picture-description` | Disable image captions | Prevents AI-generated hallucinated descriptions |
| `--no-enrich-chart-extraction` | Disable chart-to-table | Prevents garbage from failed chart parsing |
| `--ocr` (enabled by default) | Extract text from scans | **Keep enabled** for scanned PDFs, needed for INTEL_PROX docs |

**DO NOT USE:**
- `--no-ocr` - Disabling OCR makes scanned documents useless
- `--image-export-mode embedded` - Creates massive base64 strings
- `--enrich-picture-*` flags enabled - Generates garbage descriptions

#### 3.1.2 Docling JSON Structure

The JSON output contains critical page provenance data:

```json
{
  "texts": [
    {
      "text": "Q      Have you seen this document before?",
      "label": "text",
      "prov": [
        {
          "page_no": 14,
          "bbox": {...},
          "charspan": [0, 40]
        }
      ]
    },
    {
      "text": "5",
      "label": "text",
      "prov": [
        {
          "page_no": 14,
          "bbox": {"l": 73.2, "t": 563.655, ...}
        }
      ]
    }
  ]
}
```

**Key observations:**
- Line numbers (1-25) appear as isolated `text` elements
- `prov[].page_no` gives PDF page number
- `bbox` coordinates show left margin position (line numbers typically `l < 100`)
- Each text element can span multiple pages (tracked in `prov` array)

### 3.2 Post-Processing

**Purpose:** Clean OCR artifacts and boilerplate while extracting citation metadata BEFORE removal.

#### 3.2.1 Post-Processor Operations

**For Markdown:**
1. Strip base64 images (regex: `!\[Image\]\(data:image/[^;]+;base64,.*?\)`)
2. Remove concordance sections (word indexes at end of depositions)
3. Remove isolated line numbers **AFTER** extracting them (see Citation Tracking)
4. Preserve standalone page numbers (e.g., "Page 45")
5. Remove boilerplate footers (copyright notices, disclaimers)
6. Deduplicate repeated headers

**For JSON:**
1. Extract Bates numbers from `page_header` and `page_footer` labels
2. Build `page_no → bates_number` mapping
3. Save as `{stem}_bates.json` sidecar file
4. Remove `page_header` and `page_footer` elements from texts array
5. Remove isolated line numbers from texts array
6. Remove concordance sections

#### 3.2.2 Bates Extraction Patterns

```python
BATES_PATTERNS = [
    re.compile(r'INTEL_PROX_\d{5,11}'),
    re.compile(r'PROX_INTEL-\d{5,11}'),
    re.compile(r'[A-Z]{2,}[-_][A-Z]{2,}[-_]\d{5,}'),
]
```

**Why this matters:** Bates numbers often appear in headers/footers that get stripped. Extract BEFORE cleaning.

#### 3.2.3 Post-Processing Output

```
converted/
  ├── document.md           # Cleaned markdown
  ├── document.json         # Cleaned JSON (optional: delete with --cleanup-json)
  └── document_bates.json   # Bates sidecar: {"1": "INTEL_PROX_00001770", "2": "INTEL_PROX_00001771", ...}
```

---

## 4. Citation Tracking System

### 4.1 The Critical Problem

**Issue Discovered:** The pipeline was tracking page numbers but NOT line numbers, column numbers, or paragraph numbers. For legal citations, you need:

- **Depositions:** `"Daniel Alexander Dep. 45:12-18"` (page 45, lines 12-18)
- **Patents:** `"'152 Patent, col. 3:45-55"` (column 3, lines 45-55)
- **Expert Reports:** `"Fuja Report ¶ 42"` (paragraph 42)

Without this, the pipeline produces **useless citations** for legal work.

### 4.2 Citation Reconstruction Architecture

**New Component:** `citation_tracker.py`

This module runs AFTER post-processing but BEFORE chunking. It:
1. Reads the Docling JSON (before it's deleted)
2. Identifies citation elements (line numbers, column markers, paragraph numbers)
3. Associates them with nearby text elements using bbox proximity
4. Builds citation ranges for each text element
5. Saves `{stem}_citations.json` metadata file

#### 4.2.1 Line Number Reconstruction (Depositions)

**Algorithm:**

```python
def reconstruct_line_numbers(texts: list) -> dict:
    """
    Build mapping: text_index → (page, line_start, line_end)

    Algorithm:
    1. Identify line number elements (text is "1"-"25", bbox.l < 100)
    2. Sort by (page, vertical position)
    3. For each text element on the page:
       - Find line numbers with similar vertical position (±20px)
       - Assign start_line = first matching line number
       - Assign end_line = last matching line number before next text
    4. Handle wrapped text (multiple bbox spans)
    """
    line_mapping = {}

    # Group by page
    for page_no in set(t['prov'][0]['page_no'] for t in texts):
        page_texts = [t for t in texts if t['prov'][0]['page_no'] == page_no]

        # Separate line numbers from content
        line_nums = [t for t in page_texts
                     if t['text'].isdigit() and 1 <= int(t['text']) <= 25
                     and t['prov'][0]['bbox']['l'] < 100]

        content = [t for t in page_texts if t not in line_nums]

        # Assign lines to content by vertical proximity
        for text_elem in content:
            text_bbox = text_elem['prov'][0]['bbox']

            # Find line numbers within ±20px vertically
            nearby_lines = [
                int(ln['text']) for ln in line_nums
                if abs(ln['prov'][0]['bbox']['t'] - text_bbox['t']) < 20
            ]

            if nearby_lines:
                line_mapping[text_elem['self_ref']] = {
                    'page': page_no,
                    'line_start': min(nearby_lines),
                    'line_end': max(nearby_lines)
                }

    return line_mapping
```

**Output Format:**

```json
{
  "#/texts/42": {
    "page": 14,
    "line_start": 5,
    "line_end": 12,
    "type": "transcript_line"
  }
}
```

#### 4.2.2 Column Number Reconstruction (Patents)

**Algorithm:**

```python
def reconstruct_column_numbers(texts: list) -> dict:
    """
    Extract column:line citations from patent text.

    Patents have two formats:
    1. Physical columns (two-column layout with line numbers)
    2. Inline citations ("col. 3:45-55")

    Algorithm:
    1. Detect column markers in text ("col. 3")
    2. Associate subsequent line numbers with that column
    3. For inline citations, parse and store ranges
    4. For physical columns, use bbox.l position (left column < 300, right > 300)
    """
    column_mapping = {}
    current_column = None

    for i, text_elem in enumerate(texts):
        text = text_elem['text']

        # Detect column marker
        col_match = re.search(r'col(?:umn)?\.?\s*(\d+)', text, re.IGNORECASE)
        if col_match:
            current_column = int(col_match.group(1))

        # Detect line number in margin
        if text.isdigit() and 1 <= int(text) <= 100:
            line_no = int(text)

            # Find next content element
            for j in range(i+1, min(i+5, len(texts))):
                if texts[j]['text'] and not texts[j]['text'].isdigit():
                    column_mapping[texts[j]['self_ref']] = {
                        'column': current_column or (1 if texts[j]['prov'][0]['bbox']['l'] < 300 else 2),
                        'line_start': line_no,
                        'type': 'patent_column'
                    }
                    break

    return column_mapping
```

**Output Format:**

```json
{
  "#/texts/156": {
    "column": 3,
    "line_start": 45,
    "line_end": 55,
    "type": "patent_column"
  }
}
```

#### 4.2.3 Paragraph Number Reconstruction (Expert Reports)

**Algorithm:**

```python
def reconstruct_paragraph_numbers(texts: list) -> dict:
    """
    Track paragraph numbers in expert reports.

    Format: "¶ 42" or "¶42" or "Paragraph 42"

    Algorithm:
    1. Find paragraph markers in text
    2. Associate with following text elements
    3. Track paragraph ranges for multi-paragraph sections
    """
    para_mapping = {}
    current_para = None

    for i, text_elem in enumerate(texts):
        text = text_elem['text']

        # Detect paragraph marker
        para_match = re.search(r'[¶§]\s*(\d+)|paragraph\s+(\d+)', text, re.IGNORECASE)
        if para_match:
            current_para = int(para_match.group(1) or para_match.group(2))

        # New paragraph marker resets
        if current_para:
            para_mapping[text_elem['self_ref']] = {
                'paragraph': current_para,
                'type': 'paragraph_number'
            }

    return para_mapping
```

#### 4.2.4 Bates Number Association

**Algorithm:**

```python
def associate_bates_numbers(texts: list, bates_sidecar: dict) -> dict:
    """
    Associate Bates stamps with text elements.

    Input: bates_sidecar = {"1": "INTEL_PROX_00001770", "2": "INTEL_PROX_00001771", ...}

    Algorithm:
    1. For each text element, get its page_no from prov
    2. Look up Bates number for that page
    3. Store in mapping
    """
    bates_mapping = {}

    for text_elem in texts:
        page_no = text_elem['prov'][0]['page_no']
        bates = bates_sidecar.get(str(page_no))

        if bates:
            bates_mapping[text_elem['self_ref']] = {
                'bates': bates,
                'page': page_no,
                'type': 'bates_number'
            }

    return bates_mapping
```

### 4.3 Citation Metadata Output

**File:** `converted/{stem}_citations.json`

```json
{
  "#/texts/42": {
    "page": 14,
    "line_start": 5,
    "line_end": 12,
    "bates": "INTEL_PROX_00001784",
    "type": "transcript_line"
  },
  "#/texts/156": {
    "page": 3,
    "column": 2,
    "line_start": 45,
    "line_end": 55,
    "type": "patent_column"
  },
  "#/texts/89": {
    "page": 12,
    "paragraph": 42,
    "bates": "EXPERT_FUJA_00023",
    "type": "paragraph_number"
  }
}
```

### 4.4 Integration with Chunking

**Critical:** Chunking must inherit citation metadata from source elements.

When chunking creates a chunk from texts [42, 43, 44]:
1. Look up citation metadata for each source text
2. Merge page ranges: pages = [14] (if all same page) or [14, 15] (if spanning)
3. Merge line ranges: line_start = min(all line_starts), line_end = max(all line_ends)
4. Store in chunk metadata

**Chunk Citation Format:**

```json
{
  "chunk_id": "daniel-alexander-depo-chunk-5",
  "source": "Daniel Alexander - 10-24-2025.pdf",
  "core_text": "Q  Have you seen this document before? ...",
  "citation": {
    "pdf_pages": [14],
    "transcript_pages": [14],
    "transcript_lines": {"14": [5, 12]},
    "bates_range": ["INTEL_PROX_00001784"],
    "paragraph_numbers": []
  },
  "citation_string": "Daniel Alexander Dep. 14:5-12"
}
```

---

## 5. Chunking & Semantic Processing

### 5.1 Chunking Strategy

**Goal:** Create overlapping semantic chunks suitable for LLM context windows while preserving citation boundaries.

#### 5.1.1 Section-Aware Chunking

```python
CHUNKING_CONFIG = {
    "min_chunk_chars": 300,
    "max_chunk_chars": 15000,
    "target_chunk_chars": 8000,
    "overlap_paragraphs": 3,
    "respect_boundaries": ["section", "subsection", "exhibit"]
}
```

**Algorithm:**

1. **Section Detection:** Identify natural boundaries (headers, exhibits, speaker changes in depositions)
2. **Initial Split:** Split on section boundaries
3. **Size Adjustment:** If section > max_chunk_chars, split on paragraph boundaries
4. **Overlap:** Add 3 paragraphs from previous chunk to maintain context
5. **Citation Inheritance:** Merge citation data from all source elements

#### 5.1.2 Deposition-Specific Chunking

Depositions require special handling:

```python
def chunk_deposition(sections: list, citations: dict) -> list:
    """
    Chunk depositions preserving Q&A structure.

    Rules:
    1. Never split Q&A pairs
    2. Keep line number attribution
    3. Group by topic (detect topic changes)
    4. Target 10-15 Q&A exchanges per chunk
    """
    chunks = []
    current_chunk = []
    current_lines = []

    for section in sections:
        # Detect Q or A
        if section['text'].startswith('Q ') or section['text'].startswith('A '):
            current_chunk.append(section)

            # Collect line numbers
            if section['ref'] in citations:
                cite = citations[section['ref']]
                current_lines.extend(range(cite['line_start'], cite['line_end'] + 1))

            # Chunk when reaching size limit
            if len(''.join(s['text'] for s in current_chunk)) > 8000:
                chunks.append({
                    'text': '\n\n'.join(s['text'] for s in current_chunk),
                    'citation': {
                        'line_start': min(current_lines),
                        'line_end': max(current_lines)
                    }
                })
                # Overlap: keep last 2 Q&A pairs
                current_chunk = current_chunk[-4:]  # Last 2 Q&A = 4 elements (Q, A, Q, A)
                current_lines = []

    return chunks
```

### 5.2 Chunk Metadata Structure

**Every chunk must include:**

```json
{
  "chunk_id": "unique-identifier",
  "source": "original_filename.pdf",
  "doc_type": "deposition|patent|expert_report|pleading|exhibit",
  "core_text": "The actual text content...",
  "pages": [14, 15],
  "citation": {
    "pdf_pages": [14, 15],
    "transcript_pages": [14, 15],  // If deposition
    "transcript_lines": {
      "14": [5, 25],
      "15": [1, 12]
    },
    "column_lines": {               // If patent
      "column": 3,
      "lines": [45, 55]
    },
    "paragraph_numbers": [42, 43],  // If expert report
    "bates_range": ["INTEL_PROX_00001784", "INTEL_PROX_00001785"]
  },
  "citation_string": "Daniel Alexander Dep. 14:5-15:12",
  "section_title": "Examination by Mr. Hecht",
  "tokens": 1250
}
```

---

## 6. Context Card Generation

### 6.1 Context Card Purpose

Context cards are the primary unit of retrieval. Each card represents a semantically coherent section with:
- Full citation data
- Categorization
- Entity extraction
- Optional LLM enrichment

### 6.2 Card Schema

```json
{
  "card_id": "unique-card-id",
  "source": "document.pdf",
  "doc_type": "deposition",
  "chunk_index": 5,

  "core_text": "Full text of the chunk...",

  "pages": [14, 15],
  "citation": { /* Full citation object */ },
  "citation_string": "Daniel Alexander Dep. 14:5-15:12",

  "category": "witness_statement",
  "relevance_score": "high",
  "claims_addressed": [1, 7],
  "statutes_cited": [],

  "summary": "Optional LLM-generated summary",
  "key_quotes": ["Exact quote from core_text"],

  "entities": {
    "people": ["Daniel Alexander", "David Hecht"],
    "organizations": ["Intel Corporation"],
    "technologies": ["TWT", "IEEE 802.11"]
  },

  "classification_method": "keyword|llm",
  "llm_backend": "claude-code|ollama|anthropic"
}
```

### 6.3 Keyword-Based Categorization

**Categories:**

```python
CATEGORIES = [
    "legal_argument",
    "factual_allegation",
    "evidence_reference",
    "expert_opinion",
    "witness_statement",
    "damages_analysis",
    "procedural",
    "background",
    "statutory_regulatory",
    "case_citation",
    "settlement_negotiation",
    "discovery",
    "claim_construction",
    "prior_art",
    "infringement_analysis",
    "validity_invalidity",
    "market_definition",
    "anticompetitive_conduct",
    "competitive_effects",
    "contract_interpretation",
    "breach_analysis",
    "performance_obligations",
    "unclassified"
]
```

**Categorization Logic:**

```python
def categorize_card(text: str, doc_type: str) -> str:
    """
    Keyword-based categorization with doc-type specific rules.
    """
    text_lower = text.lower()

    # Witness statements (depositions)
    if doc_type == "deposition":
        if re.search(r'\bQ\s+', text) and re.search(r'\bA\s+', text):
            return "witness_statement"

    # Expert opinions
    if re.search(r'\b(in my (opinion|view)|my analysis|conclude that)\b', text_lower):
        return "expert_opinion"

    # Legal arguments
    if re.search(r'\b(argue|contend|submit|assert|plaintiff maintains)\b', text_lower):
        return "legal_argument"

    # Claim construction (patents)
    if re.search(r'\b(claim|element|limitation|means for)\b', text_lower):
        if re.search(r'\b(construe|construction|interpret)\b', text_lower):
            return "claim_construction"
        return "infringement_analysis"

    # Damages
    if re.search(r'\b(damages|lost profits|reasonable royalty|but-for|unjust enrichment)\b', text_lower):
        return "damages_analysis"

    # Discovery
    if re.search(r'\b(interrogatory|request for production|admission|subpoena)\b', text_lower):
        return "discovery"

    # Statutory/regulatory
    if re.search(r'\b(\d+\s+(U\.S\.C\.|C\.F\.R\.|§))\b', text):
        return "statutory_regulatory"

    return "unclassified"
```

### 6.4 Entity Extraction

**Basic keyword-based extraction:**

```python
def extract_entities(text: str, doc_type: str) -> dict:
    """
    Extract entities using regex patterns.
    """
    entities = {
        "people": [],
        "organizations": [],
        "technologies": [],
        "patents": [],
        "case_citations": [],
        "statutes": []
    }

    # People (capitalized names near "testified", "stated", "Dr.", "Mr.", "Ms.")
    people_pattern = r'\b(?:Dr\.|Mr\.|Ms\.|Prof\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
    entities["people"] = list(set(re.findall(people_pattern, text)))

    # Organizations (Corp., Inc., LLC, Co.)
    org_pattern = r'\b([A-Z][A-Za-z0-9\s&]+(?:Corp(?:oration)?|Inc|LLC|Co|Ltd)\.?)\b'
    entities["organizations"] = list(set(re.findall(org_pattern, text)))

    # Patents (US patent numbers)
    patent_pattern = r'\b(?:U\.S\.\s+)?Patent\s+No\.\s+([\d,]+)\b'
    entities["patents"] = list(set(re.findall(patent_pattern, text)))

    # Statutes
    statute_pattern = r'\b\d+\s+(?:U\.S\.C\.|C\.F\.R\.)\s+§\s*[\d\.]+\b'
    entities["statutes"] = list(set(re.findall(statute_pattern, text)))

    # Case citations
    case_pattern = r'\b([A-Z][a-z]+(?:\s+v\.\s+[A-Z][a-z]+)(?:,\s+\d+\s+(?:U\.S\.|F\.\d?d?)\s+\d+)?)\b'
    entities["case_citations"] = list(set(re.findall(case_pattern, text)))

    return entities
```

---

## 7. Vector-Based Retrieval System

### 7.1 Hybrid Search Architecture

**Three-tier search system:**

1. **BM25 (Keyword)** - TF-IDF based, exact term matching
2. **Semantic Vectors (Chroma)** - Dense embeddings, conceptual similarity
3. **Cross-Encoder Reranker** - Pairwise relevance scoring

### 7.2 BM25 Index (scikit-learn)

**Implementation:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class BM25Searcher:
    def __init__(self, documents: list, k1=1.5, b=0.75):
        """
        BM25 using TF-IDF as approximation.

        Args:
            k1: Term frequency saturation (default 1.5)
            b: Length normalization (default 0.75)
        """
        self.k1 = k1
        self.b = b
        self.documents = documents

        # Build TF-IDF matrix
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            sublinear_tf=True  # Use log(tf) instead of tf
        )
        self.tfidf_matrix = self.vectorizer.fit_transform([d['core_text'] for d in documents])

        # Store document lengths for BM25
        self.doc_lengths = np.array([len(d['core_text'].split()) for d in documents])
        self.avg_doc_length = self.doc_lengths.mean()

    def search(self, query: str, top_k: int = 20) -> list:
        """
        Search using BM25 scoring.
        """
        # Vectorize query
        query_vec = self.vectorizer.transform([query])

        # Compute cosine similarity (TF-IDF approximation)
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Apply BM25 length normalization
        length_norm = 1 - self.b + self.b * (self.doc_lengths / self.avg_doc_length)
        scores = scores / length_norm

        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                **self.documents[idx],
                'score': float(scores[idx]),
                'rank': len(results) + 1,
                'method': 'bm25'
            })

        return results
```

### 7.3 Chroma Vector Store

**Implementation:**

```python
import chromadb
from chromadb.config import Settings

class ChromaSemanticSearcher:
    def __init__(self, persist_directory: str, embedding_function=None):
        """
        Chroma vector store for semantic search.

        Args:
            persist_directory: Where to store the database
            embedding_function: Optional custom embeddings (default: nomic-embed-text via Ollama)
        """
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Default: nomic-embed-text embeddings via Ollama
        if embedding_function is None:
            from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
            embedding_function = OllamaEmbeddingFunction(
                model_name="nomic-embed-text",
                url="http://localhost:11434/api/embeddings"
            )

        self.collection = self.client.get_or_create_collection(
            name="litigation_docs",
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: list):
        """
        Add context cards to vector store.
        """
        ids = [d['card_id'] for d in documents]
        texts = [d['core_text'] for d in documents]

        # Metadata for filtering
        metadatas = [{
            'source': d['source'],
            'doc_type': d.get('doc_type', 'unknown'),
            'category': d.get('category', 'unclassified'),
            'pages': str(d.get('pages', [])),
            'citation_string': d.get('citation_string', '')
        } for d in documents]

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )

    def search(self, query: str, top_k: int = 20, filter_dict: dict = None) -> list:
        """
        Semantic search with optional filtering.

        Args:
            query: Search query
            top_k: Number of results
            filter_dict: Metadata filters (e.g., {"doc_type": "deposition"})
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filter_dict
        )

        cards = []
        for i, doc_id in enumerate(results['ids'][0]):
            cards.append({
                'card_id': doc_id,
                'core_text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': 1.0 - results['distances'][0][i],  # Convert distance to similarity
                'rank': i + 1,
                'method': 'semantic'
            })

        return cards
```

### 7.4 Hybrid Fusion

**Score Fusion Strategy:**

```python
class HybridSearcher:
    def __init__(self, bm25_searcher, chroma_searcher,
                 bm25_weight=0.5, semantic_weight=0.5):
        self.bm25 = bm25_searcher
        self.chroma = chroma_searcher
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight

    def search(self, query: str, top_k: int = 10, candidate_multiplier: int = 2) -> list:
        """
        Hybrid search with reciprocal rank fusion.

        Args:
            candidate_multiplier: Fetch this many extra candidates for reranking
        """
        candidate_k = top_k * candidate_multiplier

        # Get candidates from both methods
        bm25_results = self.bm25.search(query, top_k=candidate_k)
        semantic_results = self.chroma.search(query, top_k=candidate_k)

        # Normalize scores to [0, 1]
        bm25_results = self._normalize_scores(bm25_results)
        semantic_results = self._normalize_scores(semantic_results)

        # Merge results
        merged = {}
        for result in bm25_results:
            doc_id = result['card_id']
            merged[doc_id] = {
                **result,
                'bm25_score': result['score'],
                'semantic_score': 0.0,
                'final_score': result['score'] * self.bm25_weight
            }

        for result in semantic_results:
            doc_id = result['card_id']
            if doc_id in merged:
                merged[doc_id]['semantic_score'] = result['score']
                merged[doc_id]['final_score'] = (
                    merged[doc_id]['bm25_score'] * self.bm25_weight +
                    result['score'] * self.semantic_weight
                )
            else:
                merged[doc_id] = {
                    **result,
                    'bm25_score': 0.0,
                    'semantic_score': result['score'],
                    'final_score': result['score'] * self.semantic_weight
                }

        # Sort by final score
        sorted_results = sorted(merged.values(), key=lambda x: x['final_score'], reverse=True)

        return sorted_results[:top_k]

    def _normalize_scores(self, results: list) -> list:
        """Min-max normalization to [0, 1]"""
        if not results:
            return results

        scores = [r['score'] for r in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            for r in results:
                r['score'] = 1.0
        else:
            for r in results:
                r['score'] = (r['score'] - min_score) / (max_score - min_score)

        return results
```

---

## 8. Cross-Encoder Reranker

### 8.1 Purpose

The hybrid search fetches a large candidate pool (e.g., 50 results). The cross-encoder reranks these candidates using a more accurate pairwise relevance model.

**Why this matters:**
- BM25 and semantic embeddings use **independent scoring** (query vs. document)
- Cross-encoders use **joint encoding** (query + document together)
- This captures subtle relevance signals that embeddings miss

### 8.2 Implementation Plan

*See `/Users/maximprice/Dev/doc-pipeline/docs/cross-encoder-reranker-plan.md` for complete specification.*

**Summary:**

```python
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Cross-encoder reranker.

        Model: ms-marco-MiniLM-L-6-v2
        - Size: ~80MB
        - Speed: ~50ms per query-document pair on CPU
        - Training: MS MARCO passage ranking dataset
        """
        self.model_name = model_name
        self.model = None  # Lazy load

    def _ensure_model(self):
        if self.model is None:
            self.model = CrossEncoder(self.model_name)

    def rerank(self, query: str, results: list, top_k: int = 10) -> list:
        """
        Rerank results using cross-encoder.

        Args:
            query: User query
            results: List of result dicts with 'core_text' field
            top_k: Return this many top results
        """
        self._ensure_model()

        # Prepare query-document pairs
        pairs = [[query, r['core_text'][:512]] for r in results]  # Truncate to 512 chars

        # Score all pairs
        scores = self.model.predict(pairs)

        # Add scores to results
        for result, score in zip(results, scores):
            result['pre_rerank_score'] = result.get('final_score', 0.0)
            result['rerank_score'] = float(score)
            result['final_score'] = float(score)

        # Sort by rerank score
        reranked = sorted(results, key=lambda x: x['rerank_score'], reverse=True)

        return reranked[:top_k]
```

**Integration:**

```python
# In HybridSearcher.search():
if self.reranker:
    candidate_k = top_k * self.candidate_multiplier  # e.g., 10 * 10 = 100 candidates
    candidates = self._hybrid_fusion(query, candidate_k)
    final_results = self.reranker.rerank(query, candidates, top_k)
else:
    final_results = self._hybrid_fusion(query, top_k)

return final_results
```

### 8.3 Configuration

**CLI Flags:**

```bash
doc-retrieve <dir> --query "monopolistic behavior" \
  --rerank \                              # Enable reranking
  --rerank-model "ms-marco-MiniLM-L-6-v2" \  # Model choice
  --candidate-pool 10                     # Fetch 10x candidates
```

**Graceful Degradation:**

If `sentence-transformers` not installed:
- Print warning: "Reranking disabled (sentence-transformers not installed)"
- Continue with hybrid search only
- No error, no crash

---

## 9. LLM Enrichment Pipeline

### 9.1 Purpose

Optional LLM enrichment adds:
- **Summaries** - 2-3 sentence legal significance
- **Key Quotes** - Verbatim quotes from core_text (exact match required)
- **Enhanced Categorization** - More accurate than keyword-based
- **Relevance Scoring** - High/medium/low based on legal significance
- **Claims Addressed** - Which patent claims are discussed

### 9.2 Enrichment Backends

**Three options:**

1. **Claude Code (default)** - Parallel background agents, no external dependencies
2. **Ollama (local)** - Requires local Ollama server with llama3.1:8b or similar
3. **Anthropic API (cloud)** - Requires API key, highest quality

### 9.3 Enrichment Prompt Template

```python
ENRICHMENT_PROMPT = """You are analyzing a section from a litigation document. For this section, provide:

1. **summary**: 2-3 sentences describing the legal significance. What does this section establish, argue, or prove?

2. **key_quotes**: 1-5 verbatim quotes copied EXACTLY from the text below. Each quote must be a character-for-character match. Pick the most legally significant or probative passages.

3. **category**: Exactly ONE category from this list:
   - legal_argument
   - factual_allegation
   - evidence_reference
   - expert_opinion
   - witness_statement
   - damages_analysis
   - procedural
   - background
   - statutory_regulatory
   - case_citation
   - claim_construction
   - infringement_analysis
   - validity_invalidity
   - unclassified

4. **relevance_score**: One of "high", "medium", or "low":
   - high: Key admissions, critical legal arguments, dispositive evidence, expert opinions on ultimate issues
   - medium: Relevant factual content, supporting arguments, necessary context
   - low: Procedural boilerplate, tangential background, administrative content

5. **claims_addressed**: List of patent claim numbers discussed (e.g., [1, 7, 14]). If no claims mentioned, return empty list [].

**Text to analyze:**
{core_text}

**Case context:** {case_type} case involving {parties}

**Output format (JSON):**
{{
  "summary": "...",
  "key_quotes": ["exact quote 1", "exact quote 2"],
  "category": "legal_argument",
  "relevance_score": "high",
  "claims_addressed": [1, 7]
}}
"""
```

### 9.4 Post-Processing Validation

**Critical:** LLMs hallucinate. Always validate:

```python
def validate_enrichment(card: dict, enriched: dict) -> dict:
    """
    Validate LLM output and fix common errors.
    """
    validated = enriched.copy()

    # 1. Verify key quotes are verbatim
    core_text = card['core_text']
    validated['key_quotes'] = [
        quote for quote in enriched.get('key_quotes', [])
        if quote in core_text  # Exact substring match
    ]

    # 2. Validate category
    if enriched.get('category') not in CATEGORIES:
        validated['category'] = 'unclassified'

    # 3. Validate relevance score
    if enriched.get('relevance_score') not in ['high', 'medium', 'low']:
        validated['relevance_score'] = 'medium'

    # 4. Validate claims are integers
    claims = enriched.get('claims_addressed', [])
    validated['claims_addressed'] = [
        int(c) for c in claims if isinstance(c, (int, str)) and str(c).isdigit()
    ]

    # 5. Fix common claim number errors (models confuse patent numbers with claim numbers)
    # Patent numbers are 7+ digits, claim numbers are 1-2 digits
    validated['claims_addressed'] = [
        c for c in validated['claims_addressed'] if c < 100
    ]

    return validated
```

### 9.5 Parallel Enrichment (Claude Code)

**Strategy:** Process multiple card files in parallel using background agents.

```python
def enrich_cards_parallel(card_files: list, case_context: dict):
    """
    Launch parallel agents to enrich card files.

    Args:
        card_files: List of paths to *_cards.json files
        case_context: Dict with case_type, parties, patents
    """
    agents = []

    # Launch up to 5 agents in parallel
    for i in range(0, len(card_files), math.ceil(len(card_files) / 5)):
        batch = card_files[i:i + math.ceil(len(card_files) / 5)]

        agent_prompt = f"""
        Enrich the following context card files with LLM analysis:

        Files: {', '.join(batch)}

        For each file:
        1. Read the JSON (array of cards)
        2. For each card, if not already enriched (check classification_method):
           - Read core_text (first 8000 chars if longer)
           - Generate: summary, key_quotes, category, relevance_score, claims_addressed
           - Validate key_quotes are exact substrings
           - Set classification_method = "llm"
           - Set llm_backend = "claude-code"
        3. Write updated JSON back to file (indent=2)

        Case context: {case_context}
        """

        agent = Task(
            subagent_type="general-purpose",
            prompt=agent_prompt,
            run_in_background=True
        )
        agents.append(agent)

    # Wait for completion
    for agent in agents:
        result = TaskOutput(task_id=agent.id, block=True)
        print(f"Agent {agent.id} completed: {result.summary}")
```

---

## 10. Document Type Specifications

### 10.1 Depositions

**Characteristics:**
- Q&A format
- Line numbers 1-25 per page
- Page numbers (transcript pages)
- Speaker identification
- Exhibit references

**Citation Format:** `"Daniel Alexander Dep. 45:12-18"` (page 45, lines 12-18)

**Special Handling:**
1. Never split Q&A pairs when chunking
2. Track line ranges per chunk
3. Detect speaker changes (topic shifts)
4. Extract exhibit numbers
5. Handle objections (preserve in context)

**Validation:**
- Check for Q/A patterns
- Verify line numbers 1-25 present
- Confirm page markers exist

### 10.2 Patents

**Characteristics:**
- Column layout (2 columns)
- Column:line numbering (e.g., "col. 3:45")
- Claims section (numbered elements)
- Drawings/figures references
- Technical specifications

**Citation Format:** `"'152 Patent, col. 3:45-55"` (column 3, lines 45-55)

**Special Handling:**
1. Detect column boundaries (bbox.l < 300 = left column)
2. Track column:line references
3. Extract claim numbers from text
4. Parse claim elements (limitations)
5. Link figures to text references

**Validation:**
- Check for claim numbers (1-50 typical)
- Verify column references present
- Confirm technical vocabulary

### 10.3 Expert Reports

**Characteristics:**
- Paragraph numbering (¶ 1, ¶ 2, ...)
- Section headers
- Citations to evidence
- Technical analysis
- Opinions and conclusions

**Citation Format:** `"Fuja Report ¶ 42"` (paragraph 42)

**Special Handling:**
1. Track paragraph numbers (¶ markers)
2. Detect opinion statements ("in my opinion")
3. Extract citations to other documents
4. Identify technical analysis sections
5. Link opinions to supporting evidence

**Validation:**
- Check for paragraph numbering
- Verify opinion language present
- Confirm expert qualifications mentioned

### 10.4 Pleadings

**Characteristics:**
- Caption (parties, case number)
- Paragraph numbering (1, 2, 3, ...)
- Signature blocks
- Certificate of service
- Legal arguments

**Citation Format:** `"Complaint ¶ 23"` (paragraph 23)

**Special Handling:**
1. Extract caption information
2. Track paragraph numbers (Arabic numerals)
3. Identify causes of action
4. Extract prayer for relief
5. Parse party names

**Validation:**
- Check for caption
- Verify paragraph numbering
- Confirm signature block present

### 10.5 Exhibits

**Characteristics:**
- Varied formats (contracts, emails, presentations, technical docs)
- Often have exhibit numbers
- May have Bates stamps
- Context depends on referencing document

**Citation Format:** `"Exhibit A at 5"` or `"Ex. 123, BATES_00045678"`

**Special Handling:**
1. Extract exhibit number from first page
2. Preserve Bates stamps
3. Link to referencing document
4. Type detection (contract, email, technical, etc.)

**Validation:**
- Check for exhibit marker
- Verify Bates stamps if present

---

## 11. Known Issues & Critical Fixes

### 11.1 CRITICAL: Citation Tracking Not Implemented

**Status:** ❌ Not implemented
**Severity:** BLOCKER for legal use
**Impact:** Cannot generate proper legal citations

**Required Fix:**

1. Create `citation_tracker.py` module (Section 4)
2. Implement line number reconstruction for depositions
3. Implement column number reconstruction for patents
4. Implement paragraph number tracking for expert reports
5. Integrate with chunking pipeline
6. Update context card schema to include full citation objects

**Estimated Effort:** 3-5 days

### 11.2 Post-Processor Destroys Citation Data

**Status:** ❌ Active bug
**Severity:** HIGH
**Impact:** Line numbers stripped from depositions, making them unusable for citations

**Current Behavior:**
```python
# post_processor.py lines 198-200
if is_line_number(stripped):
    continue  # WRONG: Throws away citation data
```

**Required Fix:**
Extract line numbers and associate with nearby text BEFORE stripping them (Section 4.2.1).

### 11.3 JSON Output Disabled by Default

**Status:** ⚠️ Configuration issue
**Severity:** MEDIUM
**Impact:** Page metadata not available for citation tracking

**Current Config:**
```python
"output_formats": ["md"]  # WRONG: Missing JSON
```

**Fix Applied:**
```python
"output_formats": ["md", "json"]  # Correct
```

Use `--cleanup-json` flag to delete JSON after processing to save disk space.

### 11.4 Image Enrichment Generates Garbage

**Status:** ✅ FIXED (Feb 9, 2026)
**Severity:** HIGH
**Impact:** Massive garbage text from vision models describing images

**Fix Applied:**
```bash
--no-enrich-picture-classes
--no-enrich-picture-description
--no-enrich-chart-extraction
--image-export-mode placeholder
```

### 11.5 Enrichment Quality Issues

**Status:** ⚠️ Needs validation
**Severity:** MEDIUM
**Impact:** LLM enrichment contains hallucinations

**Issues Found:**
1. Key quotes not verbatim (~30% have whitespace/typo differences)
2. Claims addressed confuses patent numbers with claim numbers
3. Relevance scores default to 0.8 (not calibrated)
4. Categories misclassify licensing as infringement_analysis

**Fix Applied:**
Post-processing validation (Section 9.4) to detect and correct errors.

**Still Needed:**
- Better prompt engineering
- Model-specific calibration
- Human validation sampling

### 11.6 Deposition Chunking Splits Q&A Pairs

**Status:** ⚠️ Needs improvement
**Severity:** MEDIUM
**Impact:** Context loss when questions and answers separated

**Required Fix:**
Implement deposition-aware chunking (Section 5.1.2) that:
1. Detects Q/A pairs
2. Never splits within a pair
3. Groups by topic (speaker changes)

---

## 12. Technical Stack

### 12.1 Core Dependencies

```txt
# Document Processing
docling>=2.70.0              # Structure-aware PDF conversion
pymupdf>=1.24.0              # High-fidelity PDF extraction
marker-pdf>=0.3.0            # Alternative converter (optional)

# Vector Search
scikit-learn>=1.5.0          # TF-IDF / BM25
chromadb>=0.4.0              # Vector store
numpy>=2.0.0                 # Numerical operations

# Embeddings (choose one)
# Option 1: Local Ollama (FREE, requires Ollama server)
#   ollama pull nomic-embed-text
#
# Option 2: OpenAI (paid)
#   openai>=1.0.0

# Reranking (optional)
sentence-transformers>=3.0.0  # Cross-encoder reranking
torch>=2.0.0                  # PyTorch for transformers

# LLM Enrichment (optional, choose one)
anthropic>=0.40.0             # Claude API
# OR use Ollama locally (FREE)
```

### 12.2 System Requirements

**Minimum:**
- Python 3.10+
- 8 GB RAM
- 10 GB disk space

**Recommended:**
- Python 3.12+
- 16 GB RAM
- 50 GB disk space (for large document sets)
- SSD for vector database

**For Ollama (optional):**
- 16 GB RAM
- 20 GB disk for models

### 12.3 External Services

**Ollama (optional, for semantic search & enrichment):**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull nomic-embed-text    # 274MB embedding model
ollama pull llama3.1:8b          # 4.7GB LLM for enrichment
```

**Anthropic API (optional, for enrichment):**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 12.4 Directory Structure

```
project/
├── lit_doc_pipeline.py        # Main pipeline
├── citation_tracker.py         # NEW: Citation reconstruction
├── post_processor.py           # Text cleaning
├── lit_doc_retriever.py       # Hybrid search + reranker
├── llm_enrichment.py          # LLM enrichment
├── chunk_documents.py         # Chunking logic
├── generate_context_cards.py # Card generation
│
├── configs/
│   ├── default_config.json
│   └── patent_config.json
│
├── bin/
│   ├── doc-pipeline            # CLI wrapper
│   └── doc-retrieve            # Search CLI
│
└── tests/
    ├── test_citation_tracking.py
    ├── test_chunking.py
    └── test_retriever.py
```

---

## 13. Processing Pipeline: End-to-End Example

### 13.1 Input

```
litigation_docs/
├── Daniel Alexander - 10-24-2025.pdf    # Deposition (2.5 MB)
├── INTEL_PROX_00001770.pdf              # IEEE spec (16 MB, scanned)
├── Fuja_Expert_Report.pdf               # Expert report (5 MB)
└── complaint.pdf                         # Pleading (500 KB)
```

### 13.2 Command

```bash
doc-pipeline \
  --input-dir litigation_docs/ \
  --output-dir processed_20260209/ \
  --case-type patent \
  --cleanup-json
```

### 13.3 Processing Steps

**STEP 1: Conversion (Docling)**
```
Processing: Daniel Alexander - 10-24-2025.pdf
  ✓ Extracted to JSON (4.6 MB) - page metadata preserved
  ✓ Extracted to MD (314 KB)
  ✓ Found 25 line numbers per page (transcript format detected)
  ✓ Extracted Bates: None (deposition, no Bates)

Processing: INTEL_PROX_00001770.pdf
  ✓ Extracted to JSON (8.2 MB) - structure + page metadata
  ✓ Extracted to MD (573 KB) via OCR
  ✓ Extracted Bates: INTEL_PROX_00001770-00001985 (216 pages)
```

**STEP 2: Post-Processing**
```
Cleaning: Daniel Alexander MD
  314 KB → 126 KB (59.8% reduction)
  - Removed concordance (word index at end)
  - Removed boilerplate
  - Preserved page markers and line context

Cleaning: INTEL_PROX JSON
  ✓ Extracted Bates to sidecar: intel_prox_00001770_bates.json
  ✓ Removed headers/footers from texts array
```

**STEP 3: Citation Reconstruction (NEW)**
```
Building citation metadata: Daniel Alexander
  ✓ Identified 1,679 line numbers (1-25 per page, 67 pages)
  ✓ Associated line numbers with 423 text elements
  ✓ Built page:line mappings
  ✓ Saved: daniel_alexander_citations.json

Building citation metadata: INTEL_PROX
  ✓ Loaded Bates sidecar (216 pages)
  ✓ Associated Bates with 1,234 text elements
  ✓ Detected column layout (2 columns)
  ✓ No column numbers found (not a patent, IEEE spec)
  ✓ Saved: intel_prox_00001770_citations.json
```

**STEP 4: Chunking**
```
Chunking: Daniel Alexander (deposition)
  ✓ Detected Q&A format
  ✓ Created 162 chunks (preserving Q&A pairs)
  ✓ Inherited line numbers from citation metadata
  ✓ Average chunk: 1,007 chars (10-15 Q&A per chunk)
  ✓ Line ranges tracked: chunk_5 = "14:5-15:12"

Chunking: INTEL_PROX (technical spec)
  ✓ Section-aware chunking
  ✓ Created 207 chunks
  ✓ Inherited Bates numbers from citation metadata
  ✓ Average chunk: 3,376 chars
  ✓ Bates ranges tracked: chunk_10 = "INTEL_PROX_00001780-00001782"
```

**STEP 5: Context Cards**
```
Generating cards: Daniel Alexander
  ✓ 162 cards created
  ✓ Categories: witness_statement (85), evidence_reference (45), procedural (18), ...
  ✓ Citations: All cards have transcript page:line ranges
  ✓ Entities: 15 people, 8 organizations extracted

Generating cards: INTEL_PROX
  ✓ 207 cards created
  ✓ Categories: statutory_regulatory (156), case_citation (32), procedural (19)
  ✓ Citations: All cards have Bates ranges and page numbers
  ✓ Statutes: 47 C.F.R. § 15.247, § 15.249 extracted
```

**STEP 6: Vector Indexing**
```
Building BM25 index...
  ✓ 369 cards indexed (162 + 207)
  ✓ Vocabulary: 8,234 terms
  ✓ TF-IDF matrix: 369 × 8,234

Building Chroma vector store...
  ✓ Generating embeddings via Ollama (nomic-embed-text)
  ✓ 369 cards embedded (768-dim vectors)
  ✓ Metadata indexed: source, doc_type, category, pages, citations
  ✓ Persisted to: processed_20260209/chroma_db/
```

**STEP 7: Cleanup**
```
Cleanup: Deleting JSON files
  ✓ Deleted 4 docling JSON files (12.8 MB freed)
  ✓ Kept citation sidecars and Bates sidecars
  ✓ Kept cleaned markdown files
```

### 13.4 Output Structure

```
processed_20260209/
├── converted/
│   ├── daniel_alexander_10_24_2025.md
│   ├── daniel_alexander_10_24_2025_citations.json
│   ├── intel_prox_00001770.md
│   ├── intel_prox_00001770_bates.json
│   └── intel_prox_00001770_citations.json
│
├── chunks/
│   ├── daniel_alexander_10_24_2025_chunks.json
│   └── intel_prox_00001770_chunks.json
│
├── context_cards/
│   ├── daniel_alexander_10_24_2025_cards.json
│   ├── intel_prox_00001770_cards.json
│   └── index.md
│
├── chroma_db/
│   └── [Chroma vector store files]
│
├── entities/
│   └── all_entities.json
│
├── timeline.json
├── relationships.json
└── manifest.json
```

### 13.5 Sample Context Card (Deposition)

```json
{
  "card_id": "daniel-alexander-10-24-2025-chunk-10",
  "source": "Daniel Alexander - 10-24-2025.pdf",
  "doc_type": "deposition",
  "chunk_index": 10,

  "core_text": "Q  Have you seen this document before? This is a notice of Rule 30(b)(6) deposition?\nA  Let me read it. I think I saw it, I saw it, yes.\nQ  When did you first see this document?\nA  A couple of days ago I think, maybe.",

  "pages": [14],
  "citation": {
    "pdf_pages": [14],
    "transcript_pages": [14],
    "transcript_lines": {"14": [5, 12]},
    "bates_range": [],
    "paragraph_numbers": []
  },
  "citation_string": "Daniel Alexander Dep. 14:5-12",

  "category": "witness_statement",
  "relevance_score": "medium",
  "claims_addressed": [],

  "summary": "Witness confirms reviewing the 30(b)(6) deposition notice a few days before the deposition.",
  "key_quotes": [
    "I think I saw it, I saw it, yes.",
    "A couple of days ago I think, maybe."
  ],

  "entities": {
    "people": ["Daniel Alexander"],
    "organizations": [],
    "technologies": []
  },

  "classification_method": "llm",
  "llm_backend": "claude-code"
}
```

### 13.6 Sample Search Query

```bash
doc-retrieve processed_20260209/ \
  --query "What is the witness's understanding of TWT technology?" \
  --rerank \
  --top-k 5
```

**Output:**

```
=== Search Results ===
Query: What is the witness's understanding of TWT technology?
Method: hybrid (BM25 + semantic) + rerank

Result 1/5 [Score: 0.94]
Source: Daniel Alexander Dep. 18:5-19:3
Category: witness_statement | Relevance: high
Match: 0.72 (BM25) + 0.89 (semantic) → 0.94 (reranked)

"A  Yeah. So I understand there is a patent that Proxense wrote a while back for
technology that relates to low power devices and in casino rooms assets, stuff like
that, as far as I read. And I understand is far from any WIFI technology that Intel's
products are doing. That's my understanding. I read through the information and I
understand it's -- you think it relates to TWT. I don't see how."

---

Result 2/5 [Score: 0.87]
Source: Daniel Alexander Dep. 45:12-46:8
Category: expert_opinion | Relevance: high
Match: 0.68 (BM25) + 0.82 (semantic) → 0.87 (reranked)

"Q  Are you familiar with Target Wake Time features in Wi-Fi?
A  Yes, I designed them.
Q  Can you explain what TWT does?
A  TWT allows a device to wake up only at scheduled times to receive data,
which saves power. It's part of the 802.11ax standard."
```

---

## 14. Success Criteria

### 14.1 Functional Requirements

✅ **Must Have:**
1. Accurate citation tracking for all document types
2. Hybrid search (BM25 + semantic)
3. No garbage text from image processing
4. Support for 100+ documents, 10,000+ pages
5. Context cards with full metadata

⚠️ **Should Have:**
1. Cross-encoder reranking
2. LLM enrichment
3. Automated quality validation
4. Parallel processing

❌ **Nice to Have:**
1. OCR language detection
2. Multi-case management
3. Collaborative features
4. Web UI

### 14.2 Quality Metrics

**Citation Accuracy:** 100% of chunks must have valid citation strings
**Search Relevance:** Top-5 results should include target document 95%+ of time
**Processing Speed:** < 5 minutes per 100 pages (on modern hardware)
**False Positive Rate:** < 10% for keyword categorization
**Storage Efficiency:** < 500 MB per 1,000 pages (with cleanup-json)

### 14.3 Testing Requirements

**Unit Tests:**
- Citation reconstruction for each document type
- Chunking preserves citation metadata
- Post-processor doesn't destroy citation data
- BM25 scoring
- Vector embedding generation
- Score fusion

**Integration Tests:**
- End-to-end pipeline on sample documents
- Search relevance on known queries
- LLM enrichment validation
- Parallel processing

**Regression Tests:**
- No garbage text from images
- Page numbers preserved
- Bates numbers extracted correctly
- Line numbers associated with text

---

## 15. Implementation Roadmap

### Phase 1: Citation Foundation (CRITICAL)
**Estimated: 1 week**

1. Create `citation_tracker.py` module
2. Implement line number reconstruction (depositions)
3. Implement column number reconstruction (patents)
4. Implement paragraph number tracking (reports)
5. Integration tests for each document type
6. Update chunking to inherit citations
7. Update context card schema

**Deliverable:** Context cards with accurate citation strings

### Phase 2: Core Pipeline
**Estimated: 1 week**

1. Implement post-processing (without destroying citations)
2. Section-aware chunking
3. Context card generation
4. Entity extraction
5. Keyword categorization
6. End-to-end tests

**Deliverable:** Functional pipeline producing context cards

### Phase 3: Vector Search
**Estimated: 3 days**

1. BM25 indexing (scikit-learn)
2. Chroma vector store integration
3. Hybrid fusion
4. Search CLI
5. Search result formatting

**Deliverable:** Searchable document corpus

### Phase 4: Cross-Encoder Reranker
**Estimated: 2 days**

1. Implement reranker class
2. Integrate with hybrid search
3. CLI flags
4. Performance testing

**Deliverable:** Improved search relevance

### Phase 5: LLM Enrichment
**Estimated: 3 days**

1. Enrichment prompt engineering
2. Post-processing validation
3. Parallel enrichment (Claude Code)
4. Ollama/Anthropic backends
5. Quality metrics

**Deliverable:** Enhanced context cards with summaries and quotes

### Phase 6: Polish & Documentation
**Estimated: 2 days**

1. Quality validation reporting
2. Error handling
3. User documentation
4. API documentation
5. Example notebooks

**Deliverable:** Production-ready system

**Total Estimated Time:** 3-4 weeks

---

## Appendix A: Sample Documents

### A.1 Deposition Page (with line numbers)

```
                                                              Page 14
1
2   Q      Have you seen this document before?
3          This is a notice of Rule 30(b)(6) deposition?
4
5   A      Let me read it. I think I
6          saw it, I saw it, yes.
7
8   Q      When did you first see this document?
9
10  A      A couple of days ago I think, maybe.
11
12  Q      I'll ask you to scroll down if you start
13         on page 3, it says attachment 2, topics?
14
15  A      Page 3, attachment 2 topics.
16
17  Q      Okay so the first topic that you're
18         designated on is actually on the next page,
19         which is topic 6.
20
21         Do you see that topic?
22
23  MR. HIRSCH:  Counsel for the record again
24         I'll state as you know Mr. Alexander as
25         been designated to testify as to topics
```

**Citation Extraction:**
- Page: 14
- Lines: 1-25 (full page)
- For specific Q at line 2-3: `"Daniel Alexander Dep. 14:2-3"`
- For A at line 5-6: `"Daniel Alexander Dep. 14:5-6"`

### A.2 Patent Page (with column numbers)

```
col. 3                                    col. 4
45  method further comprises:              1  Another embodiment of the
46    detecting a wireless signal from     2  invention includes a system
47    a portable device; and               3  comprising:
48    determining proximity based on       4    a transmitter configured to
49    signal strength.                     5    send a first signal;
50                                         6    a receiver configured to
51  In one embodiment, the portable        7    detect the first signal; and
52  device includes a transceiver          8    a processor coupled to the
53  configured to communicate using        9    receiver, the processor
54  short-range wireless technology.       10   configured to determine
55                                         11   proximity based on signal
```

**Citation Extraction:**
- Page: 5 (assuming this is page 5 of the patent)
- Column 3, lines 45-55
- Column 4, lines 1-11
- Citation for col. 3 method: `"'152 Patent, col. 3:45-54"`
- Citation for col. 4 system: `"'152 Patent, col. 4:1-11"`

### A.3 Expert Report (with paragraph numbers)

```
III. TECHNICAL BACKGROUND

¶ 42   Target Wake Time (TWT) is a power-saving feature introduced in the
       IEEE 802.11ax standard. TWT allows a device to negotiate with an
       access point to define specific times when the device will be awake
       to send or receive data.

¶ 43   In my analysis of the '152 patent, claim 1 recites "a method for
       reducing power consumption in a wireless device." The claim requires
       scheduling wake times based on predicted data transmission patterns.

¶ 44   The accused Intel Wi-Fi 6E products implement TWT functionality that
       operates identically to the method described in claim 1 of the '152
       patent. Specifically, the Intel implementation:

       a) Negotiates wake schedules with the access point (element 1[a]);
       b) Maintains sleep state between scheduled wake times (element 1[b]);
       c) Adjusts wake intervals based on traffic patterns (element 1[c]).
```

**Citation Extraction:**
- Paragraphs: 42, 43, 44
- Citation for ¶ 42: `"Fuja Report ¶ 42"`
- Citation for ¶ 43: `"Fuja Report ¶ 43"`
- Citation for ¶ 44: `"Fuja Report ¶ 44"`

---

## Appendix B: Configuration Examples

### B.1 Patent Litigation Config

```json
{
  "case_type": "patent",
  "docling_path": "docling",
  "output_formats": ["md", "json"],
  "input_formats": ["pdf", "docx", "pptx", "xlsx"],

  "overlap_paragraphs": 3,
  "min_chunk_chars": 300,
  "max_chunk_chars": 15000,

  "max_conversion_workers": 4,

  "patent_numbers": ["8,036,152"],
  "asserted_claims": [1, 7, 14],
  "plaintiff": "Proxense, LLC",
  "defendant": "Intel Corporation",

  "doc_type_patterns": {
    "deposition": ["deposition", "dep.", "transcript"],
    "patent": ["patent", "US \\d{7,}"],
    "expert_report": ["expert report", "declaration of"],
    "pleading": ["complaint", "answer", "motion"]
  },

  "bates_patterns": [
    "INTEL_PROX_\\d{5,11}",
    "PROX_INTEL-\\d{5,11}"
  ]
}
```

### B.2 Retrieval Config

```json
{
  "bm25_weight": 0.5,
  "semantic_weight": 0.5,
  "rerank_enabled": true,
  "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "candidate_multiplier": 10,

  "embedding_model": "nomic-embed-text",
  "embedding_backend": "ollama",
  "ollama_host": "http://localhost:11434",

  "chroma_persist_dir": "./chroma_db",
  "chroma_collection": "litigation_docs"
}
```

---

## Appendix C: API Reference

### C.1 Context Card Schema (TypeScript)

```typescript
interface ContextCard {
  // Identifiers
  card_id: string;
  source: string;
  doc_type: "deposition" | "patent" | "expert_report" | "pleading" | "exhibit" | "unknown";
  chunk_index: number;

  // Content
  core_text: string;
  section_title?: string;

  // Citation
  pages: number[];
  citation: Citation;
  citation_string: string;

  // Classification
  category: Category;
  relevance_score: "high" | "medium" | "low";
  claims_addressed: number[];
  statutes_cited: string[];

  // Entities
  entities: {
    people: string[];
    organizations: string[];
    technologies: string[];
    patents: string[];
    case_citations: string[];
  };

  // Enrichment (optional)
  summary?: string;
  key_quotes?: string[];

  // Metadata
  classification_method: "keyword" | "llm";
  llm_backend?: "claude-code" | "ollama" | "anthropic";
  created_at: string;
  tokens: number;
}

interface Citation {
  pdf_pages: number[];
  transcript_pages: number[];
  transcript_lines: Record<string, [number, number]>; // page -> [line_start, line_end]
  column_lines?: {
    column: number;
    lines: [number, number];
  };
  paragraph_numbers: number[];
  bates_range: string[];
}

type Category =
  | "legal_argument"
  | "factual_allegation"
  | "evidence_reference"
  | "expert_opinion"
  | "witness_statement"
  | "damages_analysis"
  | "procedural"
  | "background"
  | "statutory_regulatory"
  | "case_citation"
  | "settlement_negotiation"
  | "discovery"
  | "claim_construction"
  | "prior_art"
  | "infringement_analysis"
  | "validity_invalidity"
  | "market_definition"
  | "anticompetitive_conduct"
  | "competitive_effects"
  | "contract_interpretation"
  | "breach_analysis"
  | "performance_obligations"
  | "unclassified";
```

### C.2 Search API

```python
class HybridSearcher:
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict = None,
        rerank: bool = True
    ) -> list[SearchResult]:
        """
        Hybrid search with optional filtering and reranking.

        Args:
            query: Search query string
            top_k: Number of results to return
            filters: Metadata filters, e.g.:
                {
                    "doc_type": "deposition",
                    "category": ["witness_statement", "expert_opinion"],
                    "relevance_score": "high",
                    "pages": [14, 15]  # Contains pages 14 or 15
                }
            rerank: Whether to apply cross-encoder reranking

        Returns:
            List of SearchResult objects sorted by relevance
        """
        pass

class SearchResult:
    card_id: str
    source: str
    core_text: str
    citation_string: str

    # Scores
    bm25_score: float
    semantic_score: float
    final_score: float
    rerank_score: Optional[float]

    # Metadata
    category: str
    relevance_score: str
    pages: list[int]

    # Highlighting
    matched_terms: list[str]
    snippet: str  # Highlighted snippet with matched terms
```

---

## Document Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Feb 9, 2026 | Initial draft based on existing pipeline |
| 2.0 | Feb 9, 2026 | Added critical citation tracking requirements, cross-encoder plan, comprehensive document type specs |

---

## References

1. Docling Documentation: https://github.com/DS4SD/docling
2. ChromaDB Documentation: https://docs.trychroma.com/
3. MS MARCO Cross-Encoder: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2
4. Scikit-learn TF-IDF: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
5. Federal Rules of Civil Procedure (deposition citations): https://www.law.cornell.edu/rules/frcp
6. The Bluebook (legal citation format): https://www.legalbluebook.com/

---

**END OF DOCUMENT**

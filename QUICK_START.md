# Quick Start Guide - Litigation Document Pipeline

**Current Status:** Phase 1 Complete ✅ → Ready for Testing

This guide will get you started testing the Phase 1 implementation.

---

## Installation (5 minutes)

### 1. Install Python Dependencies

```bash
cd /Users/maximprice/Dev/lit-doc-pipeline
pip install -r requirements.txt
```

This installs:
- docling (PDF/DOCX conversion)
- openpyxl (Excel support)
- extract-msg (Email support)
- python-pptx (PowerPoint support)
- And other format handlers

### 2. Verify Installation

```bash
python -c "import docling; print('✅ Docling installed')"
python -c "import openpyxl; print('✅ Excel support installed')"
```

---

## Quick Test (2 minutes)

### Test with a Sample Document

```bash
# Quick test on any document
python test_conversion.py /path/to/document.pdf

# Or try with Excel
python test_conversion.py /path/to/spreadsheet.xlsx

# Or email
python test_conversion.py /path/to/email.eml
```

**Expected Output:**
```
============================================================
Testing Conversion: document.pdf
============================================================

Using Docling converter...

✅ Conversion successful!
Citation Coverage Summary:
  Document Type: deposition
  Pages Found: 50
  Line Markers: 1250
  Bates Stamps: 50
  ...

============================================================
Post-Processing
============================================================

✅ Post-processing complete!
Processing Result:
  Cleaned File: test_output/document.md
  Citations File: test_output/document_citations.json
  Citation Coverage: 1250 elements

============================================================
Output Files
============================================================
  Markdown: test_output/document.md
  Citations: test_output/document_citations.json
```

### Review Output Files

```bash
# View cleaned markdown
cat test_output/document.md

# View citation map
cat test_output/document_citations.json | head -50
```

---

## Full Citation Coverage Test (5 minutes)

### Run Comprehensive Testing

```bash
python tests/test_phase1_citations.py /path/to/document.pdf --output-dir test_output
```

This will:
1. ✅ Convert document with Docling
2. ✅ Post-process and enhance citations
3. ✅ Analyze citation coverage
4. ✅ Generate recommendations

**Expected Output:**
```
============================================================
CITATION COVERAGE REPORT
============================================================

Coverage Statistics:
  pages:
    Found: 50
    Status: ✅ Good
  bates:
    Found: 50
    Status: ✅ Good
  line_numbers:
    Found: 1250
    Status: ✅ Good
  citation_map_entries: 1250

Recommendations:
  ✅ Found 50 page markers - good coverage.
  ✅ Found 1250 line markers - good coverage for deposition.
  ✅ Created 1250 citation entries. Ready for chunking.

============================================================

✅ Citation coverage is sufficient. Can proceed to chunking.
```

---

## What to Test

### Priority 1: Document Types

Test on at least one of each type:

1. **Deposition Transcript** (PDF with line numbers 1-25)
   ```bash
   python tests/test_phase1_citations.py deposition.pdf
   ```
   - Should find: pages, line markers (1-25), Bates stamps
   - Expected coverage: High (>80%)

2. **Patent Document** (PDF with column layout)
   ```bash
   python tests/test_phase1_citations.py patent.pdf
   ```
   - Should find: pages, column markers (col. 1, col. 2)
   - Expected coverage: Medium-High

3. **Expert Report** (PDF with paragraph numbers)
   ```bash
   python tests/test_phase1_citations.py expert_report.pdf
   ```
   - Should find: pages, paragraph markers (¶ 1, ¶ 2, ...)
   - Expected coverage: Medium-High

4. **Excel Exhibit**
   ```bash
   python tests/test_phase1_citations.py exhibit.xlsx
   ```
   - Should convert to markdown table
   - Expected coverage: Basic (pages only)

5. **Email Exhibit**
   ```bash
   python tests/test_phase1_citations.py email.eml
   ```
   - Should extract headers, body, attachments
   - Expected coverage: Basic (pages only)

### Priority 2: Edge Cases

- [ ] Scanned PDF (OCR required)
- [ ] Multi-column patent with inline citations
- [ ] Deposition with line number gaps
- [ ] Document with multiple Bates stamp formats
- [ ] Very large file (>100 pages)
- [ ] Corrupted or partial file (error handling)

---

## Understanding the Output

### 1. Markdown File (`document.md`)

Enhanced markdown with structured citation markers:

```markdown
[PAGE:14]
[BATES:INTEL_PROX_00001784]

[LINE:5]
5  Q  Have you seen this document before?
[LINE:6]
6  A  Let me read it. I think I saw it, yes.

[PAGE:15]
[BATES:INTEL_PROX_00001785]
...
```

### 2. Citations File (`document_citations.json`)

Map of text positions to citation data:

```json
{
  "line_42": {
    "page": 14,
    "line_start": 5,
    "line_end": 5,
    "bates": "INTEL_PROX_00001784",
    "type": "transcript_line"
  },
  "line_43": {
    "page": 14,
    "line_start": 6,
    "line_end": 6,
    "bates": "INTEL_PROX_00001784",
    "type": "transcript_line"
  }
}
```

### 3. Coverage Report

Statistics on what citations were extracted:

```
Coverage Statistics:
  pages: Found 50, Status: ✅ Good
  line_numbers: Found 1250, Status: ✅ Good
  bates: Found 50, Status: ✅ Good
  citation_map_entries: 1250
```

---

## Interpreting Results

### ✅ Good Coverage (>=80%)

If coverage report shows:
- ✅ Pages found
- ✅ Type-specific markers (lines/columns/paragraphs) found
- ✅ Many citation map entries

**Next Step:** Proceed directly to Phase 3 (chunking). Phase 2 reconstruction may not be needed.

### ⚠️ Medium Coverage (50-79%)

If coverage report shows:
- ✅ Pages found
- ⚠️ Some type-specific markers missing
- ⚠️ Moderate citation map entries

**Next Step:** Try improving Phase 1 regex patterns first. If that doesn't help, implement Phase 2 reconstruction.

### ❌ Low Coverage (<50%)

If coverage report shows:
- ⚠️ Few pages found
- ❌ Most type-specific markers missing
- ❌ Few citation map entries

**Next Step:** Investigate conversion issues. May need to improve Phase 1 or implement Phase 2 reconstruction.

---

## Troubleshooting

### Conversion Fails

**Error:** `Docling conversion failed`
**Solution:**
1. Check if Docling is installed: `pip install docling`
2. Try updating: `pip install --upgrade docling`
3. Check if file is corrupted: open manually
4. Try fallback converter: rename to .txt and use textract

### No Citations Found

**Error:** `No page markers found`
**Solution:**
1. Open output markdown file manually
2. Check if page markers exist (search for "Page", "p.")
3. If markers exist but not found, improve regex in `docling_converter.py`
4. If no markers in source, document may need OCR or different handling

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'openpyxl'`
**Solution:**
```bash
pip install openpyxl extract-msg python-pptx textract
```

### Timeout Errors

**Error:** `Conversion timeout (>5 minutes)`
**Solution:**
1. Document may be very large (>200 pages)
2. Increase timeout in `docling_converter.py` (line 73)
3. Consider splitting document into smaller files

---

## Next Steps After Testing

### If Coverage is Good (>=80%)

1. ✅ Skip Phase 2 (reconstruction not needed)
2. ✅ Proceed to Phase 3: Chunking & Context Cards
3. ✅ Start implementing `chunk_documents.py`

### If Coverage is Medium (50-79%)

1. ⚠️ Iterate on Phase 1 to improve extraction
2. ⚠️ Adjust regex patterns in converters
3. ⚠️ Re-test after improvements
4. ⚠️ If still not improved, implement Phase 2 reconstruction

### If Coverage is Low (<50%)

1. ❌ Investigate root cause (OCR needed? Format issue?)
2. ❌ Implement Phase 2: Citation Reconstruction Script
3. ❌ Re-test after reconstruction

---

## Testing Checklist

Before proceeding to Phase 3:

**Basic Functionality:**
- [ ] Can convert PDF to markdown
- [ ] Can convert DOCX to markdown
- [ ] Can convert Excel to markdown
- [ ] Can convert email to markdown
- [ ] Post-processing creates citations JSON
- [ ] Coverage report generates successfully

**Citation Extraction:**
- [ ] Extracts page markers
- [ ] Extracts Bates stamps (if present)
- [ ] Extracts line numbers (for depositions)
- [ ] Extracts column markers (for patents)
- [ ] Extracts paragraph markers (for expert reports)

**Quality:**
- [ ] No garbage text from images in output
- [ ] Citation markers are accurate
- [ ] Markdown is readable
- [ ] Citations JSON is valid JSON
- [ ] Coverage report is accurate

**Decision:**
- [ ] Determined if Phase 2 is needed
- [ ] Documented gaps (if any)
- [ ] Clear path forward to Phase 3

---

## Getting Help

1. **Check Documentation:**
   - `README.md` - Project overview
   - `PHASE1_COMPLETE.md` - Phase 1 details
   - `IMPLEMENTATION_STATUS.md` - Overall status
   - `LITIGATION_DOCUMENT_PIPELINE_TRD.md` - Full technical spec

2. **Review Code:**
   - `docling_converter.py` - Conversion logic
   - `post_processor.py` - Cleaning logic
   - `citation_types.py` - Data structures

3. **Check Logs:**
   - Conversion errors appear in stdout
   - Check `test_output/` for intermediate files

---

## Summary

**Phase 1 is complete and ready for testing.**

Run this to get started:
```bash
python tests/test_phase1_citations.py /path/to/your/document.pdf
```

Review the output and decide:
- ✅ Good coverage → Proceed to Phase 3
- ⚠️ Medium coverage → Iterate on Phase 1 or implement Phase 2
- ❌ Low coverage → Fix Phase 1 or implement Phase 2

**Questions?** See `IMPLEMENTATION_STATUS.md` for full details.

# Error Handling & Recovery Guide

## Overview

The pipeline now includes comprehensive error handling and recovery features that make it robust for production use:

1. **Checkpoint/Resume** - Resume processing from where you left off
2. **Graceful Failure Handling** - Continue processing other documents when one fails
3. **Configurable Timeouts** - Adjust timeouts for large documents
4. **Retry Management** - Automatic retry limiting for problematic documents
5. **State Tracking** - Detailed tracking of processing progress

## Quick Start

### Basic Usage

```bash
# Process documents with automatic checkpointing
lit-pipeline process tests/test_docs output/
```

If the pipeline is interrupted, simply run it again with `--resume`:

```bash
# Resume from last checkpoint
lit-pipeline process tests/test_docs output/ --resume
```

### Advanced Usage

```bash
# Force reprocess all documents (ignore checkpoints)
lit-pipeline process tests/test_docs output/ --force

# Increase timeout for large scanned documents
lit-pipeline process docs/ output/ --conversion-timeout 600

# Retry documents that previously failed
lit-pipeline process docs/ output/ --resume --no-skip-failed
```

## Features in Detail

### 1. State Tracking (`.lit-pipeline-state.json`)

The pipeline automatically creates a state file in your output directory:

```json
{
  "version": "1.0",
  "started_at": "2026-02-12T10:30:00",
  "last_updated": "2026-02-12T10:35:00",
  "documents": {
    "daniel_alexander_10_24_2025": {
      "filename": "daniel_alexander_10_24_2025.pdf",
      "stem": "daniel_alexander_10_24_2025",
      "stages_completed": ["conversion", "post_processing", "citation_tracking"],
      "status": "completed",
      "last_updated": "2026-02-12T10:32:00",
      "error": null,
      "retry_count": 0
    }
  }
}
```

**Benefits:**
- Tracks which stages each document has completed
- Enables resume from interruptions
- Records error history for debugging
- Prevents unnecessary reprocessing

### 2. Pipeline Stages

The pipeline tracks progress through 4 main stages:

1. **conversion** - PDF → Markdown + JSON (via Docling)
2. **post_processing** - Text cleaning, footnote insertion
3. **citation_tracking** - Bbox-based citation reconstruction
4. **enrichment** - Optional LLM enrichment

Each stage is independently tracked and can be resumed.

### 3. Error Handling

**Try-Catch Blocks Around Each Stage:**
```python
# Each stage wrapped in try-catch
try:
    result = converter.convert_document(...)
    doc_state.mark_stage_complete("conversion")
except Exception as e:
    doc_state.mark_failed(f"Conversion error: {e}")
    # Continue with next document
```

**Benefits:**
- Pipeline continues even if one document fails
- Partial progress is saved
- Detailed error messages for debugging
- Automatic retry limiting (max 3 attempts)

### 4. Timeout Handling

**Configurable Timeouts:**
```bash
# Default: 300 seconds (5 minutes)
lit-pipeline process docs/ output/

# Increase for large documents: 600 seconds (10 minutes)
lit-pipeline process large_docs/ output/ --conversion-timeout 600
```

**How it works:**
- Docling conversion respects timeout limit
- On timeout, error is logged and pipeline continues
- Document marked as failed (can retry with longer timeout)

### 5. Resume Logic

**Automatic Skipping:**
- `--resume`: Skip documents that are already completed
- Without `--resume`: Process all documents (but still track progress)

**Example workflow:**
```bash
# First run - processes all documents
lit-pipeline process docs/ output/
# (interrupted after processing 5/10 documents)

# Resume - processes remaining 5 documents
lit-pipeline process docs/ output/ --resume

# View state summary
cat output/.lit-pipeline-state.json | jq '.documents | to_entries | group_by(.value.status)'
```

### 6. Force Reprocessing

**Use `--force` to override checkpoints:**
```bash
# Reprocess everything from scratch
lit-pipeline process docs/ output/ --force

# Reprocess with different settings
lit-pipeline process docs/ output/ --force --conversion-timeout 600
```

**When to use:**
- Pipeline code was updated
- You want to regenerate with different settings
- Previous run had partial failures

### 7. Retry Management

**Automatic Retry Limiting:**
- Documents that fail 3+ times are automatically skipped
- Prevents infinite retry loops
- Can override with `--no-skip-failed`

**Example:**
```bash
# First run - document X fails
lit-pipeline process docs/ output/

# Second run - document X fails again (retry_count=2)
lit-pipeline process docs/ output/ --resume

# Third run - document X fails again (retry_count=3)
lit-pipeline process docs/ output/ --resume

# Fourth run - document X automatically skipped
lit-pipeline process docs/ output/ --resume

# Force retry anyway
lit-pipeline process docs/ output/ --resume --no-skip-failed
```

## Command-Line Reference

### Process Command Options

```bash
lit-pipeline process <input-dir> <output-dir> [options]

Error Handling Options:
  --resume                  Resume from previous run (skip completed documents)
  --force                   Force reprocessing (ignore checkpoints)
  --no-skip-failed          Retry documents that have failed multiple times
  --conversion-timeout SEC  Timeout in seconds for document conversion (default: 300)

Standard Options:
  --use-existing DIR        Use existing converted files (skip Docling)
  --cleanup-json            Delete Docling JSON after processing (default: True)
  --enrich                  Run LLM enrichment after processing
  --enrich-backend BACKEND  LLM backend: ollama or anthropic (default: ollama)
  --case-type TYPE          Case type for enrichment (default: patent)
  --parties NAMES           Comma-separated party names for enrichment
```

## Common Workflows

### 1. Initial Processing

```bash
# Process all documents with default settings
lit-pipeline process tests/test_docs output/
```

**What happens:**
- All PDFs are processed through 3 stages
- State file is created: `output/.lit-pipeline-state.json`
- If interrupted, progress is saved

### 2. Resume After Interruption

```bash
# Resume from checkpoint
lit-pipeline process tests/test_docs output/ --resume
```

**What happens:**
- State file is loaded
- Completed documents are skipped
- Processing continues from where it left off
- Failed documents are retried (up to 3 times)

### 3. Handling Large Documents

```bash
# Increase timeout for large scanned documents
lit-pipeline process large_docs/ output/ --conversion-timeout 600
```

**What happens:**
- Docling timeout increased to 10 minutes
- Large documents have more time to process
- Smaller documents still process quickly

### 4. Fixing Failed Documents

```bash
# View failed documents
cat output/.lit-pipeline-state.json | jq '.documents[] | select(.status=="failed")'

# Retry with increased timeout
lit-pipeline process docs/ output/ --resume --conversion-timeout 600 --no-skip-failed
```

### 5. Clean Slate Reprocessing

```bash
# Delete state file and reprocess from scratch
rm output/.lit-pipeline-state.json
lit-pipeline process docs/ output/

# Or use --force (keeps history)
lit-pipeline process docs/ output/ --force
```

## Monitoring Progress

### View State Summary

The pipeline automatically prints a summary at the end:

```
Pipeline State Summary:
  Total documents: 10
  Completed: 8
  In progress: 1
  Failed: 1
  Started: 2026-02-12T10:30:00
  Last updated: 2026-02-12T11:00:00

Failed documents:
  - large_doc.pdf: Conversion timeout (>300 seconds) (retries: 2)
```

### Check State File Manually

```bash
# Pretty-print state file
cat output/.lit-pipeline-state.json | jq '.'

# Count documents by status
cat output/.lit-pipeline-state.json | jq '.documents | group_by(.status) | map({status: .[0].status, count: length})'

# List failed documents
cat output/.lit-pipeline-state.json | jq '.documents[] | select(.status=="failed") | .filename'

# Check specific document
cat output/.lit-pipeline-state.json | jq '.documents["daniel_alexander_10_24_2025"]'
```

## Troubleshooting

### Problem: Document times out

**Solution:**
```bash
# Increase timeout
lit-pipeline process docs/ output/ --resume --conversion-timeout 600
```

### Problem: Document fails repeatedly

**Diagnosis:**
```bash
# Check error message
cat output/.lit-pipeline-state.json | jq '.documents["problematic_doc"] | .error'
```

**Solutions:**
- Check if PDF is corrupted: `pdfinfo doc.pdf`
- Try with longer timeout: `--conversion-timeout 900`
- Check disk space: `df -h output/`
- Check Docling logs for details

### Problem: Want to reprocess with new settings

**Solution:**
```bash
# Force reprocess all
lit-pipeline process docs/ output/ --force

# Or delete state and start fresh
rm output/.lit-pipeline-state.json
lit-pipeline process docs/ output/
```

### Problem: Pipeline keeps retrying failed document

**Solution:**
```bash
# Skip documents that have failed 3+ times (default behavior)
lit-pipeline process docs/ output/ --resume

# Or manually edit state file to increase retry_count to 3+
```

## Implementation Details

### State File Location

- **File:** `<output-dir>/.lit-pipeline-state.json`
- **Format:** JSON with document metadata
- **Atomic writes:** Uses temp file + rename for safety
- **Automatic backups:** None (state can be regenerated)

### Stage Completion Tracking

Each document tracks completed stages as a list:
```json
"stages_completed": ["conversion", "post_processing", "citation_tracking"]
```

Stages are added sequentially as they complete.

### Error Recovery

When a stage fails:
1. Error message is logged
2. Document status set to "failed"
3. Retry count is incremented
4. State is saved to disk
5. Pipeline continues with next document

### Retry Logic

```python
def should_process_document(stem, stage, force):
    if force:
        return True  # Always process
    if stage in completed_stages:
        return False  # Skip completed stage
    if retry_count >= 3:
        return False  # Skip after 3 failures
    return True  # Process otherwise
```

## Testing

Run the test suite:

```bash
# Test error handling features
.venv/bin/python test_error_handling.py

# Test full pipeline
.venv/bin/python -m pytest tests/test_citation_tracker.py -v
```

## Performance Impact

**Overhead:**
- State file I/O: <1ms per document
- State tracking logic: <1ms per stage
- Negligible impact on total processing time

**Benefits:**
- Saves hours on interrupted large batch jobs
- Prevents reprocessing of 100+ page documents
- Enables safe experimentation (can always resume)

## Future Enhancements

Potential additions:
- Progress bars with tqdm
- Email notifications on failure
- Slack webhook integration
- Detailed performance metrics
- Parallel processing with shared state
- Incremental indexing based on state

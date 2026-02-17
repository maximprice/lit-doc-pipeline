# Search Instructions for the Litigation Corpus

This file explains how to run BM25 searches against the Proxense v. Intel document corpus from two different environments.

## Environment 1: Claude Code (macOS terminal)

Claude Code runs directly on the Mac. The `.venv` works here.

### Basic Search

```bash
cd ~/path/to/lit-doc-pipeline
.venv/bin/python lit_pipeline.py search output/kindler_fuja/ "<QUERY>" \
  --mode bm25 --top-k 10
```

### Save Results to File (for Cowork to read)

```bash
cd ~/path/to/lit-doc-pipeline

echo "=== SEARCH 1: <topic> ===" > output/kindler_fuja/search_results.txt

.venv/bin/python lit_pipeline.py search output/kindler_fuja/ "<query1>" \
  --mode bm25 --top-k 10 2>/dev/null >> output/kindler_fuja/search_results.txt

echo -e "\n\n=== SEARCH 2: <topic> ===" >> output/kindler_fuja/search_results.txt

.venv/bin/python lit_pipeline.py search output/kindler_fuja/ "<query2>" \
  --mode bm25 --top-k 10 2>/dev/null >> output/kindler_fuja/search_results.txt
```

The `2>/dev/null` suppresses progress bars and warnings so only clean results go to the file. Cowork can then read `output/kindler_fuja/search_results.txt` directly.

### Hybrid Search (if Ollama is running)

```bash
.venv/bin/python lit_pipeline.py search output/kindler_fuja/ "<QUERY>" \
  --mode hybrid --rerank --top-k 10
```

Hybrid mode combines BM25 + vector similarity. Add `--rerank` for cross-encoder reranking (slower but more precise).

---

## Environment 2: Cowork (Linux sandbox)

Cowork runs in a Linux VM. The `.venv` is broken here (symlinks point to `/opt/homebrew/...` which doesn't exist in Linux). Use system Python instead.

### Basic Search

```bash
cd /sessions/busy-clever-curie/mnt/lit-doc-pipeline && \
python3 lit_pipeline.py search output/kindler_fuja/ "<QUERY>" \
  --mode bm25 --top-k 10
```

### Save Results to File

```bash
cd /sessions/busy-clever-curie/mnt/lit-doc-pipeline && \
python3 lit_pipeline.py search output/kindler_fuja/ "<QUERY>" \
  --mode bm25 --top-k 10 2>/dev/null > output/kindler_fuja/search_results.txt
```

### Multiple Searches to One File

```bash
cd /sessions/busy-clever-curie/mnt/lit-doc-pipeline

echo "=== SEARCH 1: <topic> ===" > output/kindler_fuja/search_results.txt

python3 lit_pipeline.py search output/kindler_fuja/ "<query1>" \
  --mode bm25 --top-k 10 2>/dev/null >> output/kindler_fuja/search_results.txt

echo -e "\n\n=== SEARCH 2: <topic> ===" >> output/kindler_fuja/search_results.txt

python3 lit_pipeline.py search output/kindler_fuja/ "<query2>" \
  --mode bm25 --top-k 10 2>/dev/null >> output/kindler_fuja/search_results.txt
```

### Cowork-Specific Notes

- **Use `python3`** (system Python at `/usr/bin/python3`), NOT `.venv/bin/python`
- **Working directory must be** the pipeline root (imports are relative)
- **Only `bm25` mode works** — Ollama is not available in the sandbox
- **Sklearn version warning** is harmless — index was built with 1.8.0, system has 1.7.2, results are correct

---

## Shared Reference

### Key Details

- **Corpus:** 453 documents, 16,082 chunks in `output/kindler_fuja/`
- **`--top-k N`**: Number of results (default 10, max ~50 useful)
- **`--mode bm25`**: Keyword search, always works, <1ms per query
- **`--mode hybrid`**: BM25 + vector, requires Ollama with nomic-embed-text
- **`--rerank`**: Cross-encoder reranking, adds ~19s latency, improves precision

### What Results Look Like

Each result includes:
- **Document name** (the source file stem)
- **Document type** (deposition, patent, expert_report, etc.)
- **Citation string** (with page numbers, Bates stamps, line numbers)
- **BM25 score** (1.0 = best match, relative to query)
- **Preview** (~500 chars of matching chunk text)

### Search Tips

- Use **specific legal terms**: "claim construction plain and ordinary meaning"
- Use **names**: "Fuja timer schedule"
- Use **Bates prefixes**: "INTEL_PROX_00557"
- Use **document identifiers**: "post-mortem TWT ChromeOS"
- BM25 is keyword-based — more specific terms get better results than long natural language queries
- Run **multiple narrow searches** rather than one broad one

### Other Useful Commands

```bash
# Corpus statistics (use .venv/bin/python on Mac, python3 in Cowork)
python3 lit_pipeline.py stats output/kindler_fuja/

# View all indexed document names
ls output/kindler_fuja/converted/*_chunks.json | sed 's/.*converted\///' | sed 's/_chunks.json//' | sort
```

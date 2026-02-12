"""
Pipeline state management for checkpoint/resume functionality.

Tracks which documents have been processed through each pipeline stage,
enabling recovery from failures and incremental processing.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class DocumentState:
    """State tracking for a single document."""
    filename: str
    stem: str
    stages_completed: List[str]  # ["conversion", "post_processing", "citation_tracking", "enrichment"]
    status: str  # "in_progress", "completed", "failed", "skipped"
    last_updated: str
    error: Optional[str] = None
    retry_count: int = 0

    def is_stage_complete(self, stage: str) -> bool:
        """Check if a specific stage is complete."""
        return stage in self.stages_completed

    def mark_stage_complete(self, stage: str):
        """Mark a stage as complete."""
        if stage not in self.stages_completed:
            self.stages_completed.append(stage)
        self.last_updated = datetime.now().isoformat()

    def mark_failed(self, error: str):
        """Mark document as failed."""
        self.status = "failed"
        self.error = error
        self.retry_count += 1
        self.last_updated = datetime.now().isoformat()

    def mark_completed(self):
        """Mark document as fully completed."""
        self.status = "completed"
        self.last_updated = datetime.now().isoformat()


class PipelineState:
    """
    Manages pipeline state for checkpoint/resume functionality.

    State file format:
    {
        "version": "1.0",
        "started_at": "2026-02-12T10:30:00",
        "last_updated": "2026-02-12T10:35:00",
        "documents": {
            "daniel_alexander_10_24_2025": {
                "filename": "daniel_alexander_10_24_2025.pdf",
                "stem": "daniel_alexander_10_24_2025",
                "stages_completed": ["conversion", "post_processing"],
                "status": "in_progress",
                "last_updated": "2026-02-12T10:32:00",
                "error": null,
                "retry_count": 0
            }
        }
    }
    """

    STATE_FILE = ".lit-pipeline-state.json"
    VERSION = "1.0"

    def __init__(self, output_dir: Path):
        """
        Initialize pipeline state.

        Args:
            output_dir: Directory where state file is stored
        """
        self.output_dir = output_dir
        self.state_file = output_dir / self.STATE_FILE
        self.documents: Dict[str, DocumentState] = {}
        self.started_at: Optional[str] = None
        self.last_updated: Optional[str] = None

        # Load existing state if present
        if self.state_file.exists():
            self.load()
        else:
            self.started_at = datetime.now().isoformat()
            self.last_updated = self.started_at

    def load(self):
        """Load state from file."""
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)

            self.started_at = data.get("started_at")
            self.last_updated = data.get("last_updated")

            # Reconstruct DocumentState objects
            self.documents = {}
            for stem, doc_data in data.get("documents", {}).items():
                self.documents[stem] = DocumentState(**doc_data)

            logger.info("Loaded pipeline state from %s (%d documents)",
                       self.state_file, len(self.documents))

        except Exception as e:
            logger.warning("Failed to load state file: %s", e)
            self.started_at = datetime.now().isoformat()
            self.last_updated = self.started_at

    def save(self):
        """Save state to file."""
        self.last_updated = datetime.now().isoformat()

        data = {
            "version": self.VERSION,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
            "documents": {
                stem: asdict(doc) for stem, doc in self.documents.items()
            }
        }

        try:
            # Write atomically (write to temp, then rename)
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            temp_file.replace(self.state_file)

            logger.debug("Saved pipeline state to %s", self.state_file)

        except Exception as e:
            logger.error("Failed to save state file: %s", e)

    def get_document(self, stem: str, filename: str) -> DocumentState:
        """
        Get or create document state.

        Args:
            stem: Normalized document stem
            filename: Original filename

        Returns:
            DocumentState for this document
        """
        if stem not in self.documents:
            self.documents[stem] = DocumentState(
                filename=filename,
                stem=stem,
                stages_completed=[],
                status="in_progress",
                last_updated=datetime.now().isoformat(),
            )
        return self.documents[stem]

    def get_incomplete_documents(self) -> List[DocumentState]:
        """Get documents that are not yet completed."""
        return [
            doc for doc in self.documents.values()
            if doc.status != "completed"
        ]

    def get_failed_documents(self) -> List[DocumentState]:
        """Get documents that have failed."""
        return [
            doc for doc in self.documents.values()
            if doc.status == "failed"
        ]

    def should_process_document(self, stem: str, stage: str, force: bool = False) -> bool:
        """
        Check if document/stage should be processed.

        Args:
            stem: Document stem
            stage: Stage name
            force: Force reprocessing even if complete

        Returns:
            True if should process, False to skip
        """
        if force:
            return True

        if stem not in self.documents:
            return True

        doc = self.documents[stem]

        # Skip if stage already complete
        if doc.is_stage_complete(stage):
            return False

        # Skip if document failed too many times
        if doc.status == "failed" and doc.retry_count >= 3:
            logger.info("Skipping %s (failed %d times)", doc.filename, doc.retry_count)
            return False

        return True

    def summary(self) -> str:
        """Generate summary of pipeline state."""
        total = len(self.documents)
        if total == 0:
            return "No documents processed yet"

        completed = sum(1 for d in self.documents.values() if d.status == "completed")
        failed = sum(1 for d in self.documents.values() if d.status == "failed")
        in_progress = sum(1 for d in self.documents.values() if d.status == "in_progress")

        lines = [
            f"Pipeline State Summary:",
            f"  Total documents: {total}",
            f"  Completed: {completed}",
            f"  In progress: {in_progress}",
            f"  Failed: {failed}",
            f"  Started: {self.started_at}",
            f"  Last updated: {self.last_updated}",
        ]

        if failed > 0:
            lines.append("\nFailed documents:")
            for doc in self.get_failed_documents():
                lines.append(f"  - {doc.filename}: {doc.error or 'Unknown error'} "
                           f"(retries: {doc.retry_count})")

        return "\n".join(lines)

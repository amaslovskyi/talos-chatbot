"""
Auto-indexing module for detecting and indexing new documents.
Provides functionality to automatically index new documents when detected.
"""

import os
import logging
import hashlib
from typing import List, Dict, Set
from pathlib import Path

from src.document_loader import load_and_chunk_documents
from src.vector_store import create_vector_store
from config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoIndexer:
    """
    Automatically detects and indexes new documents.
    Tracks which documents have been indexed to avoid duplicates.
    """

    def __init__(self):
        """Initialize the auto-indexer."""
        self.settings = get_settings()
        self.vector_store = create_vector_store()
        self.indexed_files_path = (
            Path(self.settings.vector_db_path) / "indexed_files.txt"
        )

    def get_indexed_files(self) -> Set[str]:
        """Get list of already indexed files."""
        if not self.indexed_files_path.exists():
            return set()

        try:
            with open(self.indexed_files_path, "r") as f:
                return set(line.strip() for line in f if line.strip())
        except Exception as e:
            logger.warning(f"Error reading indexed files list: {e}")
            return set()

    def save_indexed_files(self, indexed_files: Set[str]):
        """Save list of indexed files."""
        try:
            os.makedirs(self.indexed_files_path.parent, exist_ok=True)
            with open(self.indexed_files_path, "w") as f:
                for file_path in sorted(indexed_files):
                    f.write(f"{file_path}\n")
        except Exception as e:
            logger.error(f"Error saving indexed files list: {e}")

    def get_file_hash(self, file_path: Path) -> str:
        """Get hash of file for change detection."""
        try:
            with open(file_path, "rb") as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        except Exception:
            return ""

    def scan_for_documents(self) -> List[Path]:
        """Scan documents directory for supported files."""
        docs_dir = Path(self.settings.docs_directory)
        if not docs_dir.exists():
            return []

        supported_extensions = {".pdf", ".docx", ".txt", ".md", ".html"}
        documents = []

        for file_path in docs_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                documents.append(file_path)

        return documents

    def detect_new_documents(self) -> List[Path]:
        """Detect new or modified documents that need indexing."""
        all_documents = self.scan_for_documents()
        indexed_files = self.get_indexed_files()
        new_documents = []

        for doc_path in all_documents:
            # Create file identifier with path and hash
            file_hash = self.get_file_hash(doc_path)
            file_id = f"{doc_path}:{file_hash}"

            if file_id not in indexed_files:
                new_documents.append(doc_path)
                logger.info(f"📄 New/modified document detected: {doc_path.name}")

        return new_documents

    def detect_deleted_documents(self) -> List[Path]:
        """Detect documents that were indexed but no longer exist."""
        indexed_files = self.get_indexed_files()
        existing_files = {str(doc) for doc in self.scan_for_documents()}
        deleted_files = []

        for file_record in indexed_files:
            # Extract file path from record (format: "path:hash")
            if ":" in file_record:
                file_path = file_record.split(":")[0]
                if file_path not in existing_files:
                    deleted_files.append(Path(file_path))

        return deleted_files

    def sync_documents(self, force_reindex: bool = False) -> Dict[str, any]:
        """
        Synchronize documents - add new ones and remove deleted ones.

        Args:
            force_reindex: If True, completely rebuild the index

        Returns:
            Dictionary with sync results
        """
        results = {
            "new_documents": 0,
            "deleted_documents": 0,
            "total_chunks": 0,
            "errors": [],
            "indexed_files": [],
            "deleted_files": [],
        }

        try:
            if force_reindex:
                logger.info(
                    "🔄 Force reindexing - clearing and rebuilding entire database..."
                )
                self.vector_store.clear_collection()
                self.save_indexed_files(set())
                new_documents = self.scan_for_documents()
                deleted_files = []
            else:
                # Find new documents
                new_documents = self.detect_new_documents()
                # Find deleted documents
                deleted_files = self.detect_deleted_documents()

            # Handle deletions
            if deleted_files:
                logger.info(
                    f"🗑️ Removing {len(deleted_files)} deleted documents from index"
                )
                indexed_files = self.get_indexed_files()
                for deleted_file in deleted_files:
                    # Remove from indexed files list
                    indexed_files = {
                        f for f in indexed_files if not f.startswith(str(deleted_file))
                    }
                    results["deleted_files"].append(str(deleted_file))

                self.save_indexed_files(indexed_files)
                results["deleted_documents"] = len(deleted_files)

                # For deletions, we need to rebuild the index since ChromaDB doesn't easily support deletion by source
                if deleted_files and not force_reindex:
                    logger.info("🔄 Rebuilding index due to deletions...")
                    return self.sync_documents(force_reindex=True)

            # Handle new documents
            if new_documents:
                logger.info(f"📚 Found {len(new_documents)} new documents to index")

                # Load and chunk documents
                documents = load_and_chunk_documents(
                    [str(doc) for doc in new_documents]
                )

                if documents:
                    # Add to vector store
                    self.vector_store.add_documents(documents)

                    # Update indexed files list
                    indexed_files = self.get_indexed_files()
                    for doc_path in new_documents:
                        file_hash = self.get_file_hash(doc_path)
                        file_id = f"{doc_path}:{file_hash}"
                        indexed_files.add(file_id)
                        results["indexed_files"].append(str(doc_path))

                    self.save_indexed_files(indexed_files)
                    results["new_documents"] = len(new_documents)
                    results["total_chunks"] = len(documents)

                    logger.info(
                        f"✅ Successfully indexed {len(new_documents)} documents ({len(documents)} chunks)"
                    )
                else:
                    logger.warning("⚠️ No content extracted from documents")

            if not new_documents and not deleted_files:
                logger.info("✅ No changes detected - everything is up to date")

        except Exception as e:
            error_msg = f"Error during document sync: {str(e)}"
            logger.error(error_msg)
            results["errors"].append(error_msg)

        return results

    def index_new_documents(self, force_reindex: bool = False) -> Dict[str, any]:
        """
        Legacy method - calls sync_documents for backward compatibility.

        Args:
            force_reindex: If True, reindex all documents

        Returns:
            Dictionary with indexing results
        """
        return self.sync_documents(force_reindex)

    def get_indexing_status(self) -> Dict[str, any]:
        """Get current indexing status."""
        all_docs = self.scan_for_documents()
        new_docs = self.detect_new_documents()
        deleted_docs = self.detect_deleted_documents()
        indexed_files = self.get_indexed_files()

        stats = self.vector_store.get_collection_stats()

        return {
            "total_files_in_directory": len(all_docs),
            "new_files_detected": len(new_docs),
            "deleted_files_detected": len(deleted_docs),
            "indexed_file_records": len(indexed_files),
            "documents_in_database": stats["document_count"],
            "needs_indexing": len(new_docs) > 0 or len(deleted_docs) > 0,
            "new_files": [str(doc.name) for doc in new_docs],
            "deleted_files": [str(doc.name) for doc in deleted_docs],
        }


def create_auto_indexer() -> AutoIndexer:
    """Factory function to create an auto-indexer."""
    return AutoIndexer()

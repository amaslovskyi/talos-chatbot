"""
File system watcher for automatic document indexing.
Monitors the documents directory for file changes and triggers re-indexing.

MIT License - Copyright (c) 2025 talos-chatbot
"""

import logging
import time
import threading
from pathlib import Path
from typing import Callable, Set

try:
    from watchdog.observers import Observer
    from watchdog.events import (
        FileSystemEventHandler,
        FileModifiedEvent,
        FileCreatedEvent,
        FileDeletedEvent,
    )

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

    # Create dummy classes for when watchdog is not available
    class FileSystemEventHandler:
        def __init__(self):
            pass

    class FileModifiedEvent:
        pass

    class FileCreatedEvent:
        pass

    class FileDeletedEvent:
        pass


from config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentChangeHandler(FileSystemEventHandler):
    """Handles file system events for document changes."""

    def __init__(self, on_change_callback: Callable):
        """
        Initialize the handler.

        Args:
            on_change_callback: Function to call when documents change
        """
        super().__init__()
        self.on_change_callback = on_change_callback
        self.supported_extensions = {".pdf", ".docx", ".txt", ".md", ".html"}
        self.last_event_time = 0
        self.debounce_seconds = (
            2  # Wait 2 seconds before processing to avoid duplicate events
        )

    def is_supported_file(self, file_path: str) -> bool:
        """Check if file has a supported extension."""
        return Path(file_path).suffix.lower() in self.supported_extensions

    def should_process_event(self) -> bool:
        """Debounce events to avoid processing duplicates."""
        current_time = time.time()
        if current_time - self.last_event_time < self.debounce_seconds:
            return False
        self.last_event_time = current_time
        return True

    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory and self.is_supported_file(event.src_path):
            if self.should_process_event():
                logger.info(f"📄 New document detected: {Path(event.src_path).name}")
                self._trigger_reindex("Document added")

    def on_deleted(self, event):
        """Handle file deletion events."""
        if not event.is_directory and self.is_supported_file(event.src_path):
            if self.should_process_event():
                logger.info(f"🗑️ Document deleted: {Path(event.src_path).name}")
                self._trigger_reindex("Document deleted")

    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory and self.is_supported_file(event.src_path):
            if self.should_process_event():
                logger.info(f"✏️ Document modified: {Path(event.src_path).name}")
                self._trigger_reindex("Document modified")

    def _trigger_reindex(self, reason: str):
        """Trigger reindexing in a separate thread."""

        def reindex():
            try:
                logger.info(f"🔄 Triggering reindex: {reason}")
                self.on_change_callback()
            except Exception as e:
                logger.error(f"Error during auto-reindex: {str(e)}")

        # Run in separate thread to avoid blocking the file watcher
        threading.Thread(target=reindex, daemon=True).start()


class DocumentWatcher:
    """Watches the documents directory for changes and triggers re-indexing."""

    def __init__(self, on_change_callback: Callable):
        """
        Initialize the document watcher.

        Args:
            on_change_callback: Function to call when documents change
        """
        self.settings = get_settings()
        self.on_change_callback = on_change_callback
        self._is_watching = False

        if not WATCHDOG_AVAILABLE:
            logger.warning(
                "⚠️ Watchdog not available. Install with: pip install watchdog"
            )
            self.observer = None
            return

        self.observer = Observer()
        self.handler = DocumentChangeHandler(on_change_callback)

    def start_watching(self) -> bool:
        """
        Start watching the documents directory.

        Returns:
            True if watching started successfully, False otherwise
        """
        if not WATCHDOG_AVAILABLE:
            logger.warning("📁 File watching not available - watchdog not installed")
            return False

        if self._is_watching:
            logger.info("👀 File watcher already running")
            return True

        docs_path = Path(self.settings.docs_directory)
        if not docs_path.exists():
            logger.warning(f"📁 Documents directory not found: {docs_path}")
            return False

        try:
            self.observer.schedule(self.handler, str(docs_path), recursive=True)
            self.observer.start()
            self._is_watching = True
            logger.info(f"👀 Started watching documents directory: {docs_path}")
            return True
        except Exception as e:
            logger.error(f"Error starting file watcher: {str(e)}")
            return False

    def stop_watching(self):
        """Stop watching the documents directory."""
        if self.observer and self._is_watching:
            self.observer.stop()
            self.observer.join()
            self._is_watching = False
            logger.info("🛑 Stopped watching documents directory")

    def is_watching(self) -> bool:
        """Check if currently watching."""
        return self._is_watching

    def get_status(self) -> dict:
        """Get watcher status."""
        return {
            "available": WATCHDOG_AVAILABLE,
            "watching": self._is_watching,
            "docs_directory": self.settings.docs_directory,
        }


def create_document_watcher(on_change_callback: Callable) -> DocumentWatcher:
    """
    Factory function to create a document watcher.

    Args:
        on_change_callback: Function to call when documents change

    Returns:
        DocumentWatcher instance
    """
    return DocumentWatcher(on_change_callback)

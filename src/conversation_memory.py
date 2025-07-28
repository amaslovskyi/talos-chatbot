"""
Conversation memory management for the RAG chatbot.
Handles storing and retrieving conversation history to provide context for follow-up questions.

MIT License - Copyright (c) 2025 talos-chatbot
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from uuid import uuid4

from config import get_settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConversationMessage:
    """Represents a single message in a conversation."""

    timestamp: str
    role: str  # 'user' or 'assistant'
    content: str
    sources: List[Dict[str, Any]] = None  # For assistant messages
    confidence: float = None  # For assistant messages
    metadata: Dict[str, Any] = None


@dataclass
class ConversationSession:
    """Represents a complete conversation session."""

    session_id: str
    created_at: str
    last_updated: str
    messages: List[ConversationMessage]
    metadata: Dict[str, Any] = None


class ConversationMemory:
    """
    Manages conversation sessions and provides context for follow-up questions.
    Supports both in-memory and persistent storage.
    """

    def __init__(
        self, persistent_storage: bool = True, max_session_age_hours: int = 24
    ):
        """
        Initialize conversation memory.

        Args:
            persistent_storage: Whether to persist conversations to disk
            max_session_age_hours: Maximum age of sessions before cleanup
        """
        self.settings = get_settings()
        self.persistent_storage = persistent_storage
        self.max_session_age_hours = max_session_age_hours

        # In-memory storage for active sessions
        self.active_sessions: Dict[str, ConversationSession] = {}

        # Set up persistent storage directory
        if self.persistent_storage:
            self.storage_dir = os.path.join(
                self.settings.vector_db_path, "conversations"
            )
            os.makedirs(self.storage_dir, exist_ok=True)
            logger.info(f"Conversation storage: {self.storage_dir}")

        # Load existing sessions
        self._load_sessions()

    def create_session(self) -> str:
        """
        Create a new conversation session.

        Returns:
            Session ID string
        """
        session_id = str(uuid4())
        timestamp = datetime.now().isoformat()

        session = ConversationSession(
            session_id=session_id,
            created_at=timestamp,
            last_updated=timestamp,
            messages=[],
            metadata={},
        )

        self.active_sessions[session_id] = session
        self._save_session(session)

        logger.info(f"Created new conversation session: {session_id}")
        return session_id

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add a message to a conversation session.

        Args:
            session_id: Session identifier
            role: 'user' or 'assistant'
            content: Message content
            sources: Source documents (for assistant messages)
            confidence: Response confidence (for assistant messages)
            metadata: Additional metadata

        Returns:
            True if message was added successfully
        """
        if session_id not in self.active_sessions:
            # Try to load session from storage
            if not self._load_session(session_id):
                logger.warning(f"Session not found: {session_id}")
                return False

        message = ConversationMessage(
            timestamp=datetime.now().isoformat(),
            role=role,
            content=content,
            sources=sources,
            confidence=confidence,
            metadata=metadata or {},
        )

        session = self.active_sessions[session_id]
        session.messages.append(message)
        session.last_updated = datetime.now().isoformat()

        # Save to persistent storage
        self._save_session(session)

        logger.debug(f"Added {role} message to session {session_id}")
        return True

    def get_conversation_context(
        self, session_id: str, max_messages: int = 10, include_sources: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get conversation context for a session.

        Args:
            session_id: Session identifier
            max_messages: Maximum number of recent messages to include
            include_sources: Whether to include source information

        Returns:
            List of message dictionaries for context
        """
        if session_id not in self.active_sessions:
            if not self._load_session(session_id):
                return []

        session = self.active_sessions[session_id]
        recent_messages = session.messages[-max_messages:]

        context = []
        for msg in recent_messages:
            msg_dict = {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
            }

            if include_sources and msg.sources:
                msg_dict["sources"] = msg.sources
            if msg.confidence is not None:
                msg_dict["confidence"] = msg.confidence

            context.append(msg_dict)

        return context

    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get summary information about a session.

        Args:
            session_id: Session identifier

        Returns:
            Session summary dictionary or None
        """
        if session_id not in self.active_sessions:
            if not self._load_session(session_id):
                return None

        session = self.active_sessions[session_id]

        return {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "last_updated": session.last_updated,
            "message_count": len(session.messages),
            "user_messages": len([m for m in session.messages if m.role == "user"]),
            "assistant_messages": len(
                [m for m in session.messages if m.role == "assistant"]
            ),
        }

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List recent conversation sessions.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session summary dictionaries
        """
        # Load all sessions first
        self._load_sessions()

        sessions = []
        for session in self.active_sessions.values():
            summary = self.get_session_summary(session.session_id)
            if summary:
                sessions.append(summary)

        # Sort by last updated time (most recent first)
        sessions.sort(key=lambda x: x["last_updated"], reverse=True)

        return sessions[:limit]

    def cleanup_old_sessions(self) -> int:
        """
        Remove sessions older than max_session_age_hours.

        Returns:
            Number of sessions cleaned up
        """
        cutoff_time = datetime.now() - timedelta(hours=self.max_session_age_hours)
        cutoff_iso = cutoff_time.isoformat()

        old_session_ids = []
        for session_id, session in self.active_sessions.items():
            if session.last_updated < cutoff_iso:
                old_session_ids.append(session_id)

        # Remove old sessions
        for session_id in old_session_ids:
            self._delete_session(session_id)

        logger.info(f"Cleaned up {len(old_session_ids)} old sessions")
        return len(old_session_ids)

    def _load_sessions(self):
        """Load all session files from persistent storage."""
        if not self.persistent_storage:
            return

        try:
            for filename in os.listdir(self.storage_dir):
                if filename.endswith(".json"):
                    session_id = filename[:-5]  # Remove .json extension
                    self._load_session(session_id)
        except FileNotFoundError:
            pass  # Storage directory doesn't exist yet
        except Exception as e:
            logger.error(f"Error loading sessions: {str(e)}")

    def _load_session(self, session_id: str) -> bool:
        """Load a specific session from persistent storage."""
        if not self.persistent_storage:
            return False

        try:
            filepath = os.path.join(self.storage_dir, f"{session_id}.json")
            if not os.path.exists(filepath):
                return False

            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Reconstruct session object
            messages = []
            for msg_data in data.get("messages", []):
                messages.append(ConversationMessage(**msg_data))

            session = ConversationSession(
                session_id=data["session_id"],
                created_at=data["created_at"],
                last_updated=data["last_updated"],
                messages=messages,
                metadata=data.get("metadata", {}),
            )

            self.active_sessions[session_id] = session
            return True

        except Exception as e:
            logger.error(f"Error loading session {session_id}: {str(e)}")
            return False

    def _save_session(self, session: ConversationSession):
        """Save a session to persistent storage."""
        if not self.persistent_storage:
            return

        try:
            filepath = os.path.join(self.storage_dir, f"{session.session_id}.json")

            # Convert to serializable format
            data = asdict(session)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Error saving session {session.session_id}: {str(e)}")

    def _delete_session(self, session_id: str):
        """Delete a session from memory and persistent storage."""
        # Remove from memory
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

        # Remove from storage
        if self.persistent_storage:
            try:
                filepath = os.path.join(self.storage_dir, f"{session_id}.json")
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as e:
                logger.error(f"Error deleting session file {session_id}: {str(e)}")


def create_conversation_memory() -> ConversationMemory:
    """Factory function to create a ConversationMemory instance."""
    return ConversationMemory()

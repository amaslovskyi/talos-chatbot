"""
Flask web interface for the RAG chatbot.
Provides a simple web UI for interacting with the chatbot.

MIT License - Copyright (c) 2025 talos-chatbot
"""

import os
import logging
from typing import Dict, Any
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS

from src.chatbot import create_chatbot, RAGChatbot
from src.hybrid_chatbot import create_hybrid_chatbot, HybridRAGChatbot
from src.auto_indexer import create_auto_indexer
from src.file_watcher import create_document_watcher
from config import get_settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app with static folder
app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)  # Enable CORS for API endpoints

# Global instances
chatbot: HybridRAGChatbot = None
document_watcher = None


def reindex_documents():
    """Reindex documents (called by file watcher)."""
    try:
        logger.info("🔄 Auto-reindexing triggered by file change")
        indexer = create_auto_indexer()
        results = indexer.sync_documents()

        if results["new_documents"] > 0 or results["deleted_documents"] > 0:
            logger.info(
                f"📄 Sync complete: +{results['new_documents']} -{results['deleted_documents']} documents"
            )

    except Exception as e:
        logger.error(f"Error during auto-reindex: {str(e)}")


def init_chatbot():
    """Initialize the chatbot instance."""
    global chatbot, document_watcher
    if chatbot is None:
        try:
            logger.info("Initializing chatbot...")

            # Auto-index new documents before starting chatbot
            try:
                logger.info("Checking for new documents to index...")
                indexer = create_auto_indexer()
                results = indexer.sync_documents()
                if results["new_documents"] > 0 or results["deleted_documents"] > 0:
                    logger.info(
                        f"Sync complete: +{results['new_documents']} new, -{results['deleted_documents']} deleted ({results['total_chunks']} chunks)"
                    )
                else:
                    logger.info("No changes detected - database is up to date")
            except Exception as e:
                logger.warning(f"Auto-indexing failed: {str(e)}")

            chatbot = create_hybrid_chatbot()

            # Start file watcher
            try:
                document_watcher = create_document_watcher(reindex_documents)
                if document_watcher.start_watching():
                    logger.info("👀 Document file watcher started")
                else:
                    logger.warning("⚠️ File watcher could not be started")
            except Exception as e:
                logger.warning(f"File watcher initialization failed: {str(e)}")

            # Check if documents directory exists
            settings = get_settings()
            if os.path.exists(settings.docs_directory):
                logger.info("Documents directory found - chatbot ready")
            else:
                logger.warning(
                    f"Documents directory not found: {settings.docs_directory}"
                )

        except Exception as e:
            logger.error(f"Error initializing chatbot: {str(e)}")
            raise


@app.route("/")
def index():
    """Serve the main chat interface."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
@app.route("/api/chat", methods=["POST"])  # Keep both for compatibility
def chat_api():
    """
    API endpoint for chatbot interactions.

    Expected JSON payload:
    {
        "message": "User's message",  # or "question" for compatibility
        "max_sources": 5  # Optional
    }
    """
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Support both 'message' and 'question' for compatibility
        question = data.get("message") or data.get("question")
        if not question:
            return jsonify({"error": "No message provided"}), 400

        question = question.strip()
        max_sources = data.get("max_sources", 5)

        if not question:
            return jsonify({"error": "Empty message"}), 400

        # Get chatbot response
        response = chatbot.chat(question, max_sources)

        # Format sources for JSON response
        sources_data = []
        for source in response.sources:
            source_data = {
                "title": source.title,
                "content": source.content[:300] + "..."
                if len(source.content) > 300
                else source.content,
                "source": source.source,
                "relevance_score": source.relevance_score,
                "url": source.url,
                "metadata": source.metadata,
            }
            sources_data.append(source_data)

        # Return response in the format expected by frontend
        return jsonify(
            {
                "response": response.answer,  # Frontend expects 'response', not 'answer'
                "sources": sources_data,
                "confidence": response.confidence,
                "retrieval_metadata": response.retrieval_metadata,
                "model_used": response.model_used,
            }
        )

    except Exception as e:
        logger.error(f"Error in chat API: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route("/api/stats")
def stats_api():
    """Get chatbot statistics and status."""
    try:
        stats = chatbot.get_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/reindex", methods=["POST"])
def reindex_api():
    """API endpoint to manually trigger document re-indexing."""
    try:
        data = request.get_json() or {}
        force = data.get("force", False)

        indexer = create_auto_indexer()
        results = indexer.sync_documents(force_reindex=force)

        return jsonify(
            {"success": True, "message": "Document sync completed", "results": results}
        )

    except Exception as e:
        logger.error(f"Error during manual reindex: {str(e)}")
        return jsonify({"success": False, "error": f"Reindex failed: {str(e)}"}), 500


@app.route("/index-status")
def index_status_api():
    """API endpoint to get indexing status."""
    try:
        indexer = create_auto_indexer()
        status = indexer.get_indexing_status()

        # Add file watcher status
        watcher_status = {"available": False, "watching": False}
        if document_watcher:
            watcher_status = document_watcher.get_status()

        return jsonify({"indexing": status, "file_watcher": watcher_status})

    except Exception as e:
        logger.error(f"Error getting index status: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/initialize", methods=["POST"])
def initialize_api():
    """Reinitialize the knowledge base."""
    try:
        data = request.get_json() or {}
        documents_directory = data.get("documents_directory")

        chatbot.initialize_knowledge_base(documents_directory)

        return jsonify({"message": "Knowledge base initialized successfully"})

    except Exception as e:
        logger.error(f"Error initializing knowledge base: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/clear", methods=["POST"])
def clear_api():
    """Clear the knowledge base."""
    try:
        chatbot.clear_knowledge_base()
        return jsonify({"message": "Knowledge base cleared successfully"})

    except Exception as e:
        logger.error(f"Error clearing knowledge base: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500


def create_app(config: Dict[str, Any] = None) -> Flask:
    """
    Factory function to create Flask app with configuration.

    Args:
        config: Optional configuration dictionary.

    Returns:
        Configured Flask app.
    """
    if config:
        app.config.update(config)

    # Initialize chatbot
    init_chatbot()

    return app


if __name__ == "__main__":
    # Get settings
    settings = get_settings()

    # Initialize chatbot
    init_chatbot()

    # Run Flask app
    app.run(
        host=settings.flask_host, port=settings.flask_port, debug=settings.flask_debug
    )

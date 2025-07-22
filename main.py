#!/usr/bin/env python3
"""
Main entry point for the RAG Chatbot application.
Provides command-line interface and web server startup.

MIT License - Copyright (c) 2025 talos-chatbot
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from src.chatbot import create_chatbot
from src.hybrid_chatbot import create_hybrid_chatbot
from src.document_loader import load_and_chunk_documents
from src.auto_indexer import create_auto_indexer
from config import get_settings, validate_settings

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_directories():
    """
    Create necessary directories if they don't exist.
    """
    settings = get_settings()

    # Create documents directory
    docs_path = Path(settings.docs_directory)
    docs_path.mkdir(exist_ok=True)
    logger.info(f"Documents directory: {docs_path.absolute()}")

    # Create vector database directory
    vector_db_path = Path(settings.vector_db_path)
    vector_db_path.mkdir(exist_ok=True)
    logger.info(f"Vector database directory: {vector_db_path.absolute()}")


def interactive_chat():
    """
    Run interactive command-line chat interface.
    """
    print("🤖 RAG Chatbot - Interactive Mode")
    print("Type 'quit', 'exit', or press Ctrl+C to stop")
    print("=" * 50)

    try:
        # Initialize hybrid chatbot with fallback support
        chatbot = create_hybrid_chatbot()

        # Initialize knowledge base
        chatbot.initialize_knowledge_base()

        # Get stats
        stats = chatbot.get_stats()
        vector_stats = stats["retrieval_engine"]["vector_store"]
        print(f"📊 Loaded {vector_stats['document_count']} documents")
        print("=" * 50)

        while True:
            try:
                # Get user input
                question = input("\n❓ Your question: ").strip()

                if question.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break

                if not question:
                    continue

                # Get response
                print("🤔 Thinking...")
                response = chatbot.chat(question)

                # Display answer
                print(f"\n🤖 Answer (confidence: {response.confidence:.0%}):")
                print(response.answer)

                # Display sources
                if response.sources:
                    print(f"\n📚 Sources ({len(response.sources)}):")
                    for i, source in enumerate(response.sources, 1):
                        score_percent = int(source.relevance_score * 100)
                        print(
                            f"  {i}. {source.title} ({source.source}) - {score_percent}% relevance"
                        )

                print("=" * 50)

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error in interactive chat: {str(e)}")
        return 1

    return 0


def index_documents(directory: str):
    """
    Index documents from a specific directory.

    Args:
        directory: Path to documents directory.
    """
    print(f"📁 Indexing documents from: {directory}")

    try:
        # Load and chunk documents
        documents = load_and_chunk_documents(directory)

        if not documents:
            print("⚠️  No documents found to index")
            return 1

        print(f"📄 Found {len(documents)} document chunks")

        # Initialize hybrid chatbot and add documents
        chatbot = create_hybrid_chatbot()
        chatbot.retrieval_engine.vector_store.add_documents(documents)

        print("✅ Documents indexed successfully!")

        # Show stats
        stats = chatbot.get_stats()
        vector_stats = stats["retrieval_engine"]["vector_store"]
        print(f"📊 Total documents in database: {vector_stats['document_count']}")

    except Exception as e:
        logger.error(f"Error indexing documents: {str(e)}")
        return 1

    return 0


def start_web_server():
    """
    Start the Flask web server.
    """
    try:
        from web_interface import app, init_chatbot

        # Initialize chatbot
        init_chatbot()

        settings = get_settings()

        print(
            f"🌐 Starting web server at http://{settings.flask_host}:{settings.flask_port}"
        )
        print("Press Ctrl+C to stop the server")

        app.run(
            host=settings.flask_host,
            port=settings.flask_port,
            debug=settings.flask_debug,
        )

    except Exception as e:
        logger.error(f"Error starting web server: {str(e)}")
        return 1

    return 0


def check_configuration():
    """
    Check and display configuration status.
    """
    print("🔧 Configuration Check")
    print("=" * 30)

    settings = get_settings()

    # Check OpenAI API key
    if settings.openai_api_key:
        print("✅ OpenAI API key configured")
    else:
        print("❌ OpenAI API key not configured")
        print("   Set OPENAI_API_KEY environment variable")

    # Check documents directory
    if os.path.exists(settings.docs_directory):
        doc_count = len(list(Path(settings.docs_directory).rglob("*.*")))
        print(f"✅ Documents directory exists ({doc_count} files)")
    else:
        print("❌ Documents directory not found")
        print(f"   Create directory: {settings.docs_directory}")

    # Check corporate portal configuration
    if settings.corporate_portal_url != "https://your-company-portal.com":
        print("✅ Corporate portal URL configured")
    else:
        print("⚠️  Corporate portal URL not configured (using default)")

    # Display key settings
    print("\n📋 Current Settings:")
    print(f"   Documents directory: {settings.docs_directory}")
    print(f"   Vector DB path: {settings.vector_db_path}")
    print(f"   Embedding model: {settings.embedding_model}")
    print(f"   Similarity threshold: {settings.similarity_threshold}")
    print(f"   Max docs to retrieve: {settings.max_docs_to_retrieve}")
    print(f"   Chunk size: {settings.chunk_size}")

    return 0


def auto_index_documents():
    """Automatically detect and index new documents."""
    print("🔄 Auto-Indexing Documents")
    print("=" * 30)

    try:
        indexer = create_auto_indexer()
        results = indexer.sync_documents()

        if results["new_documents"] > 0 or results["deleted_documents"] > 0:
            print(f"✅ Document sync completed!")
            print(f"📄 New documents: {results['new_documents']}")
            print(f"🗑️ Deleted documents: {results['deleted_documents']}")
            print(f"📊 Total chunks: {results['total_chunks']}")

            if results["indexed_files"]:
                print("\n📚 New files indexed:")
                for file_path in results["indexed_files"]:
                    print(f"   • {file_path}")

            if results["deleted_files"]:
                print("\n🗑️ Files removed from index:")
                for file_path in results["deleted_files"]:
                    print(f"   • {file_path}")
        else:
            print("✅ No changes detected - everything is up to date!")

        if results["errors"]:
            print("\n⚠️  Errors encountered:")
            for error in results["errors"]:
                print(f"   • {error}")
            return 1

        return 0

    except Exception as e:
        logger.error(f"Error during auto-indexing: {str(e)}")
        return 1


def show_indexing_status():
    """Show current indexing status."""
    print("📊 Document Indexing Status")
    print("=" * 30)

    try:
        indexer = create_auto_indexer()
        status = indexer.get_indexing_status()

        print(f"📁 Files in documents directory: {status['total_files_in_directory']}")
        print(f"🆕 New files detected: {status['new_files_detected']}")
        print(f"🗑️ Deleted files detected: {status['deleted_files_detected']}")
        print(f"💾 Documents in database: {status['documents_in_database']}")
        print(f"📋 Indexed file records: {status['indexed_file_records']}")

        if status["needs_indexing"]:
            if status["new_files_detected"] > 0:
                print(f"\n🆕 New files that need indexing:")
                for file_name in status["new_files"]:
                    print(f"   • {file_name}")
            if status["deleted_files_detected"] > 0:
                print(f"\n🗑️ Deleted files to remove from index:")
                for file_name in status["deleted_files"]:
                    print(f"   • {file_name}")
            print(f"\n💡 Run 'python main.py auto-index' to sync changes automatically")
        else:
            print("\n✅ All documents are synchronized and up to date!")

        return 0

    except Exception as e:
        logger.error(f"Error checking indexing status: {str(e)}")
        return 1


def test_connectivity():
    """Test connectivity to local and external services."""
    print("🔧 Testing Connectivity")
    print("=" * 30)

    try:
        # Test basic connectivity without full initialization
        settings = get_settings()

        # Test Ollama server connectivity
        try:
            import requests

            response = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
            ollama_available = response.status_code == 200
            print(
                f"{'✅' if ollama_available else '❌'} Ollama Server: {'Available' if ollama_available else 'Unavailable'}"
            )
        except Exception as e:
            print(f"❌ Ollama Server: Unavailable ({str(e)})")
            ollama_available = False

        # Test embedding model availability
        try:
            from src.ollama_embeddings import create_ollama_embeddings

            embedding_model = create_ollama_embeddings()
            test_embedding = embedding_model.encode(["test"])
            print(f"✅ Local Embeddings: Available (nomic-embed-text:latest)")
            embeddings_available = True
        except Exception as e:
            print(f"❌ Local Embeddings: Unavailable ({str(e)})")
            embeddings_available = False

        # Test LLM model availability
        try:
            from langchain_ollama import ChatOllama
            from langchain.schema import HumanMessage

            llm = ChatOllama(
                model=settings.ollama_model, base_url=settings.ollama_base_url
            )
            test_response = llm([HumanMessage(content="Test")])
            print(f"✅ Local LLM: Available ({settings.ollama_model})")
            llm_available = bool(test_response.content)
        except Exception as e:
            print(f"❌ Local LLM: Unavailable ({str(e)})")
            llm_available = False

        # Test external API if configured
        external_available = False
        if settings.enable_external_api_fallback and settings.external_api_key:
            try:
                if settings.external_api_provider.lower() == "openai":
                    from langchain_community.chat_models import ChatOpenAI

                    external_llm = ChatOpenAI(
                        model_name=settings.external_api_model,
                        openai_api_key=settings.external_api_key,
                    )
                    test_response = external_llm([HumanMessage(content="Test")])
                    external_available = bool(test_response.content)
                    print(
                        f"✅ External API: Available ({settings.external_api_provider})"
                    )
                else:
                    print(
                        f"⚠️  External API: {settings.external_api_provider} not tested"
                    )
            except Exception as e:
                print(f"❌ External API: Unavailable ({str(e)})")
        else:
            print("⚠️  External API: Not configured")

        # Test vector store directory
        try:
            import os

            vector_db_path = settings.vector_db_path
            if os.path.exists(vector_db_path):
                print(f"✅ Vector Store: Directory exists ({vector_db_path})")
                vector_store_available = True
            else:
                print(f"⚠️  Vector Store: Directory will be created ({vector_db_path})")
                vector_store_available = True
        except Exception as e:
            print(f"❌ Vector Store: Error ({str(e)})")
            vector_store_available = False

        # Summary and recommendations
        print("\n" + "=" * 30)
        print("📊 Summary:")
        print(f"   Ollama Server: {'✅' if ollama_available else '❌'}")
        print(f"   Local LLM: {'✅' if llm_available else '❌'}")
        print(f"   Local Embeddings: {'✅' if embeddings_available else '❌'}")
        print(f"   External API: {'✅' if external_available else '❌'}")
        print(f"   Vector Store: {'✅' if vector_store_available else '❌'}")

        print("\n💡 Recommendations:")
        if not ollama_available:
            print("   • Start Ollama: ollama serve")
        if not llm_available and ollama_available:
            print(f"   • Pull LLM model: ollama pull {settings.ollama_model}")
        if not embeddings_available and ollama_available:
            print(
                f"   • Pull embedding model: ollama pull {settings.ollama_embedding_model}"
            )
        if not external_available and settings.enable_external_api_fallback:
            print("   • Set EXTERNAL_API_KEY in .env for fallback support")

        return 0

    except Exception as e:
        logger.error(f"Error testing connectivity: {str(e)}")
        return 1


def main():
    """
    Main entry point with command-line argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="RAG Chatbot - Document-based question answering system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py web                    # Start web interface
  python main.py chat                   # Interactive chat mode
  python main.py index ./my-docs        # Index documents manually
  python main.py auto-index             # Auto-detect and index new documents
  python main.py status                 # Check indexing status
  python main.py config                 # Check configuration
  python main.py test                   # Test connectivity
        """,
    )

    parser.add_argument(
        "command",
        choices=["web", "chat", "index", "config", "test", "auto-index", "status"],
        help="Command to run",
    )

    parser.add_argument("path", nargs="?", help="Path for index command")

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Setup directories
    setup_directories()

    # Validate configuration for commands that need it
    if args.command in ["web", "chat", "index"]:
        if not validate_settings():
            print(
                "❌ Configuration validation failed. Use 'python main.py config' to check settings."
            )
            return 1

    # Execute command
    try:
        if args.command == "web":
            return start_web_server()
        elif args.command == "chat":
            return interactive_chat()
        elif args.command == "index":
            if not args.path:
                parser.error("index command requires a path argument")
            return index_documents(args.path)
        elif args.command == "config":
            return check_configuration()
        elif args.command == "test":
            return test_connectivity()
        elif args.command == "auto-index":
            return auto_index_documents()
        elif args.command == "status":
            return show_indexing_status()

    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

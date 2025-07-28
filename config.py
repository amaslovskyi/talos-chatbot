"""
Configuration module for the RAG chatbot.
Manages all environment variables and application settings.

MIT License - Copyright (c) 2025 talos-chatbot
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM Configuration
    use_local_llm: bool = True
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma3n:latest"
    openai_api_key: str = ""  # Only needed if use_local_llm = False

    # Embedding Configuration
    use_local_embeddings: bool = True
    ollama_embedding_model: str = "nomic-embed-text:latest"
    embedding_model: str = "all-MiniLM-L6-v2"  # Fallback for sentence-transformers

    # Knowledge Base Configuration
    # Documents directory (default local knowledge base)
    documents_directory: str = "documents"

    # Additional knowledge base paths (comma-separated)
    additional_knowledge_paths: str = ""  # e.g., "/path/to/kb1,/path/to/kb2"

    # Corporate Portal Configuration
    corporate_portal_url: str = ""  # Configure in .env file
    corporate_portal_api_key: str = ""
    corporate_portal_username: str = ""
    corporate_portal_password: str = ""

    # External URL Search Configuration
    enable_external_url_search: bool = True
    external_search_urls: str = (
        "https://github.com/snort3/snort3"  # Comma-separated URLs
    )
    url_search_timeout: int = 15  # seconds (increased for advanced crawler)
    url_fallback_threshold: float = 0.5  # minimum confidence to skip URL search
    strict_knowledge_base_mode: bool = (
        True  # Only answer from knowledge base, politely decline others
    )

    # Advanced Web Crawler Configuration
    use_advanced_crawler: bool = (
        True  # Use advanced crawler instead of simple URL search
    )
    max_crawl_depth: int = 2  # Maximum crawl depth for following links
    max_content_length: int = 5000  # Maximum content length per result
    extract_code_blocks: bool = True  # Extract code blocks from documentation
    render_javascript: bool = False  # Render JavaScript (requires more resources)

    # External API Configuration (fallback when local models fail)
    enable_external_api_fallback: bool = True
    external_api_provider: str = "openai"  # "openai", "anthropic", "google"
    external_api_key: str = ""  # API key for external provider
    external_api_model: str = "gpt-3.5-turbo"  # Model to use for external API

    # Document Processing Configuration
    docs_directory: str = "./documents"
    chunk_size: int = 1000
    chunk_overlap: int = 100

    # Vector Database Configuration
    vector_db_path: str = "./vector_db"

    # Retrieval Configuration
    similarity_threshold: float = 0.15  # Lowered for nomic-embed-text compatibility
    max_docs_to_retrieve: int = 5

    # Flask Web Interface
    flask_host: str = "127.0.0.1"
    flask_port: int = 5001
    flask_debug: bool = True

    class Config:
        # Load from .env file if it exists
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


def validate_settings() -> bool:
    """
    Validate that required settings are configured.
    Returns True if valid, False otherwise.
    """
    # Check LLM configuration
    if settings.use_local_llm:
        print(
            f"✅ Using local LLM: {settings.ollama_model} at {settings.ollama_base_url}"
        )
        # Check external API fallback
        if settings.enable_external_api_fallback:
            if settings.external_api_key:
                print(
                    f"✅ External API fallback enabled: {settings.external_api_provider}"
                )
            else:
                print("⚠️  External API fallback enabled but no API key set")
        # TODO: Could add Ollama connectivity check here
    else:
        if not settings.openai_api_key:
            print("❌ OPENAI_API_KEY not set (required when use_local_llm=False)")
            return False
        print("✅ Using OpenAI API")

    # Check embeddings configuration
    if settings.use_local_embeddings:
        print(f"✅ Using local embeddings: {settings.ollama_embedding_model}")
    else:
        print(f"✅ Using sentence-transformers: {settings.embedding_model}")

    if not os.path.exists(settings.docs_directory):
        print(f"⚠️  Documents directory {settings.docs_directory} does not exist")
        print("   This will be created automatically")

    return True


def get_settings() -> Settings:
    """Get the current settings instance."""
    return settings

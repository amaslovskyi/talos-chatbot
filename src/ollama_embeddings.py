"""
Custom embedding implementation for Ollama models.
Supports nomic-embed-text and other Ollama embedding models.

MIT License - Copyright (c) 2025 talos-chatbot
"""

import logging
from typing import List, Optional
import requests
import numpy as np

from config import get_settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaEmbeddings:
    """
    Custom embedding class for Ollama models.
    Compatible with ChromaDB and sentence-transformers interface.
    """

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize Ollama embeddings.

        Args:
            model: Ollama model name (e.g., "nomic-embed-text:latest")
            base_url: Ollama server URL (default: http://localhost:11434)
        """
        self.settings = get_settings()
        self.model = model or self.settings.ollama_embedding_model
        self.base_url = base_url or self.settings.ollama_base_url

        # Ensure base_url doesn't end with slash
        self.base_url = self.base_url.rstrip("/")

        logger.info(f"Initialized Ollama embeddings with model: {self.model}")

        # Test connection
        self._test_connection()

    def _test_connection(self):
        """Test connection to Ollama server."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]

                if self.model in model_names:
                    logger.info(f"✅ Found model {self.model} in Ollama")
                else:
                    logger.warning(
                        f"⚠️  Model {self.model} not found in Ollama. Available models: {model_names}"
                    )
                    logger.warning(f"💡 Run: ollama pull {self.model}")
            else:
                logger.warning(f"⚠️  Could not connect to Ollama at {self.base_url}")
        except Exception as e:
            logger.warning(f"⚠️  Ollama connection test failed: {str(e)}")
            logger.warning("💡 Make sure Ollama is running: ollama serve")

    def encode(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
        """
        Encode texts to embeddings.
        Compatible with sentence-transformers interface.

        Args:
            texts: List of texts to encode
            show_progress_bar: Whether to show progress (for compatibility)

        Returns:
            NumPy array of embeddings
        """
        if not texts:
            return np.array([])

        embeddings = []

        if show_progress_bar:
            try:
                from tqdm import tqdm

                texts_iter = tqdm(texts, desc="Generating embeddings")
            except ImportError:
                texts_iter = texts
        else:
            texts_iter = texts

        for text in texts_iter:
            try:
                embedding = self._get_embedding(text)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error getting embedding for text: {str(e)}")
                # Return zero vector as fallback
                embeddings.append([0.0] * 768)  # Assuming 768-dim embeddings

        return np.array(embeddings)

    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of embedding values
        """
        url = f"{self.base_url}/api/embeddings"

        payload = {"model": self.model, "prompt": text}

        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()

            data = response.json()
            embedding = data.get("embedding", [])

            if not embedding:
                logger.error(f"Empty embedding returned for text: {text[:50]}...")
                return [0.0] * 768  # Fallback

            return embedding

        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error processing embedding response: {str(e)}")
            raise

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings from this model.

        Returns:
            Embedding dimension
        """
        try:
            # Test with a simple text to get dimension
            test_embedding = self._get_embedding("test")
            return len(test_embedding)
        except Exception as e:
            logger.warning(f"Could not determine embedding dimension: {str(e)}")
            # Common dimensions for embedding models
            if "nomic" in self.model.lower():
                return 768
            else:
                return 768  # Default assumption


def create_ollama_embeddings(model: Optional[str] = None) -> OllamaEmbeddings:
    """
    Factory function to create Ollama embeddings.

    Args:
        model: Ollama model name

    Returns:
        OllamaEmbeddings instance
    """
    return OllamaEmbeddings(model)

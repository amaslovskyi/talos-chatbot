"""
Vector store module for document embeddings and similarity search.
Uses ChromaDB for vector storage and SentenceTransformers for embeddings.

MIT License - Copyright (c) 2025 talos-chatbot
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple

# Vector storage and embedding imports
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

# LangChain imports
from langchain.schema import Document as LangChainDocument

from config import get_settings
from src.ollama_embeddings import create_ollama_embeddings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages vector embeddings and similarity search using ChromaDB.
    Handles document embedding, storage, and retrieval operations.
    """

    def __init__(self, collection_name: str = "documents"):
        """
        Initialize the vector store with ChromaDB and embedding model.

        Args:
            collection_name: Name of the ChromaDB collection to use.
        """
        self.settings = get_settings()
        self.collection_name = collection_name

        # Initialize embedding model
        if self.settings.use_local_embeddings:
            logger.info(
                f"Loading Ollama embedding model: {self.settings.ollama_embedding_model}"
            )
            self.embedding_model = create_ollama_embeddings()
        else:
            logger.info(
                f"Loading sentence-transformers model: {self.settings.embedding_model}"
            )
            self.embedding_model = SentenceTransformer(self.settings.embedding_model)

        # Initialize ChromaDB client
        self._init_chromadb()

        # Get or create collection
        self.collection = self._get_or_create_collection()

    def _init_chromadb(self):
        """Initialize ChromaDB client with persistent storage."""
        # Ensure vector database directory exists
        os.makedirs(self.settings.vector_db_path, exist_ok=True)

        # Create ChromaDB client with persistent storage
        self.chroma_client = chromadb.PersistentClient(
            path=self.settings.vector_db_path,
            settings=ChromaSettings(anonymized_telemetry=False, is_persistent=True),
        )
        logger.info(f"Initialized ChromaDB at: {self.settings.vector_db_path}")

    def _get_or_create_collection(self):
        """Get existing collection or create a new one."""
        try:
            # Try to get existing collection
            collection = self.chroma_client.get_collection(name=self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
            return collection
        except (ValueError, Exception):
            # Create new collection if it doesn't exist
            try:
                collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Document embeddings for RAG chatbot"},
                )
                logger.info(f"Created new collection: {self.collection_name}")
                return collection
            except Exception as e:
                logger.error(f"Error creating collection: {str(e)}")
                raise

    def add_documents(self, documents: List[LangChainDocument]) -> None:
        """
        Add documents to the vector store with embeddings.

        Args:
            documents: List of LangChain documents to add.
        """
        if not documents:
            logger.warning("No documents to add")
            return

        logger.info(f"Adding {len(documents)} documents to vector store")

        # Prepare data for ChromaDB
        texts = []
        metadatas = []
        ids = []

        for i, doc in enumerate(documents):
            # Create unique ID for each document chunk
            doc_id = f"doc_{i}_{hash(doc.page_content[:100])}"

            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
            ids.append(doc_id)

        # Generate embeddings for all texts
        logger.info("Generating embeddings...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        # Add to ChromaDB collection
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )

        logger.info(f"Successfully added {len(documents)} documents to vector store")

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[Tuple[LangChainDocument, float]]:
        """
        Search for similar documents using vector similarity.

        Args:
            query: Search query text.
            k: Number of documents to retrieve. Uses config default if None.
            similarity_threshold: Minimum similarity score. Uses config default if None.

        Returns:
            List of tuples containing (document, similarity_score).
        """
        if k is None:
            k = self.settings.max_docs_to_retrieve
        if similarity_threshold is None:
            similarity_threshold = self.settings.similarity_threshold

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])

        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(), n_results=k
        )

        # Process results
        documents_with_scores = []

        if results["documents"] and results["documents"][0]:
            for i, (doc_text, metadata, distance) in enumerate(
                zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ):
                # For nomic-embed-text, use distance-based filtering (lower distance = more similar)
                # Convert to a simple similarity score for consistent API
                if (
                    distance <= 400
                ):  # Reasonable distance threshold for nomic-embed-text
                    similarity_score = max(
                        0.1, 1 - (distance / 400)
                    )  # Scale to 0.1-1.0 range
                else:
                    similarity_score = 0.05  # Very low similarity for high distances

                # Filter by similarity threshold
                if similarity_score >= similarity_threshold:
                    doc = LangChainDocument(
                        page_content=doc_text, metadata=metadata or {}
                    )
                    documents_with_scores.append((doc, similarity_score))

        logger.info(
            f"Found {len(documents_with_scores)} documents above threshold {similarity_threshold}"
        )
        return documents_with_scores

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store collection.

        Returns:
            Dictionary with collection statistics.
        """
        count = self.collection.count()
        embedding_info = (
            self.settings.ollama_embedding_model
            if self.settings.use_local_embeddings
            else self.settings.embedding_model
        )

        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "embedding_model": embedding_info,
            "embedding_type": "ollama"
            if self.settings.use_local_embeddings
            else "sentence-transformers",
            "vector_db_path": self.settings.vector_db_path,
        }

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        # Delete the collection and recreate it
        self.chroma_client.delete_collection(name=self.collection_name)
        self.collection = self._get_or_create_collection()
        logger.info(f"Cleared collection: {self.collection_name}")

    def document_exists(self, doc_id: str) -> bool:
        """
        Check if a document with the given ID exists in the collection.

        Args:
            doc_id: Document ID to check.

        Returns:
            True if document exists, False otherwise.
        """
        try:
            result = self.collection.get(ids=[doc_id])
            return len(result["ids"]) > 0
        except Exception:
            return False


def create_vector_store(collection_name: str = "documents") -> VectorStore:
    """
    Factory function to create a vector store instance.

    Args:
        collection_name: Name of the ChromaDB collection.

    Returns:
        Initialized VectorStore instance.
    """
    return VectorStore(collection_name)

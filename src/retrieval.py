"""
Retrieval logic module.
Orchestrates document retrieval from local vector store and corporate portal.
Implements confidence scoring and fallback mechanisms.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

# LangChain imports
from langchain.schema import Document as LangChainDocument

from src.vector_store import VectorStore, create_vector_store
from src.corporate_portal import CorporatePortalClient, create_portal_client
from config import get_settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """
    Represents a single retrieval result with metadata.
    """

    content: str
    title: str
    source: str  # 'local' or 'corporate_portal'
    relevance_score: float
    metadata: Dict[str, Any]
    url: Optional[str] = None


class RetrievalEngine:
    """
    Main retrieval engine that coordinates local and corporate portal searches.
    Implements intelligent fallback logic based on confidence scores.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        portal_client: Optional[CorporatePortalClient] = None,
    ):
        """
        Initialize the retrieval engine.

        Args:
            vector_store: Vector store instance. Creates new one if None.
            portal_client: Corporate portal client. Creates new one if None.
        """
        self.settings = get_settings()

        # Initialize components
        self.vector_store = vector_store or create_vector_store()
        self.portal_client = portal_client or create_portal_client()

        # Retrieval thresholds
        self.local_confidence_threshold = self.settings.similarity_threshold
        self.min_local_results = 2  # Minimum local results before considering fallback

        logger.info("Initialized retrieval engine")

    def retrieve_documents(
        self, query: str, max_results: int = 5
    ) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        """
        Retrieve relevant documents using local vector store and corporate portal fallback.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.

        Returns:
            Tuple of (retrieved documents, retrieval metadata).
        """
        logger.info(f"Retrieving documents for query: '{query}'")

        # Initialize retrieval metadata
        retrieval_metadata = {
            "query": query,
            "local_results_count": 0,
            "portal_results_count": 0,
            "used_fallback": False,
            "confidence_scores": [],
        }

        # Step 1: Search local vector store
        local_results = self._search_local_documents(query, max_results)
        retrieval_metadata["local_results_count"] = len(local_results)

        # Step 2: Evaluate local results confidence
        high_confidence_local = [
            r
            for r in local_results
            if r.relevance_score >= self.local_confidence_threshold
        ]

        # Step 3: Decide whether to use corporate portal fallback
        use_fallback = self._should_use_fallback(high_confidence_local, query)
        retrieval_metadata["used_fallback"] = use_fallback

        all_results = []

        # Add high-confidence local results
        all_results.extend(high_confidence_local)

        # Step 4: Corporate portal fallback if needed
        if use_fallback:
            logger.info("Using corporate portal fallback")
            portal_results = self._search_corporate_portal(query, max_results)
            retrieval_metadata["portal_results_count"] = len(portal_results)
            all_results.extend(portal_results)
        else:
            # Add remaining local results if not using fallback
            low_confidence_local = [
                r
                for r in local_results
                if r.relevance_score < self.local_confidence_threshold
            ]
            all_results.extend(low_confidence_local)

        # Step 5: Sort and limit results
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)
        final_results = all_results[:max_results]

        # Update metadata
        retrieval_metadata["confidence_scores"] = [
            r.relevance_score for r in final_results
        ]
        retrieval_metadata["final_results_count"] = len(final_results)

        logger.info(
            f"Retrieved {len(final_results)} documents "
            f"({retrieval_metadata['local_results_count']} local, "
            f"{retrieval_metadata['portal_results_count']} portal)"
        )

        return final_results, retrieval_metadata

    def _search_local_documents(
        self, query: str, max_results: int
    ) -> List[RetrievalResult]:
        """
        Search local vector store for relevant documents.

        Args:
            query: Search query string.
            max_results: Maximum number of results.

        Returns:
            List of local retrieval results.
        """
        try:
            # Check if collection has any documents
            collection_stats = self.vector_store.get_collection_stats()
            if collection_stats["document_count"] == 0:
                logger.info("No documents in vector store, skipping local search")
                return []

            # Search vector store
            docs_with_scores = self.vector_store.similarity_search(
                query=query,
                k=max_results,
                similarity_threshold=0.0,  # Get all results for confidence evaluation
            )

            # Convert to RetrievalResult objects
            results = []
            for doc, score in docs_with_scores:
                result = RetrievalResult(
                    content=doc.page_content,
                    title=doc.metadata.get("filename", "Unknown Document"),
                    source="local",
                    relevance_score=score,
                    metadata=doc.metadata,
                    url=doc.metadata.get("source"),
                )
                results.append(result)

            logger.info(f"Found {len(results)} local documents")
            return results

        except Exception as e:
            logger.error(f"Error searching local documents: {str(e)}")
            return []

    def _search_corporate_portal(
        self, query: str, max_results: int
    ) -> List[RetrievalResult]:
        """
        Search corporate portal for relevant documents.

        Args:
            query: Search query string.
            max_results: Maximum number of results.

        Returns:
            List of corporate portal retrieval results.
        """
        try:
            # Search corporate portal
            portal_docs = self.portal_client.search_documents(query, max_results)

            # Convert to RetrievalResult objects
            results = []
            for doc in portal_docs:
                result = RetrievalResult(
                    content=doc.get("content", ""),
                    title=doc.get("title", "Unknown Document"),
                    source="corporate_portal",
                    relevance_score=doc.get("relevance_score", 0.5),
                    metadata=doc.get("metadata", {}),
                    url=doc.get("url"),
                )
                results.append(result)

            logger.info(f"Found {len(results)} corporate portal documents")
            return results

        except Exception as e:
            logger.error(f"Error searching corporate portal: {str(e)}")
            return []

    def _should_use_fallback(
        self, local_results: List[RetrievalResult], query: str
    ) -> bool:
        """
        Determine whether to use corporate portal fallback based on local results quality.

        Args:
            local_results: High-confidence local results.
            query: Original search query.

        Returns:
            True if should use fallback, False otherwise.
        """
        # Use fallback if:
        # 1. No high-confidence local results
        # 2. Very few high-confidence local results
        # 3. Average confidence is low

        if not local_results:
            logger.info("No high-confidence local results, using fallback")
            return True

        if len(local_results) < self.min_local_results:
            logger.info(f"Only {len(local_results)} local results, using fallback")
            return True

        # Calculate average confidence
        avg_confidence = sum(r.relevance_score for r in local_results) / len(
            local_results
        )
        confidence_threshold_for_fallback = self.local_confidence_threshold + 0.1

        if avg_confidence < confidence_threshold_for_fallback:
            logger.info(
                f"Low average confidence ({avg_confidence:.3f}), using fallback"
            )
            return True

        logger.info(f"Sufficient local results (avg confidence: {avg_confidence:.3f})")
        return False

    def get_document_content(self, result: RetrievalResult) -> Optional[str]:
        """
        Get full content for a retrieval result.

        Args:
            result: RetrievalResult to get full content for.

        Returns:
            Full document content, or None if unavailable.
        """
        if result.source == "local":
            # Local documents already have full content
            return result.content
        elif result.source == "corporate_portal" and result.url:
            # Fetch full content from corporate portal
            return self.portal_client.get_document_content(result.url)
        else:
            return result.content

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the retrieval engine components.

        Returns:
            Dictionary with retrieval engine statistics.
        """
        vector_stats = self.vector_store.get_collection_stats()

        return {
            "vector_store": vector_stats,
            "corporate_portal_url": self.settings.corporate_portal_url,
            "similarity_threshold": self.local_confidence_threshold,
            "min_local_results": self.min_local_results,
        }


def create_retrieval_engine() -> RetrievalEngine:
    """
    Factory function to create a retrieval engine.

    Returns:
        Initialized RetrievalEngine instance.
    """
    return RetrievalEngine()

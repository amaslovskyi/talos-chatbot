"""
Retrieval logic module.
Orchestrates document retrieval from local vector store, corporate portal, and external URLs.
Implements confidence scoring and fallback mechanisms.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

# LangChain imports
from langchain.schema import Document as LangChainDocument

from src.vector_store import VectorStore, create_vector_store
from src.corporate_portal import CorporatePortalClient, create_portal_client
from src.url_search import ExternalURLSearcher, URLSearchResult, create_url_searcher
from src.crawler_manager import (
    CrawlerManager,
    UnifiedCrawlResult,
    create_crawler_manager,
)
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
        url_searcher: Optional[ExternalURLSearcher] = None,
        crawler_manager: Optional[CrawlerManager] = None,
    ):
        """
        Initialize the retrieval engine.

        Args:
            vector_store: Vector store instance. Creates new one if None.
            portal_client: Corporate portal client. Creates new one if None.
            url_searcher: External URL searcher. Creates new one if None.
            crawler_manager: Unified crawler manager. Creates new one if None.
        """
        self.settings = get_settings()

        # Initialize components
        self.vector_store = vector_store or create_vector_store()
        self.portal_client = portal_client or create_portal_client()
        self.url_searcher = url_searcher or create_url_searcher()
        self.crawler_manager = crawler_manager or create_crawler_manager()

        # Retrieval thresholds
        self.local_confidence_threshold = self.settings.similarity_threshold
        self.min_local_results = 2  # Minimum local results before considering fallback
        self.url_fallback_threshold = getattr(
            self.settings, "url_fallback_threshold", 0.5
        )  # Threshold for using URL fallback

        # Use unified crawler manager by default
        self.use_crawler_manager = getattr(self.settings, "use_advanced_crawler", True)

        logger.info(
            "Initialized retrieval engine with unified crawler manager and URL search fallback"
        )

    def retrieve_documents(
        self, query: str, max_results: int = 5
    ) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        """
        Retrieve relevant documents using local vector store, corporate portal, and external URL fallback.

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
            "url_results_count": 0,
            "used_fallback": False,
            "used_url_fallback": False,
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

        # Step 3: Decide whether to use fallback mechanisms
        use_fallback = self._should_use_fallback(high_confidence_local, query)
        retrieval_metadata["used_fallback"] = use_fallback

        all_results = []

        # Add ALL local results first (prioritize local content)
        all_results.extend(local_results)

        logger.info(f"Found {len(local_results)} local documents")
        if local_results:
            avg_local_confidence = sum(r.relevance_score for r in local_results) / len(
                local_results
            )
            logger.info(
                f"Sufficient local results (avg confidence: {avg_local_confidence:.3f})"
            )

        # Step 4: Corporate portal fallback if needed
        if use_fallback:
            logger.info("Using corporate portal fallback")
            portal_results = self._search_corporate_portal(query, max_results)
            retrieval_metadata["portal_results_count"] = len(portal_results)
            all_results.extend(portal_results)

        # Step 5: External URL fallback - only for specialized queries with low local confidence
        use_url_fallback = self._should_use_url_fallback(all_results, query)
        retrieval_metadata["used_url_fallback"] = use_url_fallback

        if use_url_fallback:
            logger.info("🌐 Using external URL fallback search")
            url_results = self._search_external_urls(query, max_results)
            retrieval_metadata["url_results_count"] = len(url_results)
            all_results.extend(url_results)

        # Step 6: Sort and limit results
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
            f"{retrieval_metadata['portal_results_count']} portal, "
            f"{retrieval_metadata['url_results_count']} external)"
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

    def _should_use_url_fallback(
        self, current_results: List[RetrievalResult], query: str
    ) -> bool:
        """
        Determine if external URL search should be used as fallback.
        Only searches external URLs for topics that might be relevant to configured sources.

        Args:
            current_results: Results from local and portal searches
            query: Original search query

        Returns:
            True if URL fallback should be used
        """
        # Check if external URL search is enabled
        if not getattr(self.settings, "enable_external_url_search", True):
            return False

        # Use URL fallback only if:
        # 1. No results at all, OR
        # 2. All results have very low confidence scores (below threshold)
        # Note: We're being more restrictive - only search external URLs when we have
        # some indication the query might be relevant to our knowledge domain

        if not current_results:
            logger.info(
                "📊 No local/portal results found - checking if topic might be in external sources"
            )
            # Only search external URLs if query contains keywords related to our domain
            if self._is_query_potentially_relevant(query):
                logger.info(
                    "📊 Query seems relevant to knowledge domain - triggering URL fallback"
                )
                return True
            else:
                logger.info(
                    "📊 Query not relevant to knowledge domain - skipping URL fallback"
                )
                return False

        # Check if we have insufficient information to answer the question properly
        # Use URL fallback if:
        # 1. All results have low confidence (below threshold), OR
        # 2. We have some results but they might not be comprehensive enough

        high_confidence_results = [
            r
            for r in current_results
            if r.relevance_score >= self.url_fallback_threshold
        ]

        # Calculate average confidence of current results
        avg_confidence = (
            sum(r.relevance_score for r in current_results) / len(current_results)
            if current_results
            else 0
        )

        # Trigger URL fallback if we don't have high-confidence, comprehensive results
        should_search_urls = False

        # Be more conservative - only use external for very specific cases
        query_lower = query.lower()
        is_specialized_query = any(
            term in query_lower
            for term in [
                "upgrade",
                "migration",
                "plugin",
                "plugins",
                "module",
                "modules",
                "install",
                "setup",
                "build",
                "configure",
            ]
        )

        # Only use external URLs when:
        # 1. No results at all, OR
        # 2. Very few results AND it's a specialized query that likely needs external documentation
        if not current_results:
            logger.info("📊 No local/portal results found")
            should_search_urls = True
        elif len(current_results) < 2 and is_specialized_query:
            logger.info(
                "📊 Too few results for specialized query - checking external sources"
            )
            should_search_urls = True
        elif avg_confidence < 0.3 and is_specialized_query:
            logger.info(
                f"📊 Low confidence ({avg_confidence:.2f}) for specialized query - checking external sources"
            )
            should_search_urls = True

        if should_search_urls:
            if self._is_query_potentially_relevant(query):
                logger.info(
                    "📊 Query seems relevant to knowledge domain - triggering URL fallback"
                )
                return True
            else:
                logger.info(
                    "📊 Query not relevant to knowledge domain - skipping URL fallback"
                )
                return False

        logger.info(
            f"📊 Sufficient local/portal results found (avg confidence: {avg_confidence:.2f})"
        )
        return False

    def _is_query_potentially_relevant(self, query: str) -> bool:
        """
        Check if the query might be relevant to our configured knowledge domain.

        Args:
            query: Search query string

        Returns:
            True if query might be relevant to our external sources
        """
        query_lower = query.lower()

        # Extract domain keywords from configured URLs
        domain_keywords = set()

        for url in self.url_searcher.default_urls:
            if "snort" in url.lower():
                domain_keywords.update(
                    [
                        "snort",
                        "intrusion",
                        "detection",
                        "security",
                        "network",
                        "ips",
                        "ids",
                    ]
                )
            if "andrewng" in url.lower() or "ai" in url.lower():
                domain_keywords.update(
                    [
                        "ai",
                        "machine learning",
                        "deep learning",
                        "artificial intelligence",
                        "neural",
                    ]
                )

        # Add keywords from local documents (basic detection)
        domain_keywords.update(
            ["install", "configuration", "setup", "build", "compile", "documentation"]
        )

        # Check if query contains any relevant keywords
        for keyword in domain_keywords:
            if keyword in query_lower:
                return True

        # Check if query is asking about the configured tools/projects specifically
        if any(
            term in query_lower
            for term in [
                "how to",
                "install",
                "configure",
                "setup",
                "build",
                "compile",
                "contribute",
            ]
        ):
            return True

        return False

    def _search_external_urls(
        self, query: str, max_results: int
    ) -> List[RetrievalResult]:
        """
        Search external URLs for relevant information.

        Args:
            query: Search query string
            max_results: Maximum number of results

        Returns:
            List of retrieval results from external URLs
        """
        try:
            # Get configured external URLs
            external_urls = []
            if (
                hasattr(self.settings, "external_search_urls")
                and self.settings.external_search_urls
            ):
                external_urls = [
                    url.strip() for url in self.settings.external_search_urls.split(",")
                ]

            if self.settings.corporate_portal_url:
                external_urls.append(self.settings.corporate_portal_url)

            if not external_urls:
                logger.warning("No external URLs configured for search")
                return []

            # Use unified crawler manager if available and enabled
            if self.use_crawler_manager and self.crawler_manager:
                logger.info(
                    f"🕷️ Using unified crawler manager for {len(external_urls)} URLs"
                )
                crawl_results = self.crawler_manager.crawl_urls(
                    external_urls, query, max_results
                )

                # Convert UnifiedCrawlResult to RetrievalResult
                retrieval_results = []
                for crawl_result in crawl_results:
                    # Create comprehensive content from all sections with priority for important sections
                    comprehensive_content = crawl_result.content

                    # Prioritize build/install sections for technical queries
                    high_priority = ["DEPENDENCIES", "BUILD", "INSTALL", "DOWNLOAD"]
                    medium_priority = ["RUN", "USAGE", "OVERVIEW"]

                    high_priority_sections = []
                    medium_priority_sections = []
                    other_sections = []

                    if crawl_result.extracted_sections:
                        for (
                            section_name,
                            section_content,
                        ) in crawl_result.extracted_sections.items():
                            if section_content.strip():
                                section_text = f"## {section_name}\n\n{section_content}"

                                # Check for high priority sections (build/install related)
                                if any(
                                    priority in section_name.upper()
                                    for priority in high_priority
                                ):
                                    high_priority_sections.append(section_text)
                                # Check for medium priority sections
                                elif any(
                                    priority in section_name.upper()
                                    for priority in medium_priority
                                ):
                                    medium_priority_sections.append(section_text)
                                else:
                                    other_sections.append(section_text)

                        # Always prioritize high priority sections (build/install) for technical documentation
                        all_sections = (
                            high_priority_sections
                            + medium_priority_sections[:2]
                            + other_sections[:1]
                        )

                        if all_sections:
                            comprehensive_content = "\n\n".join(all_sections)

                    # Allow more content for build/install queries
                    query_lower = query.lower()
                    is_build_query = any(
                        term in query_lower
                        for term in [
                            "build",
                            "install",
                            "compile",
                            "setup",
                            "dependencies",
                            "cmake",
                        ]
                    )
                    max_length = 15000 if is_build_query else 8000

                    result = RetrievalResult(
                        content=comprehensive_content[
                            :max_length
                        ],  # Much longer content for detailed technical documentation
                        title=crawl_result.title,
                        source=f"{crawl_result.crawler_type}_crawler_{crawl_result.content_type}",
                        relevance_score=crawl_result.relevance_score,
                        metadata={
                            **crawl_result.metadata,
                            "search_source": f"{crawl_result.crawler_type}_crawler",
                            "extraction_method": crawl_result.extraction_method,
                            "content_type": crawl_result.content_type,
                            "sections_count": len(crawl_result.extracted_sections),
                            "timestamp": crawl_result.timestamp,
                        },
                        url=crawl_result.url,
                    )
                    retrieval_results.append(result)

                logger.info(
                    f"🕷️ Unified crawler manager found {len(retrieval_results)} results"
                )
                return retrieval_results

            else:
                # Fallback to simple URL searcher
                logger.info("🌐 Using simple URL searcher as fallback")
                url_results = self.url_searcher.search_external_sources(
                    query=query, max_results=max_results
                )

                # Convert URLSearchResult to RetrievalResult
                retrieval_results = []
                for url_result in url_results:
                    result = RetrievalResult(
                        content=url_result.content,
                        title=url_result.title,
                        source=f"external_url_{url_result.source_type}",
                        relevance_score=url_result.relevance_score,
                        metadata={
                            **url_result.metadata,
                            "search_source": "external_url",
                            "url_source_type": url_result.source_type,
                        },
                        url=url_result.url,
                    )
                    retrieval_results.append(result)

                logger.info(
                    f"🌐 Simple URL searcher found {len(retrieval_results)} results"
                )
                return retrieval_results

        except Exception as e:
            logger.error(f"❌ Error searching external URLs: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            return []


def create_retrieval_engine() -> RetrievalEngine:
    """
    Factory function to create a retrieval engine.

    Returns:
        Initialized RetrievalEngine instance.
    """
    return RetrievalEngine()

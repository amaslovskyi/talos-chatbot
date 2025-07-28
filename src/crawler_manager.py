"""
Unified crawler manager that routes URLs to appropriate specialized crawlers.
Supports GitHub repositories and general web content with intelligent URL type detection.
"""

import logging
from typing import List, Dict, Any, Union
from urllib.parse import urlparse
from dataclasses import dataclass

from src.github_crawler import (
    GitHubCrawler,
    CrawlResult as GitHubCrawlResult,
    create_github_crawler,
)
from src.web_crawler import WebCrawler, WebCrawlResult, create_web_crawler
from config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class UnifiedCrawlResult:
    """Unified result container that can hold either GitHub or Web crawl results."""

    url: str
    title: str
    content: str
    extracted_sections: Dict[str, str]
    links: List[Dict[str, str]]
    metadata: Dict[str, Any]
    relevance_score: float
    content_type: str
    extraction_method: str
    timestamp: float
    crawler_type: str  # 'github' or 'web'


class CrawlerManager:
    """
    Unified crawler manager that intelligently routes URLs to appropriate specialized crawlers.
    """

    def __init__(self):
        """Initialize the crawler manager with both GitHub and Web crawlers."""
        self.settings = get_settings()

        # Initialize specialized crawlers
        self.github_crawler = create_github_crawler()
        self.web_crawler = create_web_crawler()

        logger.info("Crawler manager initialized with GitHub and Web crawlers")

    def crawl_urls(
        self, urls: List[str], query: str, max_results: int = 10
    ) -> List[UnifiedCrawlResult]:
        """
        Crawl multiple URLs using the appropriate crawler for each URL type.

        Args:
            urls: List of URLs to crawl
            query: Search query for relevance scoring
            max_results: Maximum number of results to return

        Returns:
            List of UnifiedCrawlResult objects
        """
        logger.info(
            f"🚀 Starting unified crawl of {len(urls)} URLs for query: '{query}'"
        )

        all_results = []

        # Categorize URLs by type
        github_urls = []
        web_urls = []

        for url in urls:
            if self._is_github_url(url):
                github_urls.append(url)
            else:
                web_urls.append(url)

        logger.info(f"📂 GitHub URLs: {len(github_urls)}, 🌐 Web URLs: {len(web_urls)}")

        # Crawl GitHub URLs
        if github_urls:
            try:
                github_results = self.github_crawler.crawl_urls(
                    github_urls, query, max_results
                )
                for result in github_results:
                    unified_result = self._convert_github_result(result)
                    all_results.append(unified_result)
                logger.info(f"📂 GitHub crawler found {len(github_results)} results")
            except Exception as e:
                logger.error(f"Error in GitHub crawler: {str(e)}")

        # Crawl Web URLs
        if web_urls:
            try:
                web_results = self.web_crawler.crawl_urls(web_urls, query, max_results)
                for result in web_results:
                    unified_result = self._convert_web_result(result)
                    all_results.append(unified_result)
                logger.info(f"🌐 Web crawler found {len(web_results)} results")
            except Exception as e:
                logger.error(f"Error in Web crawler: {str(e)}")

        # Sort by relevance and limit results
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)
        final_results = all_results[:max_results]

        logger.info(f"🎯 Unified crawl completed: {len(final_results)} total results")
        return final_results

    def _is_github_url(self, url: str) -> bool:
        """Check if URL is a GitHub repository or GitHub-related URL."""
        github_indicators = [
            "github.com",
            "raw.githubusercontent.com",
            "api.github.com",
        ]

        url_lower = url.lower()
        return any(indicator in url_lower for indicator in github_indicators)

    def _convert_github_result(
        self, github_result: GitHubCrawlResult
    ) -> UnifiedCrawlResult:
        """Convert a GitHub crawl result to unified format."""
        return UnifiedCrawlResult(
            url=github_result.url,
            title=github_result.title,
            content=github_result.content,
            extracted_sections=github_result.extracted_sections,
            links=github_result.links,
            metadata=github_result.metadata,
            relevance_score=github_result.relevance_score,
            content_type=github_result.content_type,
            extraction_method=github_result.extraction_method,
            timestamp=github_result.timestamp,
            crawler_type="github",
        )

    def _convert_web_result(self, web_result: WebCrawlResult) -> UnifiedCrawlResult:
        """Convert a Web crawl result to unified format."""
        return UnifiedCrawlResult(
            url=web_result.url,
            title=web_result.title,
            content=web_result.content,
            extracted_sections=web_result.extracted_sections,
            links=web_result.links,
            metadata=web_result.metadata,
            relevance_score=web_result.relevance_score,
            content_type=web_result.content_type,
            extraction_method=web_result.extraction_method,
            timestamp=web_result.timestamp,
            crawler_type="web",
        )


def create_crawler_manager() -> CrawlerManager:
    """Factory function to create a crawler manager instance."""
    return CrawlerManager()

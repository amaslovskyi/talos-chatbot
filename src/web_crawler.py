"""
Web crawler for general HTML websites and documentation portals.
Supports comprehensive content extraction from various web sources using multiple extraction libraries.
"""

import logging
import time
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse, parse_qs
from config import get_settings

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class WebCrawlResult:
    """Container for web crawl results with rich metadata."""

    url: str
    title: str
    content: str
    extracted_sections: Dict[str, str]  # section_name -> content
    links: List[Dict[str, str]]  # internal links found
    metadata: Dict[str, Any]
    relevance_score: float
    content_type: str  # 'documentation', 'article', 'portal', 'general'
    extraction_method: str  # which library was used
    timestamp: float


class WebCrawler:
    """
    Web crawler for general HTML websites using multiple extraction libraries.
    Optimized for documentation portals, knowledge bases, and general web content.
    """

    def __init__(self):
        """Initialize the web crawler with multiple extraction backends."""
        self.settings = get_settings()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (compatible; TalosChatbot/1.0; +https://github.com/talos-chatbot)"
            }
        )

        # Initialize extraction libraries
        self._init_extraction_libraries()

        logger.info(
            f"Web crawler initialized with libraries: "
            f"requests-html={self.has_requests_html}, "
            f"trafilatura={self.has_trafilatura}, "
            f"newspaper={self.has_newspaper}"
        )

    def _init_extraction_libraries(self):
        """Initialize available extraction libraries."""
        # Check for requests-html
        try:
            from requests_html import HTMLSession

            self.has_requests_html = True
            self.html_session = HTMLSession()
        except ImportError:
            self.has_requests_html = False
            logger.warning("requests-html not available")

        # Check for trafilatura
        try:
            import trafilatura

            self.has_trafilatura = True
            self.trafilatura = trafilatura
        except ImportError:
            self.has_trafilatura = False
            logger.warning("trafilatura not available")

        # Check for newspaper3k
        try:
            from newspaper import Article

            self.has_newspaper = True
            self.newspaper = Article
        except ImportError:
            self.has_newspaper = False
            logger.warning("newspaper3k not available")

        # Check for Beautiful Soup (always should be available)
        try:
            from bs4 import BeautifulSoup

            self.has_bs4 = True
            self.BeautifulSoup = BeautifulSoup
        except ImportError:
            self.has_bs4 = False
            logger.error("BeautifulSoup not available - this is required")

    def crawl_urls(
        self, urls: List[str], query: str, max_results: int = 10
    ) -> List[WebCrawlResult]:
        """
        Crawl multiple URLs and extract relevant content.

        Args:
            urls: List of URLs to crawl
            query: Search query for relevance scoring
            max_results: Maximum number of results to return

        Returns:
            List of WebCrawlResult objects
        """
        logger.info(f"🌐 Starting web crawl of {len(urls)} URLs for query: '{query}'")

        all_results = []

        for url in urls:
            try:
                logger.info(f"🔍 Crawling: {url}")

                # Determine URL type and crawl accordingly
                if self._is_documentation_site(url):
                    results = self._crawl_documentation_site(url, query)
                elif self._is_portal_site(url):
                    results = self._crawl_portal_site(url, query)
                else:
                    results = self._crawl_general_website(url, query)

                all_results.extend(results)

                # Stop if we have enough results
                if len(all_results) >= max_results:
                    break

            except Exception as e:
                logger.error(f"Error crawling {url}: {str(e)}")
                continue

        # Sort by relevance and limit results
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)
        final_results = all_results[:max_results]

        logger.info(
            f"🎯 Web crawl completed: {len(final_results)} results from {len(urls)} URLs"
        )
        return final_results

    def _is_documentation_site(self, url: str) -> bool:
        """Check if URL appears to be a documentation site."""
        doc_indicators = [
            "docs.",
            "documentation.",
            "wiki.",
            "help.",
            "support.",
            "/docs/",
            "/documentation/",
            "/wiki/",
            "/help/",
            "/manual/",
            "readthedocs.io",
            "gitbook.io",
            "notion.so",
        ]
        return any(indicator in url.lower() for indicator in doc_indicators)

    def _is_portal_site(self, url: str) -> bool:
        """Check if URL appears to be a corporate portal or knowledge base."""
        portal_indicators = [
            "portal.",
            "kb.",
            "knowledge.",
            "intranet.",
            "confluence.",
            "/portal/",
            "/kb/",
            "/knowledge/",
            "/helpdesk/",
            "/servicedesk/",
        ]
        return any(indicator in url.lower() for indicator in portal_indicators)

    def _crawl_documentation_site(self, url: str, query: str) -> List[WebCrawlResult]:
        """Crawl documentation sites with specialized extraction."""
        results = []

        try:
            # Use trafilatura for clean documentation extraction
            if self.has_trafilatura:
                content = self._extract_with_trafilatura(url)
                if content:
                    result = self._create_web_result(
                        url, content, query, "documentation", "trafilatura"
                    )
                    if result:
                        results.append(result)

            # Fallback to Beautiful Soup for structured extraction
            if not results and self.has_bs4:
                content = self._extract_with_beautifulsoup(url, doc_focused=True)
                if content:
                    result = self._create_web_result(
                        url, content, query, "documentation", "beautifulsoup_doc"
                    )
                    if result:
                        results.append(result)

        except Exception as e:
            logger.error(f"Error crawling documentation site {url}: {str(e)}")

        return results

    def _crawl_portal_site(self, url: str, query: str) -> List[WebCrawlResult]:
        """Crawl corporate portals and knowledge bases."""
        results = []

        try:
            # Use requests-html for JavaScript-heavy portals
            if self.has_requests_html:
                content = self._extract_with_requests_html(url)
                if content:
                    result = self._create_web_result(
                        url, content, query, "portal", "requests_html"
                    )
                    if result:
                        results.append(result)

            # Fallback to newspaper for article-like content
            if not results and self.has_newspaper:
                content = self._extract_with_newspaper(url)
                if content:
                    result = self._create_web_result(
                        url, content, query, "portal", "newspaper"
                    )
                    if result:
                        results.append(result)

        except Exception as e:
            logger.error(f"Error crawling portal site {url}: {str(e)}")

        return results

    def _crawl_general_website(self, url: str, query: str) -> List[WebCrawlResult]:
        """Crawl general websites with multiple extraction methods."""
        results = []

        try:
            # Try multiple extraction methods and pick the best result
            extraction_methods = []

            # Method 1: Trafilatura (best for clean text)
            if self.has_trafilatura:
                content = self._extract_with_trafilatura(url)
                if content:
                    extraction_methods.append(("trafilatura", content))

            # Method 2: Newspaper (good for article content)
            if self.has_newspaper:
                content = self._extract_with_newspaper(url)
                if content:
                    extraction_methods.append(("newspaper", content))

            # Method 3: Beautiful Soup (fallback)
            if self.has_bs4:
                content = self._extract_with_beautifulsoup(url)
                if content:
                    extraction_methods.append(("beautifulsoup", content))

            # Create results from all successful extractions
            for method, content in extraction_methods:
                result = self._create_web_result(url, content, query, "general", method)
                if result and result.relevance_score > 0.2:
                    results.append(result)

            # Sort by relevance and take the best one
            if results:
                results.sort(key=lambda x: x.relevance_score, reverse=True)
                results = results[:1]  # Take only the best result per URL

        except Exception as e:
            logger.error(f"Error crawling general website {url}: {str(e)}")

        return results

    def _extract_with_trafilatura(self, url: str) -> Optional[str]:
        """Extract content using trafilatura."""
        try:
            import trafilatura

            response = self.session.get(url, timeout=self.settings.url_search_timeout)
            if response.status_code == 200:
                content = trafilatura.extract(
                    response.content,
                    include_links=True,
                    include_tables=True,
                    include_formatting=True,
                )
                return content
        except Exception as e:
            logger.debug(f"Trafilatura extraction failed for {url}: {str(e)}")
        return None

    def _extract_with_newspaper(self, url: str) -> Optional[str]:
        """Extract content using newspaper3k."""
        try:
            from newspaper import Article

            article = Article(url)
            article.download()
            article.parse()

            if article.text:
                # Combine title and text
                content = (
                    f"# {article.title}\n\n{article.text}"
                    if article.title
                    else article.text
                )
                return content
        except Exception as e:
            logger.debug(f"Newspaper extraction failed for {url}: {str(e)}")
        return None

    def _extract_with_requests_html(self, url: str) -> Optional[str]:
        """Extract content using requests-html (JavaScript support)."""
        try:
            from requests_html import HTMLSession

            session = HTMLSession()
            response = session.get(url, timeout=self.settings.url_search_timeout)

            if response.status_code == 200:
                # Render JavaScript if needed
                if self.settings.render_javascript:
                    response.html.render(timeout=10)

                # Extract text content
                text_content = response.html.text

                # Try to get a title
                title = ""
                title_elem = response.html.find("title", first=True)
                if title_elem:
                    title = title_elem.text

                if text_content:
                    content = f"# {title}\n\n{text_content}" if title else text_content
                    return content
        except Exception as e:
            logger.debug(f"Requests-html extraction failed for {url}: {str(e)}")
        return None

    def _extract_with_beautifulsoup(
        self, url: str, doc_focused: bool = False
    ) -> Optional[str]:
        """Extract content using Beautiful Soup."""
        try:
            from bs4 import BeautifulSoup

            response = self.session.get(url, timeout=self.settings.url_search_timeout)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")

                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()

                # Get title
                title = ""
                title_elem = soup.find("title")
                if title_elem:
                    title = title_elem.get_text().strip()

                # Extract main content
                if doc_focused:
                    # Look for documentation-specific containers
                    content_containers = soup.find_all(
                        ["main", "article", "div"],
                        class_=lambda x: x
                        and any(
                            term in str(x).lower()
                            for term in [
                                "content",
                                "documentation",
                                "docs",
                                "article",
                                "main",
                            ]
                        ),
                    )
                else:
                    # General content extraction
                    content_containers = soup.find_all(["main", "article", "div"])

                if content_containers:
                    text_content = " ".join(
                        [elem.get_text() for elem in content_containers[:3]]
                    )
                else:
                    text_content = soup.get_text()

                # Clean up text
                text_content = " ".join(text_content.split())

                if text_content:
                    content = f"# {title}\n\n{text_content}" if title else text_content
                    return content
        except Exception as e:
            logger.debug(f"BeautifulSoup extraction failed for {url}: {str(e)}")
        return None

    def _create_web_result(
        self, url: str, content: str, query: str, content_type: str, method: str
    ) -> Optional[WebCrawlResult]:
        """Create a WebCrawlResult from extracted content."""
        try:
            if not content or len(content) < 100:
                return None

            # Extract title from content
            lines = content.split("\n")
            title = lines[0].replace("#", "").strip() if lines else url

            # Calculate relevance
            relevance = self._calculate_relevance(content, query)

            # Extract sections (basic implementation)
            sections = self._extract_basic_sections(content)

            # Limit content length
            max_length = getattr(self.settings, "max_content_length", 5000)
            if len(content) > max_length:
                content = content[:max_length] + "..."

            return WebCrawlResult(
                url=url,
                title=title,
                content=content,
                extracted_sections=sections,
                links=[],  # Could be enhanced to extract links
                metadata={
                    "content_type": content_type,
                    "extraction_method": method,
                    "content_length": len(content),
                },
                relevance_score=relevance,
                content_type=content_type,
                extraction_method=method,
                timestamp=time.time(),
            )

        except Exception as e:
            logger.error(f"Error creating web result for {url}: {str(e)}")
            return None

    def _calculate_relevance(self, content: str, query: str) -> float:
        """Calculate relevance score between content and query."""
        if not content or not query:
            return 0.0

        content_lower = content.lower()
        query_lower = query.lower()
        query_words = query_lower.split()

        # Count query word matches
        matches = sum(1 for word in query_words if word in content_lower)
        if not query_words:
            return 0.0

        # Base relevance
        relevance = matches / len(query_words)

        # Boost for exact phrase matches
        if query_lower in content_lower:
            relevance += 0.3

        # Boost for title matches
        if content.startswith("#"):
            title_line = content.split("\n")[0].lower()
            title_matches = sum(1 for word in query_words if word in title_line)
            relevance += (title_matches / len(query_words)) * 0.2

        return min(relevance, 1.0)

    def _extract_basic_sections(self, content: str) -> Dict[str, str]:
        """Extract basic sections from content."""
        sections = {}

        lines = content.split("\n")
        current_section = "content"
        current_content = []

        for line in lines:
            line = line.strip()
            if line.startswith("#"):
                # Save previous section
                if current_content:
                    sections[current_section] = "\n".join(current_content)

                # Start new section
                current_section = line.replace("#", "").strip().lower()
                current_content = []
            else:
                current_content.append(line)

        # Save last section
        if current_content:
            sections[current_section] = "\n".join(current_content)

        return sections


def create_web_crawler() -> WebCrawler:
    """Factory function to create a web crawler instance."""
    return WebCrawler()

"""
URL Search Module for External Content Retrieval
Provides fallback search capabilities when local documents don't contain relevant information.
Supports GitHub repositories, documentation sites, and general web content.

MIT License - Copyright (c) 2025 talos-chatbot
"""

import logging
import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse, urljoin
import json

# HTTP client imports
import requests
import aiohttp
from bs4 import BeautifulSoup

from config import get_settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class URLSearchResult:
    """Container for URL search results."""

    def __init__(
        self,
        content: str,
        title: str,
        url: str,
        relevance_score: float,
        source_type: str = "web",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.content = content
        self.title = title
        self.url = url
        self.relevance_score = relevance_score
        self.source_type = source_type  # 'github', 'docs', 'web'
        self.metadata = metadata or {}


class ExternalURLSearcher:
    """
    Searches external URLs when local documents don't contain relevant information.
    Specializes in GitHub repositories, documentation sites, and web content.
    """

    def __init__(self):
        """Initialize the URL searcher with settings."""
        self.settings = get_settings()
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "RAG-Chatbot/1.0 (https://github.com/talos-chatbot)"}
        )

        # Configure default external URLs from settings
        self.default_urls = []

        # Add corporate portal URL if configured
        if self.settings.corporate_portal_url:
            self.default_urls.append(self.settings.corporate_portal_url)

        # Add configured external search URLs
        if (
            hasattr(self.settings, "external_search_urls")
            and self.settings.external_search_urls
        ):
            # Parse comma-separated URLs
            external_urls = [
                url.strip() for url in self.settings.external_search_urls.split(",")
            ]
            self.default_urls.extend(external_urls)
        else:
            # Fallback default
            self.default_urls.append("https://github.com/snort3/snort3")

        # Filter out None values and ensure URLs are valid
        self.default_urls = [
            url for url in self.default_urls if url and self._is_valid_url(url)
        ]

        # Get timeout from settings
        self.timeout = getattr(self.settings, "url_search_timeout", 10)

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and accessible."""
        try:
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False

    def search_external_sources(
        self,
        query: str,
        max_results: int = 3,
        additional_urls: Optional[List[str]] = None,
    ) -> List[URLSearchResult]:
        """
        Search external URLs for relevant information.

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            additional_urls: Additional URLs to search beyond defaults

        Returns:
            List of URLSearchResult objects
        """
        logger.info(f"🌐 Searching external URLs for: '{query}'")

        # Combine default URLs with additional ones
        search_urls = self.default_urls.copy()
        if additional_urls:
            search_urls.extend(
                [url for url in additional_urls if self._is_valid_url(url)]
            )

        all_results = []

        for url in search_urls[:5]:  # Limit to 5 URLs to avoid timeouts
            try:
                logger.info(f"🔍 Searching: {url}")

                if "github.com" in url:
                    results = self._search_github_repository(url, query, max_results)
                else:
                    results = self._search_general_website(url, query, max_results)

                all_results.extend(results)

                # Stop if we have enough results
                if len(all_results) >= max_results:
                    break

            except Exception as e:
                logger.warning(f"⚠️ Error searching {url}: {str(e)}")
                continue

        # Sort by relevance and limit results
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return all_results[:max_results]

    def _search_github_repository(
        self, repo_url: str, query: str, max_results: int
    ) -> List[URLSearchResult]:
        """
        Search a GitHub repository for relevant content.

        Args:
            repo_url: GitHub repository URL
            query: Search query
            max_results: Maximum results to return

        Returns:
            List of search results from the repository
        """
        results = []

        try:
            # Extract owner and repo from URL
            match = re.search(r"github\.com/([^/]+)/([^/]+)", repo_url)
            if not match:
                logger.warning(f"Could not parse GitHub URL: {repo_url}")
                return results

            owner, repo = match.groups()
            repo = repo.rstrip("/")
            logger.info(f"Searching GitHub repo: {owner}/{repo}")

            # First, search for specific documentation files based on query type
            doc_files_to_check = []
            query_lower = query.lower()

            if any(
                term in query_lower
                for term in ["install", "build", "compile", "download", "setup"]
            ):
                doc_files_to_check.extend(
                    [
                        "INSTALL.md",
                        "BUILD.md",
                        "BUILDING.md",
                        "COMPILE.md",
                        "SETUP.md",
                        "INSTALLATION.md",
                    ]
                )

            # Always check common documentation files (put README first since it's most likely to exist)
            doc_files_to_check = ["README.md"] + doc_files_to_check

            logger.info(f"Will check {len(doc_files_to_check)} documentation files")

            # Fetch documentation files
            for doc_file in doc_files_to_check[
                :10
            ]:  # Check more files but README first
                logger.info(f"Fetching {doc_file} from {owner}/{repo}")
                content = self._fetch_github_specific_file(owner, repo, doc_file)

                if content and len(content) > 100:  # Only include substantial content
                    relevance = self._calculate_relevance(content, query)
                    logger.info(f"{doc_file} relevance: {relevance:.3f}")

                    if relevance > 0.2:  # Include if somewhat relevant
                        # Extract more comprehensive content for installation/build queries
                        extracted_content = self._extract_relevant_sections(
                            content, query
                        )

                        result = URLSearchResult(
                            content=extracted_content[
                                :3000
                            ],  # More content for detailed docs
                            title=f"{doc_file} - {owner}/{repo}",
                            url=f"https://github.com/{owner}/{repo}/blob/main/{doc_file}",
                            relevance_score=relevance,
                            source_type="github",
                            metadata={
                                "repository": f"{owner}/{repo}",
                                "file_path": doc_file,
                                "file_type": "documentation",
                            },
                        )
                        results.append(result)
                        logger.info(
                            f"✅ Added {doc_file} to results (relevance: {relevance:.3f})"
                        )
                    else:
                        logger.info(
                            f"❌ Skipped {doc_file} - low relevance ({relevance:.3f})"
                        )
                else:
                    logger.info(f"❌ Skipped {doc_file} - no content or too short")

            logger.info(f"GitHub search completed with {len(results)} results")

        except Exception as e:
            logger.error(f"Error searching GitHub repository {repo_url}: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())

        return results

    def _fetch_github_file_content(self, download_url: str) -> Optional[str]:
        """Fetch content from a GitHub file download URL."""
        if not download_url:
            return None

        try:
            response = self.session.get(download_url, timeout=5)
            if response.status_code == 200:
                # Only process text files
                content_type = response.headers.get("content-type", "")
                if "text" in content_type or any(
                    ext in download_url.lower()
                    for ext in [".md", ".txt", ".py", ".js", ".html", ".cfg", ".conf"]
                ):
                    return response.text
        except Exception as e:
            logger.debug(f"Error fetching file content: {str(e)}")

        return None

    def _fetch_github_readme(self, owner: str, repo: str) -> Optional[str]:
        """Fetch README content from a GitHub repository."""
        try:
            # Try common README file names
            readme_files = ["README.md", "README.txt", "README.rst", "README"]

            for readme_file in readme_files:
                url = f"https://raw.githubusercontent.com/{owner}/{repo}/master/{readme_file}"
                response = self.session.get(url, timeout=5)

                if response.status_code == 200:
                    return response.text

                # Try main branch if master doesn't exist
                url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{readme_file}"
                response = self.session.get(url, timeout=5)

                if response.status_code == 200:
                    return response.text

        except Exception as e:
            logger.debug(f"Error fetching README: {str(e)}")

        return None

    def _fetch_github_specific_file(
        self, owner: str, repo: str, file_path: str
    ) -> Optional[str]:
        """Fetch specific file content from a GitHub repository."""
        try:
            # Try main branch first
            url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{file_path}"
            response = self.session.get(url, timeout=5)

            if response.status_code == 200:
                return response.text

            # Try master branch if main doesn't exist
            url = f"https://raw.githubusercontent.com/{owner}/{repo}/master/{file_path}"
            response = self.session.get(url, timeout=5)

            if response.status_code == 200:
                return response.text

        except Exception as e:
            logger.debug(f"Error fetching {file_path}: {str(e)}")

        return None

    def _extract_relevant_sections(self, content: str, query: str) -> str:
        """Extract the most relevant sections from documentation content."""
        if not content:
            return ""

        query_lower = query.lower()
        lines = content.split("\n")
        relevant_sections = []
        current_section = []
        section_relevance = 0

        # Common section headers to look for
        install_headers = [
            "install",
            "installation",
            "build",
            "building",
            "compile",
            "setup",
            "getting started",
        ]
        config_headers = ["configure", "configuration", "config", "setup"]

        for i, line in enumerate(lines):
            line_lower = line.lower()

            # Check if this is a header line (markdown headers or numbered sections)
            is_header = (
                line.startswith("#")
                or line.startswith("=")
                or (
                    line
                    and i > 0
                    and lines[i - 1].strip()
                    and lines[i - 1].strip().replace("=", "").replace("-", "").strip()
                    == ""
                )
            )

            if is_header:
                # Save previous section if it was relevant
                if current_section and section_relevance > 0:
                    relevant_sections.extend(current_section)

                # Start new section
                current_section = [line]

                # Calculate relevance of this section header
                section_relevance = 0
                if any(
                    term in query_lower
                    for term in ["install", "build", "download", "setup"]
                ):
                    if any(header in line_lower for header in install_headers):
                        section_relevance = 2

                if any(term in query_lower for term in ["configure", "config"]):
                    if any(header in line_lower for header in config_headers):
                        section_relevance = 2

                # General relevance check
                query_terms = [
                    term.strip()
                    for term in query_lower.split()
                    if len(term.strip()) > 2
                ]
                for term in query_terms:
                    if term in line_lower:
                        section_relevance = max(section_relevance, 1)
            else:
                current_section.append(line)

                # Check line relevance
                query_terms = [
                    term.strip()
                    for term in query_lower.split()
                    if len(term.strip()) > 2
                ]
                for term in query_terms:
                    if term in line_lower:
                        section_relevance = max(section_relevance, 1)

        # Add the last section if relevant
        if current_section and section_relevance > 0:
            relevant_sections.extend(current_section)

        # If no specific sections found, return the first part of the content
        if not relevant_sections:
            return content[:2000]

        return "\n".join(relevant_sections)

    def _search_general_website(
        self, url: str, query: str, max_results: int
    ) -> List[URLSearchResult]:
        """
        Search a general website for relevant content.

        Args:
            url: Website URL to search
            query: Search query
            max_results: Maximum results to return

        Returns:
            List of search results from the website
        """
        results = []

        try:
            # Fetch the main page
            response = self.session.get(url, timeout=self.timeout)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")

                # Extract main content (remove navigation, headers, footers)
                for element in soup(["nav", "header", "footer", "script", "style"]):
                    element.decompose()

                # Get text content
                content = soup.get_text(separator=" ", strip=True)

                # Calculate relevance
                relevance = self._calculate_relevance(content, query)

                if relevance > 0.2:  # Only include if relevant
                    # Get page title
                    title_tag = soup.find("title")
                    title = (
                        title_tag.text.strip() if title_tag else urlparse(url).netloc
                    )

                    result = URLSearchResult(
                        content=content[:2000],  # Limit content length
                        title=title,
                        url=url,
                        relevance_score=relevance,
                        source_type="web",
                        metadata={
                            "domain": urlparse(url).netloc,
                            "content_length": len(content),
                        },
                    )
                    results.append(result)

                # Look for internal links that might be relevant
                links = soup.find_all("a", href=True)
                for link in links[:10]:  # Limit to avoid too many requests
                    href = link.get("href")
                    link_text = link.get_text(strip=True)

                    # Check if link text is relevant to query
                    if self._is_text_relevant(link_text, query):
                        absolute_url = urljoin(url, href)
                        if absolute_url != url and self._is_valid_url(absolute_url):
                            # Fetch linked page
                            linked_content = self._fetch_page_content(absolute_url)
                            if linked_content:
                                link_relevance = self._calculate_relevance(
                                    linked_content, query
                                )
                                if link_relevance > 0.3:
                                    result = URLSearchResult(
                                        content=linked_content[:2000],
                                        title=f"{link_text} - {urlparse(url).netloc}",
                                        url=absolute_url,
                                        relevance_score=link_relevance,
                                        source_type="web",
                                        metadata={
                                            "domain": urlparse(url).netloc,
                                            "parent_url": url,
                                        },
                                    )
                                    results.append(result)

                                    if len(results) >= max_results:
                                        break

        except Exception as e:
            logger.error(f"Error searching website {url}: {str(e)}")

        return results

    def _fetch_page_content(self, url: str) -> Optional[str]:
        """Fetch and clean content from a web page."""
        try:
            response = self.session.get(url, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")

                # Remove unwanted elements
                for element in soup(
                    ["nav", "header", "footer", "script", "style", "aside"]
                ):
                    element.decompose()

                return soup.get_text(separator=" ", strip=True)
        except Exception as e:
            logger.debug(f"Error fetching page content: {str(e)}")

        return None

    def _calculate_relevance(self, content: str, query: str) -> float:
        """
        Calculate relevance score between content and query.

        Args:
            content: Text content to score
            query: Search query

        Returns:
            Relevance score between 0 and 1
        """
        if not content or not query:
            return 0.0

        # Convert to lowercase for comparison
        content_lower = content.lower()
        query_lower = query.lower()

        # Split query into terms
        query_terms = [
            term.strip() for term in query_lower.split() if len(term.strip()) > 2
        ]

        if not query_terms:
            return 0.0

        # Calculate term frequency and coverage
        total_score = 0.0
        content_words = content_lower.split()
        content_word_count = len(content_words)

        if content_word_count == 0:
            return 0.0

        for term in query_terms:
            # Exact term matches
            exact_matches = content_lower.count(term)
            term_score = min(exact_matches / content_word_count * 100, 1.0)

            # Partial matches (for compound terms)
            if len(term) > 4:
                partial_matches = sum(1 for word in content_words if term in word)
                partial_score = min(partial_matches / content_word_count * 50, 0.5)
                term_score += partial_score

            total_score += term_score

        # Normalize by number of terms
        relevance = min(total_score / len(query_terms), 1.0)

        return relevance

    def _is_text_relevant(self, text: str, query: str) -> bool:
        """Check if text is relevant to the query."""
        if not text or not query:
            return False

        text_lower = text.lower()
        query_terms = [
            term.strip().lower() for term in query.split() if len(term.strip()) > 2
        ]

        # Check if any query term appears in the text
        for term in query_terms:
            if term in text_lower:
                return True

        return False


def create_url_searcher() -> ExternalURLSearcher:
    """Factory function to create a URL searcher instance."""
    return ExternalURLSearcher()

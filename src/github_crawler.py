"""
Advanced Web Crawler Module
Uses multiple libraries for comprehensive content extraction from external URLs.
Supports GitHub repositories, documentation sites, and general web content.

MIT License - Copyright (c) 2025 talos-chatbot
"""

import logging
import re
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple, Set
from urllib.parse import urlparse, urljoin, quote
from dataclasses import dataclass
import json

# Standard HTTP clients
import requests
import aiohttp

# Advanced content extraction libraries
try:
    from requests_html import HTMLSession

    REQUESTS_HTML_AVAILABLE = True
except ImportError:
    REQUESTS_HTML_AVAILABLE = False

try:
    import trafilatura

    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False

try:
    from newspaper import Article

    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False

# HTML parsing
from bs4 import BeautifulSoup
import lxml

# Configuration
from config import get_settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CrawlResult:
    """Container for crawl results with rich metadata."""

    url: str
    title: str
    content: str
    extracted_sections: Dict[str, str]  # section_name -> content
    links: List[Dict[str, str]]  # internal links found
    metadata: Dict[str, Any]
    relevance_score: float
    content_type: str  # 'github', 'documentation', 'article', 'general'
    extraction_method: str  # which library was used
    timestamp: float


class GitHubCrawler:
    """
    GitHub repository crawler using multiple extraction libraries for comprehensive content retrieval.
    Specialized for GitHub repositories, documentation files, and technical documentation.
    """

    def __init__(self):
        """Initialize the advanced crawler with multiple extraction backends."""
        self.settings = get_settings()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "RAG-Chatbot-Advanced/1.0 (https://github.com/talos-chatbot; Educational Purpose)"
            }
        )

        # HTML session for JavaScript rendering
        if REQUESTS_HTML_AVAILABLE:
            self.html_session = HTMLSession()
        else:
            self.html_session = None

        # Configure timeouts and limits (optimized for speed)
        self.timeout = getattr(
            self.settings, "url_search_timeout", 10
        )  # Reduced for faster responses
        self.max_crawl_depth = 2
        self.max_links_per_page = 10
        self.crawled_urls: Set[str] = set()

        # Add simple cache for faster responses
        self._repo_cache = {}
        self._cache_timeout = 300  # 5 minutes

        logger.info(
            f"Advanced crawler initialized with libraries: "
            f"requests-html={REQUESTS_HTML_AVAILABLE}, "
            f"trafilatura={TRAFILATURA_AVAILABLE}, "
            f"newspaper={NEWSPAPER_AVAILABLE}"
        )

    def crawl_urls(
        self, urls: List[str], query: str, max_results: int = 5
    ) -> List[CrawlResult]:
        """
        Crawl multiple URLs with comprehensive content extraction.

        Args:
            urls: List of URLs to crawl
            query: Search query for relevance scoring
            max_results: Maximum number of results to return

        Returns:
            List of CrawlResult objects sorted by relevance
        """
        logger.info(
            f"🕷️ Starting advanced crawl of {len(urls)} URLs for query: '{query}'"
        )

        all_results = []
        self.crawled_urls.clear()

        for url in urls:
            try:
                logger.info(f"🔍 Crawling: {url}")

                # Determine URL type for specialized handling
                url_type = self._classify_url(url)

                if url_type == "github":
                    results = self._crawl_github_repository(url, query)
                elif url_type == "documentation":
                    results = self._crawl_documentation_site(url, query)
                else:
                    results = self._crawl_general_website(url, query)

                all_results.extend(results)

                # For GitHub repos, don't stop early to ensure we get diverse documentation
                if url_type != "github":
                    # Stop if we have enough high-quality results (only for non-GitHub sites)
                    high_quality_results = [
                        r for r in all_results if r.relevance_score > 0.5
                    ]
                    if len(high_quality_results) >= max_results:
                        logger.info(
                            f"✅ Found {len(high_quality_results)} high-quality results, stopping crawl"
                        )
                        break

            except Exception as e:
                logger.error(f"❌ Error crawling {url}: {str(e)}")
                continue

        # Sort by relevance and return top results
        all_results.sort(key=lambda x: x.relevance_score, reverse=True)
        final_results = all_results[:max_results]

        logger.info(
            f"🎯 Crawl completed: {len(final_results)} results from {len(self.crawled_urls)} URLs"
        )
        return final_results

    def _classify_url(self, url: str) -> str:
        """Classify URL type for specialized handling."""
        url_lower = url.lower()

        if "github.com" in url_lower:
            return "github"
        elif any(
            term in url_lower
            for term in ["docs", "documentation", "wiki", "readthedocs"]
        ):
            return "documentation"
        else:
            return "general"

    def _crawl_github_repository(self, repo_url: str, query: str) -> List[CrawlResult]:
        """Enhanced GitHub repository crawler with deep content extraction."""
        results = []

        try:
            # Extract owner and repo from URL
            match = re.search(r"github\.com/([^/]+)/([^/]+)", repo_url)
            if not match:
                return results

            owner, repo = match.groups()
            repo = repo.rstrip("/")

            logger.info(f"📂 Crawling GitHub repo: {owner}/{repo}")

            # 1. Crawl main repository page
            main_page_result = self._extract_github_main_page(owner, repo, query)
            if main_page_result:
                results.append(main_page_result)

            # 2. Crawl specific documentation files
            doc_results = self._crawl_github_documentation(owner, repo, query)
            results.extend(doc_results)

            # 3. Crawl wiki if available
            wiki_results = self._crawl_github_wiki(owner, repo, query)
            results.extend(wiki_results)

            # 4. Fast comprehensive file discovery (limited scope)
            if len(results) < 3:  # Only if we need more results
                comprehensive_results = self._crawl_all_repository_files(
                    owner, repo, query, max_files=50
                )
                results.extend(comprehensive_results)

            # Early termination if we have enough good results
            if len(results) >= 5:
                logger.info(f"🎯 Early termination: found {len(results)} results")
                return results

            # 5. Search for relevant files via GitHub API (only if still need more)
            if len(results) < 4:
                api_results = self._search_github_api_content(owner, repo, query)
                results.extend(api_results)

        except Exception as e:
            logger.error(f"Error crawling GitHub repository {repo_url}: {str(e)}")

        return results

    def _search_github_api_content(
        self, owner: str, repo: str, query: str
    ) -> List[CrawlResult]:
        """Search for relevant files using GitHub API based on query keywords."""
        results = []

        try:
            # Define search terms based on query
            query_lower = query.lower()
            search_terms = []

            if any(term in query_lower for term in ["upgrade", "updating", "migrate"]):
                search_terms = ["upgrade", "migration", "changelog", "changes"]
            elif any(term in query_lower for term in ["plugin", "plugins", "module"]):
                search_terms = ["plugin", "module", "extension"]
            elif any(term in query_lower for term in ["install", "build", "setup"]):
                search_terms = ["install", "build", "setup", "configure"]

            if not search_terms:
                return results

            # Search repository contents via GitHub API
            for term in search_terms[:3]:  # Limit API calls
                try:
                    import requests

                    api_url = f"https://api.github.com/search/code"
                    params = {"q": f"{term} repo:{owner}/{repo}", "per_page": 10}

                    response = requests.get(api_url, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()

                        for item in data.get("items", [])[:5]:  # Limit results
                            file_path = item.get("path", "")
                            if any(
                                ext in file_path.lower()
                                for ext in [".md", ".txt", ".rst"]
                            ):
                                # Fetch the file content
                                content = self._fetch_github_raw_file(
                                    owner, repo, file_path
                                )
                                if content and len(content) > 100:
                                    processed_content = (
                                        self._process_documentation_content(
                                            content, query
                                        )
                                    )
                                    sections = self._extract_markdown_sections(content)
                                    relevance = self._calculate_relevance(
                                        processed_content, query
                                    )

                                    if relevance > 0.3:
                                        import time

                                        result = CrawlResult(
                                            url=f"https://github.com/{owner}/{repo}/blob/main/{file_path}",
                                            title=f"{file_path} - {owner}/{repo}",
                                            content=processed_content,
                                            extracted_sections=sections,
                                            links=[],  # No links extracted for now
                                            metadata={
                                                "repository": f"{owner}/{repo}",
                                                "file_path": file_path,
                                                "search_term": term,
                                                "type": "github_documentation",
                                            },
                                            relevance_score=relevance,
                                            content_type="github",
                                            extraction_method="api_search",
                                            timestamp=time.time(),
                                        )
                                        results.append(result)

                except Exception as e:
                    logger.debug(f"API search failed for term '{term}': {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Error in GitHub API content search: {str(e)}")

        return results

    def _crawl_all_repository_files(
        self, owner: str, repo: str, query: str, max_files: int = 138
    ) -> List[CrawlResult]:
        """Comprehensively crawl all files in a GitHub repository."""
        results = []

        try:
            logger.info(
                f"🔍 Comprehensive repository file discovery for {owner}/{repo}"
            )

            # Generate comprehensive file paths based on common repository structures
            candidate_files = self._generate_comprehensive_file_paths(query)

            # Limit files to process for speed
            candidate_files = candidate_files[:max_files]

            logger.info(
                f"📁 Checking {len(candidate_files)} potential documentation files (limited to {max_files})"
            )

            processed_count = 0
            found_count = 0

            # Process each candidate file
            for file_path in candidate_files:
                try:
                    content = self._fetch_github_raw_file(owner, repo, file_path)
                    processed_count += 1

                    if content and len(content) > 100:  # Only substantial content
                        processed_content = self._process_documentation_content(
                            content, query
                        )
                        sections = self._extract_markdown_sections(content)
                        relevance = self._calculate_relevance(processed_content, query)

                        if (
                            relevance > 0.2
                        ):  # Lower threshold for comprehensive crawling
                            import time

                            result = CrawlResult(
                                url=f"https://github.com/{owner}/{repo}/blob/main/{file_path}",
                                title=f"{file_path} - {owner}/{repo}",
                                content=processed_content,
                                extracted_sections=sections,
                                links=[],  # No links extracted for now
                                metadata={
                                    "repository": f"{owner}/{repo}",
                                    "file_path": file_path,
                                    "type": "github_comprehensive",
                                },
                                relevance_score=relevance,
                                content_type="github",
                                extraction_method="comprehensive_crawl",
                                timestamp=time.time(),
                            )
                            results.append(result)
                            found_count += 1
                            logger.info(
                                f"📄 Added {file_path} (relevance: {relevance:.3f})"
                            )

                            # Query-specific early termination for better speed
                            query_lower = query.lower()
                            is_build_query = any(
                                term in query_lower
                                for term in [
                                    "build",
                                    "install",
                                    "setup",
                                    "compile",
                                    "download",
                                ]
                            )

                            if is_build_query and any(
                                build_file in file_path.lower()
                                for build_file in ["build", "install", "setup"]
                            ):
                                if (
                                    found_count >= 3
                                ):  # Earlier exit for build queries with relevant files
                                    logger.info(
                                        f"🎯 Quick exit: found {found_count} build-specific files"
                                    )
                                    break
                            elif found_count >= 10:  # Regular early termination
                                logger.info(
                                    f"🚀 Early termination: found {found_count} relevant files"
                                )
                                break

                except Exception as e:
                    # File doesn't exist or can't be accessed - this is normal
                    continue

            logger.info(
                f"📊 Checked {processed_count} files, found {found_count} relevant documents"
            )

        except Exception as e:
            logger.error(f"Error in comprehensive repository crawling: {str(e)}")

        return results

    def _get_all_repository_files(
        self, owner: str, repo: str, path: str = ""
    ) -> List[str]:
        """Get repository files using GitHub search API to avoid rate limits."""
        all_files = []

        try:
            import requests

            # Use GitHub search API instead of contents API to avoid rate limits
            # Search for different file types and common documentation patterns
            search_queries = [
                f"repo:{owner}/{repo} extension:md",
                f"repo:{owner}/{repo} extension:txt",
                f"repo:{owner}/{repo} extension:rst",
                f"repo:{owner}/{repo} filename:README",
                f"repo:{owner}/{repo} filename:INSTALL",
                f"repo:{owner}/{repo} filename:BUILD",
                f"repo:{owner}/{repo} filename:CHANGELOG",
                f"repo:{owner}/{repo} path:doc",
                f"repo:{owner}/{repo} path:docs",
                f"repo:{owner}/{repo} upgrade",
                f"repo:{owner}/{repo} plugin",
                f"repo:{owner}/{repo} install",
            ]

            for query in search_queries:
                try:
                    api_url = "https://api.github.com/search/code"
                    params = {"q": query, "per_page": 30}

                    response = requests.get(api_url, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        for item in data.get("items", []):
                            file_path = item.get("path", "")
                            if file_path and file_path not in all_files:
                                all_files.append(file_path)
                    elif response.status_code == 403:
                        logger.warning(
                            "GitHub API rate limit reached, stopping comprehensive search"
                        )
                        break

                except Exception as e:
                    logger.debug(f"Search query failed: {query}, error: {str(e)}")
                    continue

        except Exception as e:
            logger.debug(f"Error in repository file search: {str(e)}")

        return all_files

    def _generate_comprehensive_file_paths(self, query: str) -> List[str]:
        """Generate comprehensive list of potential documentation file paths."""
        query_lower = query.lower()

        # Base documentation files (prioritized for build queries)
        base_files = [
            "BUILD.md",
            "BUILD.txt",
            "BUILDING.md",
            "INSTALL.md",
            "INSTALL.txt",
            "INSTALL",
            "SETUP.md",
            "SETUP.txt",
            "CONFIGURE.md",
            "CONFIGURATION.md",
            "README.md",  # Moved down as it's often too general
            "README.txt",
            "README.rst",
            "CHANGELOG.md",
            "CHANGELOG.txt",
            "ChangeLog.md",
            "CHANGES.md",
            "CHANGES.txt",
            "NEWS.md",
            "NEWS.txt",
            "RELEASE.md",
            "RELEASE.txt",
            "TODO.md",
            "TODO.txt",
            "USAGE.md",
            "USAGE.txt",
            "TUTORIAL.md",
            "TUTORIAL.txt",
            "GUIDE.md",
            "GUIDE.txt",
            "MANUAL.md",
            "MANUAL.txt",
            "FAQ.md",
            "FAQ.txt",
            "CONTRIBUTING.md",
            "CONTRIBUTING.txt",
            "DEVELOPMENT.md",
            "DEVELOPMENT.txt",
            "LICENSE",
            "LICENSE.md",
            "LICENSE.txt",
            "COPYING",
            "COPYRIGHT",
        ]

        # Documentation directories and their common files
        doc_patterns = [
            # doc/ directory
            "doc/README.md",
            "doc/README.txt",
            "doc/index.md",
            "doc/index.txt",
            "doc/install.md",
            "doc/install.txt",
            "doc/installation.md",
            "doc/build.md",
            "doc/build.txt",
            "doc/building.md",
            "doc/setup.md",
            "doc/setup.txt",
            "doc/usage.md",
            "doc/usage.txt",
            "doc/manual.md",
            "doc/tutorial.md",
            "doc/tutorial.txt",
            "doc/guide.md",
            "doc/faq.md",
            "doc/faq.txt",
            # docs/ directory
            "docs/README.md",
            "docs/README.txt",
            "docs/index.md",
            "docs/index.txt",
            "docs/install.md",
            "docs/install.txt",
            "docs/installation.md",
            "docs/build.md",
            "docs/build.txt",
            "docs/building.md",
            "docs/setup.md",
            "docs/setup.txt",
            "docs/usage.md",
            "docs/usage.txt",
            "docs/manual.md",
            "docs/tutorial.md",
            "docs/tutorial.txt",
            "docs/guide.md",
            "docs/faq.md",
            "docs/faq.txt",
            # Common subdirectories in doc/
            "doc/user/README.md",
            "doc/user/README.txt",
            "doc/user/manual.md",
            "doc/user/tutorial.md",
            "doc/user/guide.md",
            "doc/user/usage.md",
            "doc/user/concepts.md",
            "doc/user/concepts.txt",
            "doc/user/plugins.md",
            "doc/user/plugins.txt",
            "doc/user/modules.md",
            "doc/user/modules.txt",
            "doc/user/config.md",
            "doc/user/config.txt",
            "doc/user/configuration.md",
            "doc/user/configuration.txt",
            "doc/admin/README.md",
            "doc/admin/README.txt",
            "doc/admin/manual.md",
            "doc/admin/install.md",
            "doc/admin/installation.md",
            "doc/admin/setup.md",
            "doc/admin/configuration.md",
            "doc/devel/README.md",
            "doc/devel/README.txt",
            "doc/devel/manual.md",
            "doc/devel/development.md",
            "doc/devel/contributing.md",
            "doc/devel/building.md",
            "doc/devel/extending.md",
            "doc/reference/README.md",
            "doc/reference/README.txt",
            "doc/reference/manual.md",
            "doc/reference/api.md",
            "doc/reference/reference.md",
        ]

        # Query-specific files
        query_specific = []

        if any(term in query_lower for term in ["upgrade", "migration", "update"]):
            query_specific.extend(
                [
                    "UPGRADE.md",
                    "UPGRADE.txt",
                    "UPGRADING.md",
                    "UPGRADING.txt",
                    "MIGRATION.md",
                    "MIGRATION.txt",
                    "MIGRATING.md",
                    "doc/upgrade/README.md",
                    "doc/upgrade/README.txt",
                    "doc/upgrade/overview.md",
                    "doc/upgrade/overview.txt",
                    "doc/upgrade/upgrade.md",
                    "doc/upgrade/upgrade.txt",
                    "doc/upgrade/snort_upgrade.md",
                    "doc/upgrade/snort_upgrade.txt",
                    "doc/upgrade/snort_upgrade.text",
                    "doc/upgrade/differences.md",
                    "doc/upgrade/differences.txt",
                    "doc/upgrade/config_changes.md",
                    "doc/upgrade/config_changes.txt",
                    "doc/upgrade/snort2lua.md",
                    "doc/upgrade/snort2lua.txt",
                    "doc/upgrade/migration.md",
                    "doc/upgrade/migration.txt",
                    "doc/migration/README.md",
                    "doc/migration/guide.md",
                    "docs/upgrade/README.md",
                    "docs/upgrade/guide.md",
                    "docs/migration/README.md",
                    "docs/migration/guide.md",
                ]
            )

        if any(
            term in query_lower for term in ["plugin", "plugins", "module", "modules"]
        ):
            query_specific.extend(
                [
                    "PLUGINS.md",
                    "PLUGINS.txt",
                    "MODULES.md",
                    "MODULES.txt",
                    "doc/plugins/README.md",
                    "doc/plugins/README.txt",
                    "doc/plugins/guide.md",
                    "doc/plugins/guide.txt",
                    "doc/plugins/tutorial.md",
                    "doc/plugins/tutorial.txt",
                    "doc/plugins/development.md",
                    "doc/plugins/api.md",
                    "doc/modules/README.md",
                    "doc/modules/README.txt",
                    "doc/modules/guide.md",
                    "doc/modules/guide.txt",
                    "doc/user/plugins.md",
                    "doc/user/plugins.txt",
                    "doc/user/modules.md",
                    "doc/user/modules.txt",
                    "doc/user/active.md",
                    "doc/user/active.txt",
                    "doc/user/appid.md",
                    "doc/user/appid.txt",
                    "doc/user/binder.md",
                    "doc/user/binder.txt",
                    "docs/plugins/README.md",
                    "docs/plugins/guide.md",
                    "docs/modules/README.md",
                    "docs/modules/guide.md",
                ]
            )

        if any(
            term in query_lower
            for term in ["install", "installation", "build", "setup"]
        ):
            query_specific.extend(
                [
                    "doc/install/README.md",
                    "doc/install/README.txt",
                    "doc/install/guide.md",
                    "doc/install/guide.txt",
                    "doc/installation/README.md",
                    "doc/installation/guide.md",
                    "doc/build/README.md",
                    "doc/build/README.txt",
                    "doc/build/guide.md",
                    "doc/build/guide.txt",
                    "doc/setup/README.md",
                    "doc/setup/guide.md",
                    "docs/install/README.md",
                    "docs/install/guide.md",
                    "docs/installation/README.md",
                    "docs/installation/guide.md",
                    "docs/build/README.md",
                    "docs/build/guide.md",
                    "docs/setup/README.md",
                    "docs/setup/guide.md",
                ]
            )

        # Combine all file paths and remove duplicates
        all_paths = base_files + doc_patterns + query_specific
        return list(
            dict.fromkeys(all_paths)
        )  # Remove duplicates while preserving order

    def _filter_documentation_files(
        self, all_files: List[str], query: str
    ) -> List[str]:
        """Filter files to focus on documentation and relevant content."""
        doc_files = []
        query_lower = query.lower()

        # Define file extensions and patterns for documentation
        doc_extensions = [".md", ".txt", ".rst", ".adoc", ".asciidoc", ".org", ".wiki"]
        doc_patterns = [
            "readme",
            "install",
            "build",
            "setup",
            "config",
            "upgrade",
            "migration",
            "tutorial",
            "guide",
            "manual",
            "howto",
            "faq",
            "changelog",
            "changes",
            "news",
            "release",
            "todo",
            "bug",
            "plugin",
            "module",
            "component",
            "api",
            "usage",
            "example",
        ]

        # Query-specific patterns
        query_patterns = []
        if any(term in query_lower for term in ["upgrade", "migration", "update"]):
            query_patterns.extend(
                ["upgrade", "migration", "changelog", "changes", "release"]
            )
        if any(term in query_lower for term in ["plugin", "plugins", "module"]):
            query_patterns.extend(["plugin", "module", "component", "extension"])
        if any(term in query_lower for term in ["install", "build", "setup"]):
            query_patterns.extend(["install", "build", "setup", "configure", "compile"])
        if any(term in query_lower for term in ["usage", "how", "tutorial"]):
            query_patterns.extend(["usage", "tutorial", "guide", "example", "howto"])

        for file_path in all_files:
            file_lower = file_path.lower()

            # Check file extension
            has_doc_extension = any(file_lower.endswith(ext) for ext in doc_extensions)

            # Check for documentation patterns in filename or path
            has_doc_pattern = any(pattern in file_lower for pattern in doc_patterns)

            # Check for query-specific patterns
            has_query_pattern = any(pattern in file_lower for pattern in query_patterns)

            # Include if it matches criteria
            if has_doc_extension or has_doc_pattern or has_query_pattern:
                doc_files.append(file_path)

        # Sort by relevance to query
        def file_relevance(file_path):
            score = 0
            file_lower = file_path.lower()

            # Higher score for query-specific patterns
            for pattern in query_patterns:
                if pattern in file_lower:
                    score += 3

            # Medium score for general doc patterns
            for pattern in doc_patterns:
                if pattern in file_lower:
                    score += 1

            # Bonus for being in doc directories
            if any(
                dir_name in file_lower
                for dir_name in ["doc/", "docs/", "documentation/"]
            ):
                score += 2

            # Bonus for common important files
            if any(
                important in file_lower
                for important in ["readme", "install", "changelog"]
            ):
                score += 2

            return score

        doc_files.sort(key=file_relevance, reverse=True)
        return doc_files

    def _extract_github_main_page(
        self, owner: str, repo: str, query: str
    ) -> Optional[CrawlResult]:
        """Extract content from GitHub repository main page."""
        try:
            url = f"https://github.com/{owner}/{repo}"
            response = self.session.get(url, timeout=self.timeout)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")

                # Extract repository description and README
                description = ""
                desc_elem = soup.find("p", class_="f4 my-3")
                if desc_elem:
                    description = desc_elem.get_text(strip=True)

                # Extract comprehensive README content from the rendered markdown
                readme_content = ""
                readme_elem = soup.find("article", class_="markdown-body")
                if readme_elem:
                    # Get the structured markdown content preserving formatting
                    readme_content = self._extract_structured_github_readme(readme_elem)

                # Extract topics/tags
                topics = []
                topic_elems = soup.find_all("a", class_="topic-tag")
                for topic in topic_elems:
                    topics.append(topic.get_text(strip=True))

                # Extract additional repository information
                repo_info = self._extract_github_repo_info(soup)

                # Build comprehensive content
                full_content = f"# {owner}/{repo}\n\n"
                if description:
                    full_content += f"**Description:** {description}\n\n"
                if topics:
                    full_content += f"**Topics:** {', '.join(topics)}\n\n"
                if repo_info:
                    full_content += f"**Repository Information:**\n{repo_info}\n\n"
                if readme_content:
                    full_content += f"**README Content:**\n\n{readme_content}"

                # Extract sections from the structured content
                sections = self._extract_comprehensive_sections(readme_content, soup)

                relevance = self._calculate_relevance(full_content, query)

                return CrawlResult(
                    url=url,
                    title=f"GitHub Repository: {owner}/{repo}",
                    content=full_content,
                    extracted_sections=sections,
                    links=self._extract_github_links(soup, url),
                    metadata={
                        "repository": f"{owner}/{repo}",
                        "description": description,
                        "topics": topics,
                        "type": "github_main",
                        "additional_info": repo_info,
                    },
                    relevance_score=relevance,
                    content_type="github",
                    extraction_method="enhanced_github",
                    timestamp=time.time(),
                )

        except Exception as e:
            logger.error(f"Error extracting GitHub main page: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())

        return None

    def _crawl_github_documentation(
        self, owner: str, repo: str, query: str
    ) -> List[CrawlResult]:
        """Crawl GitHub repository documentation files."""
        results = []

        # List of common documentation files to check
        doc_files = [
            "README.md",
            "INSTALL.md",
            "BUILD.md",
            "BUILDING.md",
            "SETUP.md",
            "CONFIGURE.md",
            "CONFIGURATION.md",
            "CONTRIBUTING.md",
            "DEVELOPMENT.md",
            "USAGE.md",
            "GETTING_STARTED.md",
            "QUICKSTART.md",
            "TUTORIAL.md",
            "ChangeLog.md",
            "CHANGELOG.md",
            "docs/README.md",
            "docs/index.md",
            "docs/installation.md",
            "docs/building.md",
            "doc/README.md",
            "doc/install.md",
            "doc/build.md",
            # Snort3-specific documentation paths
            "doc/upgrade/overview.txt",
            "doc/upgrade/snort_upgrade.txt",
            "doc/upgrade/differences.txt",
            "doc/upgrade/config_changes.txt",
            "doc/upgrade/snort2lua.txt",
            "doc/user/concepts.txt",
            "doc/user/plugins.txt",
            "doc/user/active.txt",
            "doc/user/appid.txt",
            "doc/user/binder.txt",
        ]

        query_lower = query.lower()

        # Prioritize files based on query content
        prioritized_files = []
        if any(term in query_lower for term in ["upgrade", "updating", "migrate"]):
            prioritized_files = [
                "doc/upgrade/overview.txt",
                "doc/upgrade/snort_upgrade.txt",
                "doc/upgrade/differences.txt",
                "doc/upgrade/config_changes.txt",
                "doc/upgrade/snort2lua.txt",
                "ChangeLog.md",
                "CHANGELOG.md",
            ]
        elif any(
            term in query_lower for term in ["plugin", "plugins", "module", "modules"]
        ):
            prioritized_files = [
                "doc/user/plugins.txt",
                "doc/user/concepts.txt",
                "doc/user/active.txt",
                "doc/user/appid.txt",
                "doc/user/binder.txt",
            ]
        elif any(
            term in query_lower for term in ["install", "build", "setup", "compile"]
        ):
            prioritized_files = [
                "INSTALL.md",
                "BUILD.md",
                "BUILDING.md",
                "SETUP.md",
                "docs/installation.md",
                "docs/building.md",
            ]
        elif any(term in query_lower for term in ["configure", "config"]):
            prioritized_files = [
                "CONFIGURE.md",
                "CONFIGURATION.md",
                "docs/configuration.md",
                "doc/upgrade/config_changes.txt",
            ]
        elif any(term in query_lower for term in ["usage", "use", "tutorial"]):
            prioritized_files = [
                "USAGE.md",
                "TUTORIAL.md",
                "GETTING_STARTED.md",
                "docs/usage.md",
                "doc/user/concepts.txt",
            ]

        # Combine prioritized and regular files
        all_files = prioritized_files + [
            f for f in doc_files if f not in prioritized_files
        ]

        for doc_file in all_files[:25]:  # Increased limit for better coverage
            content = self._fetch_github_raw_file(owner, repo, doc_file)
            if content and len(content) > 200:  # Only substantial content
                # Enhanced content extraction and processing
                processed_content = self._process_documentation_content(content, query)
                sections = self._extract_markdown_sections(content)
                relevance = self._calculate_relevance(processed_content, query)

                if relevance > 0.3:  # Only include relevant content
                    result = CrawlResult(
                        url=f"https://github.com/{owner}/{repo}/blob/main/{doc_file}",
                        title=f"{doc_file} - {owner}/{repo}",
                        content=processed_content,
                        extracted_sections=sections,
                        links=self._extract_internal_links(
                            content, f"https://github.com/{owner}/{repo}"
                        ),
                        metadata={
                            "repository": f"{owner}/{repo}",
                            "file_path": doc_file,
                            "file_size": len(content),
                            "type": "github_documentation",
                        },
                        relevance_score=relevance,
                        content_type="github",
                        extraction_method="raw_github",
                        timestamp=time.time(),
                    )
                    results.append(result)
                    logger.info(f"📄 Added {doc_file} (relevance: {relevance:.3f})")

        return results

    def _crawl_documentation_site(self, url: str, query: str) -> List[CrawlResult]:
        """Crawl documentation websites with advanced content extraction."""
        results = []

        try:
            logger.info(f"📚 Crawling documentation site: {url}")

            # Method 1: Try trafilatura for content extraction
            if TRAFILATURA_AVAILABLE:
                extracted_content = trafilatura.extract(
                    url, include_links=True, include_tables=True
                )
                if extracted_content:
                    sections = self._extract_structured_sections(extracted_content)
                    processed_content = self._process_documentation_content(
                        extracted_content, query
                    )
                    relevance = self._calculate_relevance(processed_content, query)

                    result = CrawlResult(
                        url=url,
                        title=f"Documentation: {urlparse(url).netloc}",
                        content=processed_content,
                        extracted_sections=sections,
                        links=[],
                        metadata={
                            "domain": urlparse(url).netloc,
                            "extraction_method": "trafilatura",
                            "type": "documentation",
                        },
                        relevance_score=relevance,
                        content_type="documentation",
                        extraction_method="trafilatura",
                        timestamp=time.time(),
                    )
                    results.append(result)
                    logger.info(
                        f"📖 Extracted with trafilatura (relevance: {relevance:.3f})"
                    )

            # Method 2: Try newspaper3k for article extraction
            if NEWSPAPER_AVAILABLE and len(results) == 0:
                try:
                    article = Article(url)
                    article.download()
                    article.parse()

                    if article.text:
                        processed_content = self._process_documentation_content(
                            article.text, query
                        )
                        relevance = self._calculate_relevance(processed_content, query)

                        result = CrawlResult(
                            url=url,
                            title=article.title
                            or f"Documentation: {urlparse(url).netloc}",
                            content=processed_content,
                            extracted_sections={"main_content": article.text},
                            links=[],
                            metadata={
                                "domain": urlparse(url).netloc,
                                "authors": article.authors,
                                "publish_date": str(article.publish_date)
                                if article.publish_date
                                else None,
                                "type": "documentation",
                            },
                            relevance_score=relevance,
                            content_type="documentation",
                            extraction_method="newspaper3k",
                            timestamp=time.time(),
                        )
                        results.append(result)
                        logger.info(
                            f"📰 Extracted with newspaper3k (relevance: {relevance:.3f})"
                        )
                except Exception as e:
                    logger.debug(f"Newspaper3k extraction failed: {str(e)}")

            # Method 3: Fallback to requests + BeautifulSoup
            if len(results) == 0:
                results.extend(self._crawl_general_website(url, query))

        except Exception as e:
            logger.error(f"Error crawling documentation site {url}: {str(e)}")

        return results

    def _crawl_general_website(self, url: str, query: str) -> List[CrawlResult]:
        """Crawl general websites with comprehensive extraction."""
        results = []

        try:
            logger.info(f"🌐 Crawling general website: {url}")

            # Use requests-html if available for JavaScript rendering
            if self.html_session and REQUESTS_HTML_AVAILABLE:
                try:
                    r = self.html_session.get(url, timeout=self.timeout)
                    r.html.render(timeout=20)  # Render JavaScript
                    content = r.html.text
                    title = r.html.find("title", first=True)
                    title = title.text if title else urlparse(url).netloc

                    if content:
                        processed_content = self._process_general_content(
                            content, query
                        )
                        sections = self._extract_structured_sections(content)
                        relevance = self._calculate_relevance(processed_content, query)

                        result = CrawlResult(
                            url=url,
                            title=title,
                            content=processed_content,
                            extracted_sections=sections,
                            links=[],
                            metadata={
                                "domain": urlparse(url).netloc,
                                "javascript_rendered": True,
                                "type": "general_website",
                            },
                            relevance_score=relevance,
                            content_type="general",
                            extraction_method="requests_html",
                            timestamp=time.time(),
                        )
                        results.append(result)
                        logger.info(
                            f"✨ Extracted with requests-html (relevance: {relevance:.3f})"
                        )
                        return results
                except Exception as e:
                    logger.debug(f"Requests-html failed, falling back: {str(e)}")

            # Fallback to standard requests + BeautifulSoup
            response = self.session.get(url, timeout=self.timeout)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")

                # Remove unwanted elements
                for element in soup(
                    ["nav", "header", "footer", "script", "style", "aside", "iframe"]
                ):
                    element.decompose()

                # Extract title
                title_tag = soup.find("title")
                title = title_tag.text.strip() if title_tag else urlparse(url).netloc

                # Extract main content
                content = soup.get_text(separator="\n", strip=True)

                if content:
                    processed_content = self._process_general_content(content, query)
                    sections = self._extract_structured_sections(content)
                    relevance = self._calculate_relevance(processed_content, query)

                    result = CrawlResult(
                        url=url,
                        title=title,
                        content=processed_content,
                        extracted_sections=sections,
                        links=self._extract_internal_links(str(soup), url),
                        metadata={
                            "domain": urlparse(url).netloc,
                            "content_length": len(content),
                            "type": "general_website",
                        },
                        relevance_score=relevance,
                        content_type="general",
                        extraction_method="beautifulsoup",
                        timestamp=time.time(),
                    )
                    results.append(result)
                    logger.info(
                        f"🔍 Extracted with BeautifulSoup (relevance: {relevance:.3f})"
                    )

        except Exception as e:
            logger.error(f"Error crawling general website {url}: {str(e)}")

        return results

    def _fetch_github_raw_file(
        self, owner: str, repo: str, file_path: str
    ) -> Optional[str]:
        """Fetch raw file content from GitHub."""
        try:
            # Try main branch first
            url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{file_path}"
            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                return response.text

            # Try master branch
            url = f"https://raw.githubusercontent.com/{owner}/{repo}/master/{file_path}"
            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                return response.text

        except Exception as e:
            logger.debug(f"Error fetching {file_path}: {str(e)}")

        return None

    def _process_documentation_content(self, content: str, query: str) -> str:
        """Process documentation content to extract relevant sections and enhance readability."""
        if not content:
            return ""

        # Extract query-relevant sections
        lines = content.split("\n")
        relevant_lines = []
        context_window = 3  # Lines before/after relevant line

        query_terms = [
            term.lower().strip() for term in query.split() if len(term.strip()) > 2
        ]

        for i, line in enumerate(lines):
            line_lower = line.lower()

            # Check if line is relevant
            is_relevant = any(term in line_lower for term in query_terms)

            # Include headers even if not directly relevant
            is_header = line.strip().startswith("#") or (
                i > 0
                and lines[i - 1].strip()
                and set(lines[i - 1].strip()) <= {"=", "-"}
            )

            if is_relevant or is_header:
                # Add context around relevant lines
                start = max(0, i - context_window)
                end = min(len(lines), i + context_window + 1)

                for j in range(start, end):
                    if lines[j] not in relevant_lines:
                        relevant_lines.append(lines[j])

        # If we have relevant content, use it; otherwise use first part of content
        if relevant_lines:
            processed = "\n".join(relevant_lines)
        else:
            processed = content[:3000]  # First 3000 chars

        return processed

    def _process_general_content(self, content: str, query: str) -> str:
        """Process general website content."""
        # Remove excessive whitespace
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        content = "\n".join(lines)

        # Limit length but prefer query-relevant sections
        if len(content) > 2000:
            query_terms = [
                term.lower().strip() for term in query.split() if len(term.strip()) > 2
            ]

            # Find best section
            best_section = ""
            best_score = 0

            # Split into chunks
            chunks = [content[i : i + 2000] for i in range(0, len(content), 1000)]

            for chunk in chunks[:5]:  # Check first 5 chunks
                chunk_lower = chunk.lower()
                score = sum(1 for term in query_terms if term in chunk_lower)

                if score > best_score:
                    best_score = score
                    best_section = chunk

            return best_section if best_section else content[:2000]

        return content

    def _extract_markdown_sections(self, content: str) -> Dict[str, str]:
        """Extract sections from markdown content."""
        sections = {}
        current_section = ""
        current_content = []

        for line in content.split("\n"):
            if line.startswith("#"):
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = "\n".join(current_content)

                # Start new section
                current_section = line.strip("#").strip()
                current_content = []
            else:
                current_content.append(line)

        # Save last section
        if current_section and current_content:
            sections[current_section] = "\n".join(current_content)

        return sections

    def _extract_structured_sections(self, content: str) -> Dict[str, str]:
        """Extract structured sections from general content."""
        # Simple section extraction based on common patterns
        sections = {}

        # Look for numbered sections, bullet points, etc.
        lines = content.split("\n")
        current_section = "main"
        current_content = []

        for line in lines:
            line = line.strip()

            # Detect section headers (various patterns)
            if line and (
                line.isupper()
                or line.startswith(("1.", "2.", "3.", "4.", "5."))
                or line.startswith(("Step", "STEP", "Chapter", "Section"))
                or len(line) < 50
                and line.endswith(":")
            ):
                # Save previous section
                if current_content:
                    sections[current_section] = "\n".join(current_content)

                # Start new section
                current_section = line[:50]  # Limit section name length
                current_content = []
            else:
                current_content.append(line)

        # Save last section
        if current_content:
            sections[current_section] = "\n".join(current_content)

        return sections

    def _extract_internal_links(
        self, content: str, base_url: str
    ) -> List[Dict[str, str]]:
        """Extract internal links from content."""
        links = []

        try:
            soup = BeautifulSoup(content, "html.parser")
            for link in soup.find_all("a", href=True)[:10]:  # Limit links
                href = link["href"]
                text = link.get_text(strip=True)

                if href.startswith("/"):
                    full_url = urljoin(base_url, href)
                elif href.startswith("http"):
                    full_url = href
                else:
                    continue

                if text and len(text) > 3:
                    links.append(
                        {
                            "url": full_url,
                            "text": text[:100],  # Limit text length
                            "type": "internal"
                            if urljoin(base_url, href).startswith(base_url)
                            else "external",
                        }
                    )
        except Exception as e:
            logger.debug(f"Error extracting links: {str(e)}")

        return links

    def _calculate_relevance(self, content: str, query: str) -> float:
        """Calculate relevance score between content and query."""
        if not content or not query:
            return 0.0

        content_lower = content.lower()
        query_lower = query.lower()

        # Split query into terms
        query_terms = [
            term.strip() for term in query_lower.split() if len(term.strip()) > 2
        ]

        if not query_terms:
            return 0.0

        # Calculate various relevance signals
        total_score = 0.0
        content_words = content_lower.split()
        content_word_count = len(content_words)

        if content_word_count == 0:
            return 0.0

        for term in query_terms:
            # Exact term frequency
            exact_matches = content_lower.count(term)
            term_frequency = exact_matches / content_word_count

            # Position bonus (earlier mentions are more important)
            first_occurrence = content_lower.find(term)
            position_bonus = 1.0 if first_occurrence < len(content) * 0.3 else 0.5

            # Context bonus (term appears in headers or important sections)
            context_bonus = 1.0
            if any(
                pattern in content_lower
                for pattern in [f"# {term}", f"## {term}", f"{term}:", f"{term} -"]
            ):
                context_bonus = 2.0

            term_score = min(term_frequency * 100 * position_bonus * context_bonus, 1.0)
            total_score += term_score

        # Normalize and apply content quality bonus
        relevance = min(total_score / len(query_terms), 1.0)

        # Bonus for longer, more comprehensive content
        if len(content) > 1000:
            relevance *= 1.2

        return min(relevance, 1.0)

    def _search_github_content(
        self, owner: str, repo: str, query: str
    ) -> List[CrawlResult]:
        """Search GitHub repository using the search API."""
        results = []

        try:
            # Search for relevant files/code
            search_url = "https://api.github.com/search/code"
            params = {
                "q": f"{query} repo:{owner}/{repo}",
                "per_page": 5,
                "sort": "indexed",
            }

            response = self.session.get(search_url, params=params, timeout=self.timeout)

            if response.status_code == 200:
                data = response.json()

                for item in data.get("items", []):
                    # Get file content
                    content = self._fetch_github_raw_file(
                        owner, repo, item.get("path", "")
                    )

                    if content and len(content) > 100:
                        processed_content = self._process_documentation_content(
                            content, query
                        )
                        relevance = self._calculate_relevance(processed_content, query)

                        result = CrawlResult(
                            url=item.get("html_url", ""),
                            title=f"{item.get('name', 'Unknown')} - {owner}/{repo}",
                            content=processed_content,
                            extracted_sections={"code_file": content[:1000]},
                            links=[],
                            metadata={
                                "repository": f"{owner}/{repo}",
                                "file_path": item.get("path"),
                                "file_type": "code",
                                "type": "github_search",
                            },
                            relevance_score=relevance,
                            content_type="github",
                            extraction_method="github_api",
                            timestamp=time.time(),
                        )
                        results.append(result)

        except Exception as e:
            logger.debug(f"GitHub API search failed: {str(e)}")

        return results

    def _crawl_github_wiki(
        self, owner: str, repo: str, query: str
    ) -> List[CrawlResult]:
        """Crawl GitHub wiki if available."""
        results = []

        try:
            wiki_url = f"https://github.com/{owner}/{repo}/wiki"
            response = self.session.get(wiki_url, timeout=self.timeout)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")

                # Extract wiki content
                wiki_content = soup.find("div", class_="markdown-body")
                if wiki_content:
                    content = wiki_content.get_text(separator="\n", strip=True)
                    processed_content = self._process_documentation_content(
                        content, query
                    )
                    relevance = self._calculate_relevance(processed_content, query)

                    if relevance > 0.3:
                        result = CrawlResult(
                            url=wiki_url,
                            title=f"Wiki - {owner}/{repo}",
                            content=processed_content,
                            extracted_sections={"wiki": content},
                            links=[],
                            metadata={
                                "repository": f"{owner}/{repo}",
                                "type": "github_wiki",
                            },
                            relevance_score=relevance,
                            content_type="github",
                            extraction_method="wiki_crawl",
                            timestamp=time.time(),
                        )
                        results.append(result)

        except Exception as e:
            logger.debug(f"GitHub wiki crawl failed: {str(e)}")

        return results

    def _extract_structured_github_readme(self, readme_elem) -> str:
        """Extract structured README content preserving formatting."""
        content_parts = []

        for element in readme_elem.find_all(
            ["h1", "h2", "h3", "h4", "p", "pre", "code", "ul", "ol", "blockquote"]
        ):
            if element.name.startswith("h"):
                level = int(element.name[1])
                content_parts.append(f"{'#' * level} {element.get_text(strip=True)}")
            elif element.name == "p":
                text = element.get_text(strip=True)
                if text:
                    content_parts.append(text)
            elif element.name == "pre":
                code_text = element.get_text()
                # Try to detect language from class
                lang = ""
                code_elem = element.find("code")
                if code_elem and "class" in code_elem.attrs:
                    for cls in code_elem["class"]:
                        if cls.startswith("language-"):
                            lang = cls.replace("language-", "")
                            break
                content_parts.append(f"```{lang}\n{code_text}\n```")
            elif element.name == "code" and element.parent.name != "pre":
                content_parts.append(f"`{element.get_text()}`")
            elif element.name in ["ul", "ol"]:
                list_items = []
                for li in element.find_all("li", recursive=False):
                    list_items.append(f"* {li.get_text(strip=True)}")
                if list_items:
                    content_parts.extend(list_items)
            elif element.name == "blockquote":
                quote_text = element.get_text(strip=True)
                content_parts.append(f"> {quote_text}")

        return "\n\n".join(content_parts)

    def _extract_github_repo_info(self, soup) -> str:
        """Extract additional repository information."""
        info_parts = []

        # Extract language information
        lang_elems = soup.find_all("span", class_="color-fg-default")
        for elem in lang_elems:
            text = elem.get_text(strip=True)
            if any(
                lang in text for lang in ["C++", "Python", "JavaScript", "C", "Java"]
            ):
                info_parts.append(f"Primary Language: {text}")
                break

        # Extract repository stats
        stats_elems = soup.find_all("a", class_="Link--primary")
        for elem in stats_elems:
            text = elem.get_text(strip=True)
            if "star" in text.lower() or text.isdigit():
                span_elem = elem.find("span")
                if span_elem:
                    info_parts.append(f"Stars: {span_elem.get_text(strip=True)}")
            elif "fork" in text.lower():
                span_elem = elem.find("span")
                if span_elem:
                    info_parts.append(f"Forks: {span_elem.get_text(strip=True)}")

        # Extract license info
        license_elem = soup.find(
            "a", href=lambda x: x and "blob" in x and "LICENSE" in x
        )
        if license_elem:
            info_parts.append(f"License: {license_elem.get_text(strip=True)}")

        return "\n".join(info_parts)

    def _extract_comprehensive_sections(
        self, readme_content: str, soup
    ) -> Dict[str, str]:
        """Extract comprehensive sections from GitHub content."""
        sections = self._extract_markdown_sections(readme_content)

        # Add special sections from the actual GitHub page content
        github_sections = {}

        # Extract installation/build sections specifically
        for section_name in [
            "DEPENDENCIES",
            "DOWNLOAD",
            "BUILD",
            "INSTALL",
            "RUN",
            "USAGE",
            "OVERVIEW",
        ]:
            section_content = self._find_section_in_readme(soup, section_name)
            if section_content:
                github_sections[section_name] = section_content

        # Merge with existing sections
        sections.update(github_sections)
        return sections

    def _find_section_in_readme(self, soup, section_name: str) -> str:
        """Find specific section content in GitHub README."""
        try:
            # Look for headers with the section name
            readme_article = soup.find("article", class_="markdown-body")
            if not readme_article:
                return ""

            for header in readme_article.find_all(["h1", "h2", "h3", "h4"]):
                header_text = header.get_text().strip()
                if (
                    section_name.lower() in header_text.lower()
                    or header_text.lower() == section_name.lower()
                ):
                    # Extract content until next header of same or higher level
                    content_parts = []
                    current = header.next_sibling
                    header_level = int(header.name[1])

                    while current:
                        if hasattr(current, "name") and current.name:
                            if current.name.startswith("h"):
                                current_level = int(current.name[1])
                                if current_level <= header_level:
                                    break

                            if current.name in ["p", "pre", "ul", "ol", "blockquote"]:
                                if current.name == "pre":
                                    code_text = current.get_text()
                                    content_parts.append(f"```\n{code_text}\n```")
                                elif current.name in ["ul", "ol"]:
                                    for li in current.find_all("li"):
                                        content_parts.append(
                                            f"* {li.get_text(strip=True)}"
                                        )
                                else:
                                    text = current.get_text(strip=True)
                                    if text:
                                        content_parts.append(text)

                        current = current.next_sibling

                    return "\n\n".join(content_parts)
        except Exception as e:
            logger.debug(f"Error finding section {section_name}: {str(e)}")

        return ""

    def _extract_github_links(self, soup, base_url: str) -> List[Dict[str, str]]:
        """Extract relevant links from GitHub page."""
        links = []

        # Extract important file links
        file_links = soup.find_all(
            "a",
            href=lambda x: x
            and any(
                file_type in x.upper()
                for file_type in [
                    "README",
                    "INSTALL",
                    "BUILD",
                    "LICENSE",
                    "CHANGELOG",
                    "CONFIGURE",
                ]
            ),
        )

        for link in file_links[:10]:  # Limit to 10 most important links
            href = link.get("href", "")
            text = link.get_text(strip=True)

            if href.startswith("/"):
                full_url = f"https://github.com{href}"
            else:
                full_url = href

            if text and len(text) > 2:
                links.append({"url": full_url, "text": text, "type": "documentation"})

        return links


def create_github_crawler() -> GitHubCrawler:
    """Factory function to create a GitHub crawler instance."""
    return GitHubCrawler()

"""
Corporate portal integration module.
Handles fallback searches when local documents don't contain relevant information.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
import json

# HTTP client imports
import requests
import aiohttp
from bs4 import BeautifulSoup

from config import get_settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CorporatePortalClient:
    """
    Client for interacting with corporate portal for document retrieval.
    Supports both REST API and web scraping approaches.
    """

    def __init__(self):
        """Initialize the corporate portal client with settings."""
        self.settings = get_settings()
        self.base_url = self.settings.corporate_portal_url
        self.api_key = self.settings.corporate_portal_api_key
        self.username = self.settings.corporate_portal_username
        self.password = self.settings.corporate_portal_password

        # Session for maintaining cookies/auth
        self.session = requests.Session()
        self._setup_session()

    def _setup_session(self):
        """Set up the HTTP session with authentication and headers."""
        # Add API key to headers if available
        if self.api_key:
            self.session.headers.update(
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
            )

        # Add user agent
        self.session.headers.update({"User-Agent": "RAG-Chatbot/1.0"})

    def authenticate(self) -> bool:
        """
        Authenticate with the corporate portal.

        Returns:
            True if authentication successful, False otherwise.
        """
        if not self.username or not self.password:
            logger.warning("No username/password provided for corporate portal")
            return False

        try:
            # Attempt login (this is a generic example - adjust for your portal)
            login_url = f"{self.base_url}/login"
            login_data = {"username": self.username, "password": self.password}

            response = self.session.post(login_url, data=login_data)

            if response.status_code == 200:
                logger.info("Successfully authenticated with corporate portal")
                return True
            else:
                logger.error(f"Authentication failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error during authentication: {str(e)}")
            return False

    def search_documents(
        self, query: str, max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for documents in the corporate portal.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.

        Returns:
            List of documents with metadata and content.
        """
        try:
            # First try API-based search
            api_results = self._search_via_api(query, max_results)
            if api_results:
                return api_results

            # Fallback to web scraping if API fails
            logger.info("API search failed, trying web scraping")
            return self._search_via_scraping(query, max_results)

        except Exception as e:
            logger.error(f"Error searching corporate portal: {str(e)}")
            return []

    def _search_via_api(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Search using the corporate portal API.

        Args:
            query: Search query string.
            max_results: Maximum number of results.

        Returns:
            List of search results.
        """
        try:
            # Generic API search endpoint (adjust for your portal)
            search_url = f"{self.base_url}/api/search"
            params = {"q": query, "limit": max_results, "format": "json"}

            response = self.session.get(search_url, params=params)

            if response.status_code == 200:
                data = response.json()
                return self._process_api_results(data)
            else:
                logger.warning(f"API search failed with status: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error in API search: {str(e)}")
            return []

    def _search_via_scraping(
        self, query: str, max_results: int
    ) -> List[Dict[str, Any]]:
        """
        Search using web scraping when API is not available.

        Args:
            query: Search query string.
            max_results: Maximum number of results.

        Returns:
            List of search results.
        """
        try:
            # Generic search page (adjust for your portal)
            search_url = f"{self.base_url}/search"
            params = {"q": query}

            response = self.session.get(search_url, params=params)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                return self._extract_search_results(soup, max_results)
            else:
                logger.warning(f"Web search failed with status: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error in web scraping search: {str(e)}")
            return []

    def _process_api_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process API response data into standardized format.

        Args:
            data: Raw API response data.

        Returns:
            Processed list of documents.
        """
        results = []

        # Adjust this based on your portal's API response format
        items = data.get("results", data.get("items", []))

        for item in items:
            result = {
                "title": item.get("title", "Untitled"),
                "content": item.get("content", item.get("summary", "")),
                "url": item.get("url", ""),
                "source": "corporate_portal_api",
                "relevance_score": item.get("score", 0.5),
                "metadata": {
                    "author": item.get("author", ""),
                    "created_date": item.get("created_date", ""),
                    "modified_date": item.get("modified_date", ""),
                    "document_type": item.get("type", "unknown"),
                },
            }
            results.append(result)

        return results

    def _extract_search_results(
        self, soup: BeautifulSoup, max_results: int
    ) -> List[Dict[str, Any]]:
        """
        Extract search results from HTML using BeautifulSoup.

        Args:
            soup: BeautifulSoup object of the search results page.
            max_results: Maximum number of results to extract.

        Returns:
            List of extracted documents.
        """
        results = []

        # Generic extraction logic (customize for your portal's HTML structure)
        # Look for common patterns like search result containers
        result_containers = soup.find_all(
            ["div", "article", "li"], class_=["result", "search-result", "document"]
        )

        for i, container in enumerate(result_containers[:max_results]):
            try:
                # Extract title
                title_elem = container.find(
                    ["h1", "h2", "h3", "a"], class_=["title", "heading"]
                )
                title = title_elem.get_text(strip=True) if title_elem else "Untitled"

                # Extract content/summary
                content_elem = container.find(
                    ["p", "div"], class_=["summary", "content", "description"]
                )
                content = content_elem.get_text(strip=True) if content_elem else ""

                # Extract URL
                link_elem = container.find("a", href=True)
                url = link_elem["href"] if link_elem else ""
                if url and not url.startswith("http"):
                    url = f"{self.base_url}{url}"

                result = {
                    "title": title,
                    "content": content,
                    "url": url,
                    "source": "corporate_portal_scraping",
                    "relevance_score": 0.5,  # Default score for scraped results
                    "metadata": {
                        "extraction_method": "web_scraping",
                        "result_position": i + 1,
                    },
                }
                results.append(result)

            except Exception as e:
                logger.warning(f"Error extracting result {i}: {str(e)}")
                continue

        return results

    def get_document_content(self, url: str) -> Optional[str]:
        """
        Retrieve full content of a document by URL.

        Args:
            url: URL of the document to retrieve.

        Returns:
            Document content as string, or None if retrieval failed.
        """
        try:
            response = self.session.get(url)

            if response.status_code == 200:
                # Try to extract text content
                soup = BeautifulSoup(response.text, "html.parser")

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                # Get text content
                text = soup.get_text()

                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (
                    phrase.strip() for line in lines for phrase in line.split("  ")
                )
                text = " ".join(chunk for chunk in chunks if chunk)

                return text
            else:
                logger.warning(f"Failed to retrieve document: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error retrieving document content: {str(e)}")
            return None

    async def async_search(
        self, query: str, max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Asynchronous version of document search.

        Args:
            query: Search query string.
            max_results: Maximum number of results.

        Returns:
            List of search results.
        """
        try:
            async with aiohttp.ClientSession() as session:
                search_url = f"{self.base_url}/api/search"
                params = {"q": query, "limit": max_results, "format": "json"}

                # Add authentication headers
                headers = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"

                async with session.get(
                    search_url, params=params, headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._process_api_results(data)
                    else:
                        logger.warning(
                            f"Async search failed with status: {response.status}"
                        )
                        return []

        except Exception as e:
            logger.error(f"Error in async search: {str(e)}")
            return []


def create_portal_client() -> CorporatePortalClient:
    """
    Factory function to create a corporate portal client.

    Returns:
        Initialized CorporatePortalClient instance.
    """
    return CorporatePortalClient()

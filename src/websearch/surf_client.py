"""
Surf-like Web Search Client

This module provides a thin wrapper around a generic Surf-style web search API.
It expects:
- an endpoint URL, e.g. https://your-surf-endpoint.example.com/search
- an API key passed as a header or query parameter

You can adapt the request/response schema to your actual Surf API.
"""

from typing import List, Dict, Optional
from urllib.parse import urlparse
import logging
import requests

logger = logging.getLogger(__name__)


class WebSearchClient:
    """Client for a Surf-like web search API."""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        default_max_results: int = 5,
    ) -> None:
        """
        Initialize the web search client.

        Args:
            endpoint: Surf search endpoint URL.
            api_key: Surf API key.
            default_max_results: Default number of results to return.
        """
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.default_max_results = default_max_results

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
    ) -> List[Dict]:
        """
        Execute a web search query.

        Args:
            query: Natural language query to search the web.
            max_results: Optional override for number of results.

        Returns:
            List of result dicts with at least: title, url, snippet.
        """
        if not query:
            return []

        limit = max_results or self.default_max_results

        try:
            # Example request; adapt to your Surf API contract.
            # SerpAPI-style authentication uses api_key as query param.
            params: Dict[str, object] = {"q": query, "api_key": self.api_key}

            parsed = urlparse(self.endpoint)
            is_serpapi = "serpapi.com" in parsed.netloc.lower()
            has_engine_in_endpoint = "engine=" in (parsed.query or "")

            if is_serpapi:
                # SerpAPI uses "num" for result count and requires an engine.
                params["num"] = limit
                if not has_engine_in_endpoint:
                    params["engine"] = "google"
            else:
                # Generic Surf-style APIs often use "limit".
                params["limit"] = limit

            response = requests.get(self.endpoint, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            # Expecting a generic schema; normalize into a simple list.
            # Adjust this block to match your actual Surf response.
            if data.get("error"):
                logger.error(f"Error during web search: {data.get('error')}")
                return []

            items = (
                data.get("results")
                or data.get("items")
                or data.get("organic_results")
                or data.get("news_results")
                or []
            )

            normalized: List[Dict] = []
            for item in items[:limit]:
                normalized.append(
                    {
                        "title": item.get("title") or item.get("name") or "",
                        "url": item.get("url") or item.get("link") or "",
                        "snippet": item.get("snippet")
                        or item.get("description")
                        or "",
                    }
                )

            return normalized

        except Exception as e:
            logger.error(f"Error during web search: {e}")
            return []


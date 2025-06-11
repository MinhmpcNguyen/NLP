import asyncio
import html
import logging
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import requests
from googlesearch import search
from search_engines import bing_search, yahoo_search

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time

from sklearn.metrics.pairwise import cosine_similarity

from chunking.sem_len_chunk import extract_chunks_from_paragraphs
from crawl_data.craw4ai_full import crawl_url

# --- Load model phù hợp semantic search ---
model: Optional[object] = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.0.0 Safari/537.36"
    )
}


def random_sleep(min_delay=1.0, max_delay=3.0):
    time.sleep(random.uniform(min_delay, max_delay))


def get_search_url(query: str, lang="vn", country="vn") -> dict[str, dict[str, str]]:
    d = {}

    # Google Search
    try:
        google_result = search(query, advanced=True, lang=lang, region=country)
        if google_result:
            for result in google_result:
                d[result.url] = {
                    "title": result.title,
                    "description": result.description,
                }
            return d
    except Exception as e:
        print(f"Google search failed: {e}")

    random_sleep()

    # Bing Search
    try:
        url = bing_search.get_search_url(query, latest=True, country=country)
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        html = response.text
        results, _ = bing_search.extract_search_results(html, url)
        if results:
            for result in results:
                d[result["url"]] = {
                    "title": result["title"],
                    "description": result["preview_text"],
                }
            return d
    except Exception as e:
        print(f"Bing search failed: {e}")
    return {}


async def fetch_content_from_url(url: str) -> str:
    try:
        content = await asyncio.wait_for(crawl_url(url), timeout=15.0)
        if content and content[0].get("markdown"):
            return content[0]["markdown"][0]
        return []
    except asyncio.TimeoutError:
        logger.warning(f"Timeout while fetching: {url}")
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
    return ""


def calculate_cosine_similarity(
    query_vector: np.ndarray, doc_vectors: np.ndarray
) -> np.ndarray:
    query = query_vector.reshape(1, -1)
    sims = cosine_similarity(query, doc_vectors)
    return np.clip(sims[0], 0.0, 1.0)


async def process_html_content(
    query: str, html_content: dict, top_k: int = 3
) -> dict[str, list[str]]:
    global model
    try:
        # Lazy-load the model
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("intfloat/multilingual-e5-large")

        # Extract semantic chunks asynchronously
        chunks_dict = defaultdict(list)
        for url, content in html_content.items():
            chunks = await extract_chunks_from_paragraphs(model, content)
            chunks_dict[url].extend(chunks)

        # Encode query and passage embeddings
        query_vec = model.encode(
            f"query: {query}", convert_to_numpy=True, normalize_embeddings=True
        )
        relevant_chunks = defaultdict(list)
        for url, chunks in chunks_dict.items():
            if not chunks:
                continue
            passage_vecs = model.encode(
                [f"passage: {chunk}" for chunk in chunks],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            # Compute cosine similarity and select top_k
            similarities = calculate_cosine_similarity(query_vec, passage_vecs)
            top_indices = np.argsort(similarities)[::-1][:top_k]
            top_chunks = [chunks[i] for i in top_indices]
            relevant_chunks[url].extend(top_chunks)

        return relevant_chunks

    except Exception as e:
        logger.error(f"Error processing content for query '{query}': {e}")
        return {}


async def fetch_and_process_urls(
    search_results: dict, top_k: int = 5
) -> dict[str, str | None]:
    # Sort by deprioritizing URLs that contain "wikipedia.org"
    filtered_urls = [
        url
        for url, _ in sorted(
            search_results.items(), key=lambda i: "wikipedia.org" in i[0].lower()
        )
    ][:top_k]

    try:
        # Fetch all URLs with timeout
        contents = await asyncio.wait_for(
            asyncio.gather(
                *[fetch_content_from_url(url) for url in filtered_urls],
                return_exceptions=True,
            ),
            timeout=30.0,
        )

        # Map URLs to content or None if an exception occurred
        result = {}
        for url, content in zip(filtered_urls, contents):
            if isinstance(content, Exception):
                logger.warning(f"Failed to fetch {url}: {content}")
                result[url] = None
            else:
                result[url] = content

        return result

    except asyncio.TimeoutError:
        logger.error("Timed out during fetch of URLs")
        return {url: None for url in filtered_urls}


async def get_interest_search(query: str, top_k: int = 5, crawl: bool = False):
    search_results: dict[str, dict[str, str]] = get_search_url(query)
    contents = defaultdict(str)

    if crawl:
        html_pages = await fetch_and_process_urls(search_results)
        relevant_chunk = await process_html_content(query, html_pages)
        for url, values in relevant_chunk.items():
            contents[url] = "\n".join(values)
        return contents
    else:
        contents = {
            url: f"{info['description']}" if info["description"] else ""
            for url, info in search_results.items()
        }
    return contents


async def main():
    query = "giám đốc đại học bách khoa hà nội qua các thời kì"
    try:
        results = await get_interest_search(query, top_k=5, crawl=True)
        print(f"Search results for '{query}':")
        for url, content in results.items():
            print(f"URL: {url}\nContent: {content}\n")
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

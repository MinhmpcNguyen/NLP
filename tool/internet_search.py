import asyncio
import html
import logging
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import requests
from googlesearch import search
from search_engines import bing_search, yahoo_search

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sklearn.metrics.pairwise import cosine_similarity

from chunking.sem_len_chunk import extract_chunks_from_paragraph
from crawl_data.craw4ai_full import crawl_url

# --- Load model phù hợp semantic search ---
model: Optional[object] = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_search_url(query: str, lang="vn", country="vn"):
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

    # Bing Search
    try:
        url = bing_search.get_search_url(query, latest=True, country=country)
        response = requests.get(url)
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

    # Yahoo Search
    try:
        url = yahoo_search.get_search_url(query, latest=True, country=country)
        response = requests.get(url)
        response.raise_for_status()
        html = response.text
        results, _ = yahoo_search.extract_search_results(html, url)
        if results:
            for result in results:
                d[result["url"]] = {
                    "title": result["title"],
                    "description": result["preview_text"],
                }
            return d
    except Exception as e:
        print(f"Yahoo search failed: {e}")
    # If all fail
    print("No results found for the query.")
    return {}


async def fetch_content_from_url(url: str) -> str:
    try:
        content = await asyncio.wait_for(crawl_url(url), timeout=15.0)
        if content and content[0].get("markdown"):
            return content[0]["markdown"][0][0]
        return ""
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
    query: str, html_content: str, top_k: int = 3
) -> List[str]:
    global model
    try:
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("intfloat/multilingual-e5-large")

        chunks = await extract_chunks_from_paragraph(html_content)
        if not chunks:
            logger.warning(f"No text chunks extracted for query: {query}")
            return []

        query_vec = model.encode(
            f"query: {query}", convert_to_numpy=True, normalize_embeddings=True
        )
        passage_vecs = model.encode(
            [f"passage: {chunk}" for chunk in chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        similarities = calculate_cosine_similarity(query_vec, passage_vecs)
        top_chunks = [chunks[i] for i in np.argsort(similarities)[::-1][:top_k]]
        return top_chunks
    except Exception as e:
        logger.error(f"Error processing content for query '{query}': {e}")
        return []


async def fetch_and_process_urls(search_results: Dict, top_k: int = 5) -> List[str]:
    filtered_urls = [
        url
        for url, _ in sorted(
            search_results.items(), key=lambda i: "wikipedia.org" not in i[0].lower()
        )
    ][:top_k]
    try:
        contents = await asyncio.wait_for(
            asyncio.gather(
                *[fetch_content_from_url(url) for url in filtered_urls],
                return_exceptions=True,
            ),
            timeout=30.0,
        )
        return contents
    except asyncio.TimeoutError:
        logger.error("Timed out during fetch of URLs")
        return []


async def get_interest_search(
    query: str, top_k: int = 5, crawl: bool = False
) -> List[str]:
    search_results = get_search_url(query)
    contents = []

    if crawl:
        html_pages = await fetch_and_process_urls(search_results, top_k)
        for page in html_pages:
            if isinstance(page, str) and page:
                processed = await process_html_content(query, page)
                contents.extend(processed)
    else:
        contents = [v.get("description", "") for v in search_results.values()]

    return contents


async def main():
    query = "tuyển sinh đại học Bách Khoa"
    try:
        results = await get_interest_search(query, top_k=5, crawl=True)
        print(f"Results for query '{query}':\n")
        print("\n".join(results))
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

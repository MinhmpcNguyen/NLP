import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from urllib.parse import urljoin, urlparse

import aiohttp
import fitz
import pandas as pd
import pytesseract
from crawl4ai import AsyncWebCrawler, BrowserConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from docx import Document
from pdf2image import convert_from_path
from process_markdown import *


def ensure_playwright_installed():
    try:
        import playwright
    except ImportError:
        print("Playwright chưa được cài. Đang tiến hành cài đặt...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "playwright"], check=True
        )
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"], check=True
        )
    else:
        try:
            subprocess.run(["playwright", "install", "chromium"], check=True)
        except Exception as e:
            print(f"Lỗi khi cài Chromium: {e}")


def ocr_pdf_image_to_text(pdf_path):
    try:
        images = convert_from_path(pdf_path)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img, lang="vie") + "\n"
        return text.strip()
    except Exception as e:
        print(f"❌ Lỗi OCR ảnh PDF: {e}")
        return ""


async def fetch_file_content(session, url, failed_links_file=None):
    file_ext = url.split(".")[-1].lower()
    try:
        async with session.get(url) as response:
            if response.status != 200:
                if failed_links_file:
                    with open(failed_links_file, "a", encoding="utf-8") as f:
                        f.write(url + "\n")
                return None, None

            content_type = response.headers.get("Content-Type", "")
            if file_ext == "pdf" and "pdf" not in content_type:
                if failed_links_file:
                    with open(failed_links_file, "a", encoding="utf-8") as f:
                        f.write(url + "\n")
                return None, None
            if file_ext in ["doc", "docx"] and "word" not in content_type:
                if failed_links_file:
                    with open(failed_links_file, "a", encoding="utf-8") as f:
                        f.write(url + "\n")
                return None, None
            if file_ext in ["xls", "xlsx"] and "spreadsheet" not in content_type:
                if failed_links_file:
                    with open(failed_links_file, "a", encoding="utf-8") as f:
                        f.write(url + "\n")
                return None, None

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{file_ext}"
            ) as tmp_file:
                tmp_file.write(await response.read())
                tmp_path = tmp_file.name

        content = ""
        header = ""

        if file_ext == "pdf":
            doc = fitz.open(tmp_path)
            content = "\n".join(page.get_text() for page in doc)
            doc.close()
            if not content.strip():  # Nếu không có text, dùng OCR
                print("🔍 Không phát hiện text trong PDF, chuyển sang OCR...")
                content = ocr_pdf_image_to_text(tmp_path)
                header = "PDF File (OCR)"
            else:
                header = "PDF File"
        elif file_ext in ["doc", "docx"]:
            doc = Document(tmp_path)
            content = "\n".join(para.text for para in doc.paragraphs)
            header = "Word File"
        elif file_ext in ["xls", "xlsx"]:
            df = pd.read_excel(tmp_path)
            content = df.to_string(index=False)
            header = "Excel File"

        os.unlink(tmp_path)
        return header, content.strip()

    except Exception as e:
        print(f"Error reading file {url}: {e}")
        if failed_links_file:
            with open(failed_links_file, "a", encoding="utf-8") as f:
                f.write(url + "\n")
        return None, None


async def fetch_and_process(
    crawler,
    url,
    max_depth,
    depth=0,
    visited=None,
    results=None,
    results_file=None,
    failed_links_file=None,
):
    if visited is None:
        visited = set()
    if results is None:
        results = []

    base_url = url.rstrip("/")
    parsed_url = urlparse(url)
    normalized_url = parsed_url._replace(fragment="").geturl().rstrip("/")
    if any(
        normalized_url.lower().endswith(ext)
        for ext in [".pdf", ".doc", ".docx", ".xls", ".xlsx"]
    ):
        print(f"📄 URL là file tài liệu: {normalized_url} → xử lý trực tiếp")
        async with aiohttp.ClientSession() as session:
            file_header, file_content = await fetch_file_content(
                session, normalized_url, failed_links_file
            )
            if file_content:
                results.append(
                    {
                        "depth": depth,
                        "url": normalized_url,
                        "markdown": [{file_header: [file_content]}],
                    }
                )
                with open(results_file, "w", encoding="utf-8") as file:
                    json.dump(results, file, indent=4, ensure_ascii=False)
        return results
    if normalized_url in visited:
        return results

    visited.add(normalized_url)

    try:
        print(f"Processing URL: {normalized_url} at depth {depth}")

        result = await crawler.arun(
            url=normalized_url,
            cache_mode="no_cache",
            magic=True,
            exclude_external_links=False,
            exclude_social_media_links=True,
            markdown_generator=DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(
                    threshold=0.48,
                    threshold_type="fixed",
                    min_word_threshold=0,
                ),
                options={"ignore_links": True},
            ),
        )

        markdown = (
            result.markdown_v2.fit_markdown
            if result.markdown_v2 and result.markdown_v2.fit_markdown
            else ""
        )

        processed_markdown = divide_to_paragraphs(markdown)

        # Skip pages with empty markdown content
        if not processed_markdown:
            print(f"Skipping URL with empty markdown: {normalized_url}")
            # Still process internal links for further crawling
            if depth < max_depth and result.links and "internal" in result.links:
                for link in result.links["internal"]:
                    absolute_url = urljoin(base_url, link["href"])
                    parsed_link = urlparse(absolute_url)
                    normalized_link = (
                        parsed_link._replace(fragment="").geturl().rstrip("/")
                    )

                    if normalized_link not in visited:
                        await fetch_and_process(
                            crawler,
                            normalized_link,
                            max_depth,
                            depth + 1,
                            visited=visited,
                            results=results,
                            results_file=results_file,
                            failed_links_file=failed_links_file,
                        )
            return results

        page_data = {
            "depth": depth,
            "url": normalized_url,
            "markdown": [processed_markdown],
        }

        async with aiohttp.ClientSession() as session:
            if result.links:
                for link_type in ["internal", "external"]:
                    if link_type in result.links:
                        for link in result.links[link_type]:
                            href = link.get("href", "")
                            if href.endswith(
                                (".pdf", ".doc", ".docx", ".xls", ".xlsx")
                            ):
                                absolute_url = urljoin(base_url, href)
                                file_header, file_content = await fetch_file_content(
                                    session, absolute_url, failed_links_file
                                )
                                if file_content:
                                    page_data["markdown"].append(
                                        {file_header: [file_content]}
                                    )

        results.append(page_data)

        # Only save to file if results_file is provided
        if results_file:
            with open(results_file, "w", encoding="utf-8") as file:
                json.dump(results, file, indent=4, ensure_ascii=False)

        if depth < max_depth and result.links and "internal" in result.links:
            for link in result.links["internal"]:
                absolute_url = urljoin(base_url, link["href"])
                parsed_link = urlparse(absolute_url)
                normalized_link = parsed_link._replace(fragment="").geturl().rstrip("/")

                if normalized_link not in visited:
                    await fetch_and_process(
                        crawler,
                        normalized_link,
                        max_depth,
                        depth + 1,
                        visited=visited,
                        results=results,
                        results_file=results_file,
                        failed_links_file=failed_links_file,
                    )

    except Exception as e:
        print(f"Error processing URL {normalized_url}: {e}")

    return results


async def crawl_url(url):
    browser_config = BrowserConfig(browser_type="chromium", headless=True)
    crawler = AsyncWebCrawler(config=browser_config)

    await crawler.start()

    try:
        result = await fetch_and_process(
            crawler=crawler,
            url=url,
            max_depth=0,
        )
        return result
    except Exception as e:
        print(f"Error during crawling: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        await crawler.close()


async def main():
    ensure_playwright_installed()
    start_url = "https://hust.edu.vn/vi/tuyen-sinh/dai-hoc/thong-tin-tuyen-sinh-dai-hoc-chinh-quy-nam-2025-651872.html"
    max_depth = 0
    results_dir = "crawl_data/crawl_results"
    results_file = os.path.join(results_dir, "results_http_hust.json")
    failed_links_file = os.path.join(results_dir, "failed_links_hut.txt")

    os.makedirs(results_dir, exist_ok=True)
    open(failed_links_file, "w", encoding="utf-8").close()

    browser_config = BrowserConfig(browser_type="chromium", headless=True)
    crawler = AsyncWebCrawler(config=browser_config)

    await crawler.start()

    print(f"Starting crawl from: {start_url} (max_depth={max_depth})")
    start_time = time.time()

    try:
        await fetch_and_process(
            crawler=crawler,
            url=start_url,
            max_depth=max_depth,
            results_file=results_file,
            failed_links_file=failed_links_file,
        )

        elapsed_time = time.time() - start_time
        print("Crawl completed.")
        print(f"Results saved to: {results_file}")
        print(f"Failed links saved to: {failed_links_file}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

    except Exception as e:
        print(f"Error during crawling: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        await crawler.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program interrupted by user.")

from models import CrawlParams,  DocumentRequest
from crawl4ai import AsyncWebCrawler
from fastapi import FastAPI
from craw4ai_full import fetch_and_process
from reformat import clean_text, extract_content
from chunking import initialize_embedding_utils, compute_advanced_similarities, adjust_threshold, create_chunks
app: FastAPI = FastAPI()


@app.post(
    "/final_api")
async def scrape(params: CrawlParams,request: DocumentRequest):
    """
    Scrape a website and return the content.

    Args:

    - **url**: The base URL to start fetching content from.
    - **max_depth**: The maximum depth of sublink to crawl.
    - **bypass**: Whether to bypass cache when crawling.

    Returns:

    - **list[PageData]**: A list of PageData objects containing the URL, the depth and the content of each page.
    """
    base_url: str = params.base_url
    max_depth: int = params.max_depth
    bypass: bool = params.bypass
    num_similarity_paragraphs_lookahead: int = request.num_similarity_paragraphs_lookahead
    based_threshold: float = request.base_threshold
    async with AsyncWebCrawler(verbose=True) as crawler:  # Disable verbose logs
        results =  await fetch_and_process(crawler=crawler,url=base_url, max_depth=max_depth, bypass=bypass)
    data = extract_content(results)
    await initialize_embedding_utils("gpt-4o", "text-embedding-3-small")

    processed_data = []

    for doc in data:
        paragraphs = doc.get("document_text", [])
        if not paragraphs:
            print(f"⚠️ Skipping {doc['document_name']} - No content found.")
            continue

        chunk_list = []
        only_strings = all(isinstance(para, str) for para in paragraphs)

        if only_strings:
            # 🟢 Nếu danh sách lớn chỉ chứa string -> Xử lý toàn bộ danh sách

            similarity_results = await compute_advanced_similarities(paragraphs,num_similarity_paragraphs_lookahead=num_similarity_paragraphs_lookahead)

            threshold = adjust_threshold(base_threshold=based_threshold,average=similarity_results["average"],variance= similarity_results["variance"])

            chunks = await create_chunks(paragraphs, similarity_results["similarities"], threshold)

            chunk_list.extend([{"content": chunk} for chunk in chunks])

        else:
            # 🔵 Nếu danh sách lớn chứa cả string và danh sách con
            for para in paragraphs:
                if isinstance(para, str):
                    # ✅ Thêm trực tiếp chuỗi vào chunk_list
                    chunk_list.append({"content": para})
                elif isinstance(para, list):
                    # 🔄 Xử lý danh sách con bằng compute_advanced_similarities()

                    similarity_results = await compute_advanced_similarities(para,num_similarity_paragraphs_lookahead=num_similarity_paragraphs_lookahead)

                    threshold = adjust_threshold(base_threshold=based_threshold,average=similarity_results["average"],variance= similarity_results["variance"])

                    chunks = await create_chunks(para, similarity_results["similarities"], threshold)

                    # ✅ Thêm chunk của danh sách con vào chunk_list theo đúng thứ tự
                    chunk_list.extend([{"content": chunk} for chunk in chunks])

            # 🔴 Sau khi xử lý danh sách con, xử lý lại chunk_list
            final_text_list = [chunk["content"] for chunk in chunk_list]  # Trích xuất nội dung đã xử lý

            similarity_results = await compute_advanced_similarities(final_text_list,num_similarity_paragraphs_lookahead=num_similarity_paragraphs_lookahead)

            threshold = adjust_threshold(base_threshold=based_threshold,average=similarity_results["average"],variance= similarity_results["variance"])

            final_chunks = await create_chunks(final_text_list, similarity_results["similarities"], threshold)

            # Cập nhật chunk_list với kết quả cuối cùng
            chunk_list = [{"content": chunk} for chunk in final_chunks]

        processed_data.append({"Url": doc["document_name"], "Chunks": chunk_list})

    return {"results": processed_data} 

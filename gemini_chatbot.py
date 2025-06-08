from google import genai

from search.dense_norm import dense_search
from tool.internet_search import get_interest_search


def get_relevant(results: dict):
    return " ".join(result["text"] for result in results)


def make_rag_prompt_vi(query):
    relevant_passage = get_relevant(dense_search(query))
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = (
        """Bạn là một trợ lý thân thiện và chính xác. Hãy trả lời câu hỏi bên dưới bằng thông tin từ đoạn văn tham khảo. \
Câu trả lời cần **ngắn gọn, đúng trọng tâm**, không lan man hoặc giải thích dư thừa. Chỉ đưa ra thông tin liên quan nhất đến câu hỏi. \
Viết cho người không chuyên, dễ hiểu. Nếu đoạn văn không đủ thông tin để trả lời, hãy nói rõ: **"Tôi không biết"**. Tuyệt đối **không được bịa hoặc suy đoán**.

CÂU HỎI: '{query}'
ĐOẠN VĂN THAM KHẢO: '{relevant_passage}'

CÂU TRẢ LỜI:
"""
    ).format(query=query, relevant_passage=escaped)

    return prompt


def generate_content_with_gemini(
    prompt: str, model: str = "gemini-2.0-flash", api_key: str = None
) -> str:
    """
    Generates content from Gemini model given a prompt.

    Parameters:
        prompt (str): The input text prompt for the model.
        model (str): The Gemini model to use. Default is 'gemini-2.0-flash'.
        api_key (str): Your Gemini API key.

    Returns:
        str: The generated text response.
    """
    if api_key is None:
        raise ValueError("API key must be provided.")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    return response.text


def make_interet_prompt_vi(query, search):
    relevant_passage = " ".join(search)
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = (
        """Bạn là một trợ lý thân thiện và chính xác. Hãy trả lời câu hỏi bên dưới bằng thông tin từ đoạn văn tham khảo. \
Câu trả lời cần **ngắn gọn, đúng trọng tâm**, không lan man hoặc giải thích dư thừa. Chỉ đưa ra thông tin liên quan nhất đến câu hỏi. \
Viết cho người không chuyên, dễ hiểu. Nếu đoạn văn không đủ thông tin để trả lời, hãy nói rõ: **"Tôi không biết"**. Tuyệt đối **không được bịa hoặc suy đoán**.

CÂU HỎI: '{query}'
ĐOẠN VĂN THAM KHẢO: '{relevant_passage}'

CÂU TRẢ LỜI:
"""
    ).format(query=query, relevant_passage=escaped)

    return prompt


async def answer_query(
    query: str, api_key: str, top_k: int = 5, rag_threshold: float = 0.2
):
    """
    Fixed version with better fallback logic
    """
    # Step 1: Try RAG
    rag_results = dense_search(query, top_k=top_k)

    if rag_results and len(rag_results) > 0:
        print("Answering with RAG...")
        rag_prompt = make_rag_prompt_vi(query)
        response = generate_content_with_gemini(rag_prompt, api_key=api_key)
        # More robust fallback condition
        should_fallback = (
            not response
            or response.strip() == "Tôi không biết."
            or "Tôi không biết" in response
            or len(response.strip()) < 10  # Very short responses
        )

        if should_fallback:
            print("Falling back to Internet Search...")
            try:
                search_results = await get_interest_search(query, crawl=False)
                print(search_results)
                if search_results:
                    internet_prompt = make_interet_prompt_vi(query, search_results)
                    internet_response = generate_content_with_gemini(
                        internet_prompt, api_key=api_key
                    )
                    return internet_response
                else:
                    print("No internet search results found")
                    return "Xin lỗi, tôi không thể tìm thấy thông tin về câu hỏi này."
            except Exception as e:
                print(f"Error in internet search: {e}")
                return "Đã xảy ra lỗi khi tìm kiếm thông tin."
        else:
            # RAG response is good enough
            return response

    return "Xin lỗi, tôi không thể tìm thấy thông tin về câu hỏi này."


import asyncio

if __name__ == "__main__":
    import os

    gemini_api_key = os.getenv("GEMINI_API_KEY") or "your-default-api-key"

    async def main():
        query = "tạ hải tùng"
        response_text = await answer_query(query, api_key=gemini_api_key)
        print(response_text)

    asyncio.run(main())

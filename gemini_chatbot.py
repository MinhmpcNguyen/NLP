import asyncio

from google import genai
from sentence_transformers import SentenceTransformer

from api_search import search_rrf
from prompt.rag_prompt import CHATBOT_PROMPT
from tool.internet_search import get_interest_search

model = SentenceTransformer("intfloat/multilingual-e5-large")


def get_relevant(results: dict):
    return " ".join(result["text"] for result in results)


def make_rag_prompt_vi(query, relevant_passage):
    relevant_passage = get_relevant(relevant_passage)
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = CHATBOT_PROMPT.format(query=query, relevant_passage=escaped)

    return prompt


def make_interet_prompt_vi(query, search):
    escaped = search.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = CHATBOT_PROMPT.format(query=query, relevant_passage=escaped)

    return prompt


def generate_content_with_gemini(
    prompt: str, model: str = "gemini-2.0-flash", api_key: str = None
) -> str:
    if api_key is None:
        raise ValueError("API key must be provided.")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    return response.text


async def answer_query(query: str, top_k: int = 5, crawl: bool = False):
    global model
    global api_key
    rag_results = search_rrf(model, query, top_k=top_k)
    if rag_results and len(rag_results) > 0:
        print("Answering with RAG...")
        rag_prompt = make_rag_prompt_vi(query, rag_results)
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
                search_results = await get_interest_search(query, crawl=crawl)
                if search_results:
                    all_content = ""
                    for url, content in search_results.items():
                        all_content += content
                    internet_prompt = make_interet_prompt_vi(query, all_content)
                    print(f"Internet prompt: {internet_prompt}")
                    internet_response = generate_content_with_gemini(
                        internet_prompt, api_key=api_key
                    )
                    return internet_response, search_results
                else:
                    return "Xin lỗi, tôi không thể tìm thấy thông tin về câu hỏi này."
            except Exception as e:
                print(f"Error during internet search: {e}")
                return "Đã xảy ra lỗi khi tìm kiếm thông tin."
        else:
            return response

    else:
        # No RAG results, fallback immediately
        print("No RAG results, falling back to Internet Search...")
        try:
            search_results = await get_interest_search(query, crawl=crawl)
            if search_results:
                all_content = ""
                for url, content in search_results.items():
                    all_content += content
                internet_prompt = make_interet_prompt_vi(query, all_content)
                print(f"Internet prompt: {internet_prompt}")
                internet_response = generate_content_with_gemini(
                    internet_prompt, api_key=api_key
                )
                return internet_response, search_results
            else:
                return "Xin lỗi, tôi không thể tìm thấy thông tin về câu hỏi này."
        except Exception as e:
            print(f"Error during internet search: {e}")
            return "Đã xảy ra lỗi khi tìm kiếm thông tin."


if __name__ == "__main__":

    async def main():
        query = "Ngày 11/06/2025 lịch tuần của Đại học Bách khoa Hà Nội có sự kiện gì không?"
        response_text, search_results = await answer_query(query, crawl=True)
        print(response_text)

    asyncio.run(main())

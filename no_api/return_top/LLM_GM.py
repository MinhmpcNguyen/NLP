import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pinecone import Pinecone
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
import time

# ✅ 1️⃣ Cấu hình API Keys & Model
os.environ["AZURE_OPENAI_API_KEY"] = "d539368d17bc4f609be5f18006f25800"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://openai-centic.openai.azure.com/"
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-08-01-preview"

# ✅ 2️⃣ Khởi tạo Pinecone và mô hình
pinecone = Pinecone(
    api_key="pcsk_4TwtQV_Qw2EeFkAvMDx3JURRUaCNetDU2XdqJGqm3e8dxA5SJdsFVTd36tVBpEJuw1UXCk"
)
index_name = "concurrent-vector-20-no-key"
index = pinecone.Index(index_name)

embedder = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-small",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# ✅ 3️⃣ Hybrid Search & Reranking
def content_only_search(user_query, top_k=100):
    """🔍 Tìm kiếm với số lượng kết quả lớn để tính toán phân phối similarity"""
    query_vector = embedder.embed_query(user_query)
    chunk_results = index.query(
        vector=query_vector,
        filter={"content": {"$exists": True}},
        top_k=top_k,
        include_metadata=True
    )
    results = chunk_results["matches"]
    return results


def determine_threshold_strategy(similarities):
    """📊 Tự động chọn phương pháp tính threshold dựa trên phân bố dữ liệu"""
    mean_sim = np.mean(similarities)
    std_dev_sim = np.std(similarities)
    skewness = (3 * (mean_sim - np.median(similarities))) / std_dev_sim  # Đánh giá độ lệch
    
    GM = np.exp(np.mean(np.log(similarities)))  # Trung bình hình học
    
    if abs(skewness) < 0.5:
        return mean_sim  # Normal distribution (dùng AM)
    elif skewness > 0.5:
        return GM  # Positively skewed: Chỉ lấy những điểm similarity cao dựa trên GM
    elif skewness < -0.5:
        return GM - 0.5 * std_dev_sim  # Negatively skewed: Giữ gần như tất cả điểm cao, chỉ bỏ điểm thấp dựa trên GM
    else:
        Q1 = np.percentile(similarities, 25)
        Q3 = np.percentile(similarities, 75)
        IQR = Q3 - Q1
        return Q3 + 1.5 * IQR  # Dùng IQR để xử lý outliers


def rerank_results(results, threshold):
    """📌 Lọc và rerank kết quả dựa trên threshold"""
    filtered_results = [item for item in results if item["score"] >= threshold]
    filtered_results.sort(key=lambda x: x["score"], reverse=True)
    return filtered_results


def search_and_generate_answer(user_query):
    """Tìm kiếm trong Pinecone, áp dụng Rerank và tạo câu trả lời với LLM"""
    start_time = time.time()
    results = content_only_search(user_query, top_k=100)
    similarities = np.array([item["score"] for item in results])
    threshold = determine_threshold_strategy(similarities)
    filtered_results = rerank_results(results, threshold)
    final_results = filtered_results
    retrieved_chunks = [item["metadata"]["content"] for item in final_results]
    retrieved_context = "\n".join(retrieved_chunks)
    prompt = f"""
    You are an AI assistant answering questions based on provided context.
    Context:
    {retrieved_context}
    User's Question:
    {user_query}
    Answer:
    """
    response = llm.invoke(prompt)
    total_time = time.time() - start_time
    return {
        "query": user_query,
        "results": final_results,
        "answer": response.content,
        "execution_time": total_time
    }

# ✅ **6️⃣ Chạy tìm kiếm với tập câu hỏi**
if __name__ == "__main__":
    user_query = "What is Trava Finance and what services does it provide?"
    result = search_and_generate_answer(user_query)
    print(result)

from pinecone import Pinecone, ServerlessSpec
import os
import json
from langchain_openai import AzureOpenAIEmbeddings
import time

# **1️⃣ Cấu hình Azure OpenAI**
os.environ["AZURE_OPENAI_API_KEY"] = "d539368d17bc4f609be5f18006f25800"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://openai-centic.openai.azure.com/"
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-08-01-preview"

# **2️⃣ Khởi tạo Pinecone**
pinecone = Pinecone(
    api_key="pcsk_4TwtQV_Qw2EeFkAvMDx3JURRUaCNetDU2XdqJGqm3e8dxA5SJdsFVTd36tVBpEJuw1UXCk"
)

start_time = time.time()

# **3️⃣ Tạo hoặc load Pinecone Index**
INDEX_NAME = "concurrent-vector-20-no-key"

if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(
        name=INDEX_NAME,
        dimension=1536,  # Đảm bảo phù hợp với mô hình embedding
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pinecone.Index(INDEX_NAME)

# **4️⃣ Khởi tạo mô hình embedding**
embedding_model = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-small",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# **5️⃣ Load dữ liệu từ JSON**
with open("/Users/Yuki/Documents/Chunking/para/vector_data/chunks_filtered.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# **6️⃣ Hàm đẩy dữ liệu lên Pinecone theo batch (fix lỗi 4MB)**
BATCH_SIZE = 100  # Chia nhỏ để tránh lỗi giới hạn

def batch_upsert(index, vectors):
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i:i + BATCH_SIZE]
        index.upsert(batch)
        print(f"Uploaded {len(batch)} vectors to Pinecone!")

# **7️⃣ Tạo Vector Embeddings và Index**
vectors = []
embed_times = []  # Danh sách lưu thời gian embed mỗi chunk

for doc in data:
    url = doc["Url"]
    for chunk in doc["Chunks"]:
        content = chunk["content"]  # Nội dung chunk
        keywords = ", ".join(chunk["keywords"])  # Chuyển list → string

        # **Tính thời gian embed**
        embed_start = time.time()
        chunk_embedding = embedding_model.embed_query(content)
        embed_duration = time.time() - embed_start

        # **Lưu thời gian embed**
        embed_times.append(embed_duration)

        # **Thêm chunk vào danh sách vectors**
        vectors.append((
            f"chunk_{len(vectors)}",
            chunk_embedding,
            {"url": url, "content": content, "keywords": keywords}
        ))

        print(f"✅ Embedded chunk {len(vectors)} in {embed_duration:.4f} seconds.")

# **8️⃣ Đẩy dữ liệu lên Pinecone**
batch_upsert(index, vectors)

# **9️⃣ Tính toán thống kê về thời gian embed**
total_embed_time = sum(embed_times)
avg_embed_time = total_embed_time / len(embed_times) if embed_times else 0

print(f"⏱️ Total embedding time: {total_embed_time:.2f} seconds.")
print(f"⏱️ Average embedding time per chunk: {avg_embed_time:.4f} seconds.")
print(f"✅ Full execution time: {time.time() - start_time:.2f} seconds.")


#149.69s
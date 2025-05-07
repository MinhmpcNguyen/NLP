import json

import faiss
import fasttext
import numpy as np

# ==== CONFIG ====
FASTTEXT_MODEL_PATH = "cc.vi.300.bin"
FAISS_INDEX_PATH = "/Users/Yuki/NLP/no_api/save_local_db/sem_len/vector_index.faiss"
METADATA_PATH = "/Users/Yuki/NLP/no_api/save_local_db/sem_len/vector_metadata.json"
TOP_K = 5
# ================

# ✅ Load mô hình, index và metadata
print("📦 Loading resources...")
ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
index = faiss.read_index(FAISS_INDEX_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)


# ✅ Hàm embedding
def embed_dense(text: str) -> np.ndarray:
    vec = ft_model.get_sentence_vector(text)
    norm = np.linalg.norm(vec)
    return (
        (vec / norm).astype(np.float32).reshape(1, -1)
        if norm > 0
        else vec.astype(np.float32).reshape(1, -1)
    )


# ✅ Dense vector search
def dense_search(query: str, top_k: int = TOP_K):
    query_vec = embed_dense(query)
    scores, indices = index.search(query_vec, top_k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx >= 0:
            meta = metadata[idx]
            results.append(
                {
                    "id": meta["id"],
                    "text": meta["text"],
                    "score": round(score, 4),
                    "chunk_index": meta.get("chunk_index", -1),
                    "url": meta.get("url", []),
                }
            )
    return results


# ✅ Chạy
if __name__ == "__main__":
    query = input("🔍 Nhập truy vấn: ").strip()
    results = dense_search(query)

    print("\n📊 Kết quả tìm kiếm:")
    for r in results:
        print(f"🔹 Score: {r['score']}")
        print(f"📄 Text: {r['text'][:200]}...")
        print(f"🌐 Source(s): {r['url']}")
        print("—" * 50)

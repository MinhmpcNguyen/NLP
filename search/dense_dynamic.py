import json
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ==== CONFIG ====
MODEL_NAME = "intfloat/multilingual-e5-large"
FAISS_INDEX_PATH = "NLP/save_local_db/sem_len/vector_index.faiss"
METADATA_PATH = "NLP/save_local_db/sem_len/vector_metadata.json"
TOP_K = 5
# ================

# ✅ Load model and FAISS index
print("📦 Loading model and FAISS index...")
model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(FAISS_INDEX_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)


# ✅ Embedding
def embed_dense(text: str) -> np.ndarray:
    vec = model.encode(text, normalize_embeddings=True)
    return np.array(vec, dtype=np.float32).reshape(1, -1)


# ✅ Main search logic (no rerank)
def search_dense_only(query: str, top_k: int = TOP_K):
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
                    "dense_score": float(score),
                    "chunk_index": meta.get("chunk_index", -1),
                    "url": meta.get("url", []),
                }
            )

    return {
        "query": query,
        "num_total": len(results),
        "results": results,
    }


# ✅ Demo
if __name__ == "__main__":
    query = input("🔍 Nhập truy vấn: ").strip()
    start = time.time()
    result = search_dense_only(query)

    print(f"\n📊 Query: {result['query']}")
    print(f"🔢 Top {TOP_K} Dense Results")

    for i, r in enumerate(result["results"], 1):
        print(f"\n🔹 Rank #{i}")
        print(f"   Dense Score: {r['dense_score']:.4f}")
        print(f"   Text: {r['text'][:200]}...")
        print(f"   Source(s): {r['url']}")
        print("—" * 50)

    print(f"\n⏱️ Execution time: {time.time() - start:.2f} seconds.")

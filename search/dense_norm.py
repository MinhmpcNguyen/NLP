import json

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

# ==== CONFIG ====
FAISS_INDEX_PATH = "/Users/Yuki/NLP/no_api/save_local_db/len/vector_index.faiss"
METADATA_PATH = "/Users/Yuki/NLP/no_api/save_local_db/len/vector_metadata.json"
TOP_K = 20  # lấy nhiều để rerank
FINAL_K = 5
# =================

# ✅ Load models
print("📦 Loading models and index...")
bi_encoder = SentenceTransformer("intfloat/multilingual-e5-large")
reranker = CrossEncoder("vinai/vinai-mteb-vietnamese-sbert")
index = faiss.read_index(FAISS_INDEX_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)


def embed_dense(text: str) -> np.ndarray:
    vec = bi_encoder.encode(text, normalize_embeddings=True)
    return np.array(vec, dtype=np.float32).reshape(1, -1)


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
                    "score": float(score),
                    "chunk_index": meta.get("chunk_index", -1),
                    "url": meta.get("url", []),
                }
            )
    return results


def rerank_results(query: str, results: list, top_k: int = FINAL_K):
    pairs = [[query, r["text"]] for r in results]
    rerank_scores = reranker.predict(pairs)

    for r, score in zip(results, rerank_scores):
        r["score"] = float(score)

    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    return sorted_results[:top_k]


if __name__ == "__main__":
    query = input("🔍 Nhập truy vấn: ").strip()

    # Step 1: FAISS dense retrieval
    dense_results = dense_search(query)

    # Step 2: Rerank using CrossEncoder
    reranked = rerank_results(query, dense_results)

    print(f"\n📊 Query: {query}")
    for r in reranked:
        print(f"🔹 Score: {r['score']:.4f}")
        print(f"📄 Text: {r['text'][:200]}...")
        print(f"🌐 Source: {r['url']}")
        print("—" * 50)

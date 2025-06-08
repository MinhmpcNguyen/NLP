import json
import time

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

# ==== CONFIG ====
MODEL_NAME = "intfloat/multilingual-e5-large"
RERANKER_MODEL = "vinai/vinai-mteb-vietnamese-sbert"
FAISS_INDEX_PATH = "/Users/Yuki/NLP/no_api/save_local_db/len/vector_index.faiss"
METADATA_PATH = "/Users/Yuki/NLP/no_api/save_local_db/len/vector_metadata.json"
TOP_K = 100
Final_k = 5
# ================

# ✅ Load models and index
print("📦 Loading models and FAISS index...")
model = SentenceTransformer(MODEL_NAME)
reranker = CrossEncoder(RERANKER_MODEL)
index = faiss.read_index(FAISS_INDEX_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)


# ✅ Embedding
def embed_dense(text: str) -> np.ndarray:
    vec = model.encode(text, normalize_embeddings=True)
    return np.array(vec, dtype=np.float32).reshape(1, -1)


# ✅ Dynamic threshold
def determine_dynamic_top_k(similarities, base_k=5):
    mean_sim = np.mean(similarities)
    std_dev_sim = np.std(similarities)
    skewness = (3 * (mean_sim - np.median(similarities))) / (std_dev_sim + 1e-9)
    n = len(similarities)

    if abs(skewness) < 0.5:
        # Gần đối xứng → giữ nguyên
        return min(n, base_k)
    elif skewness > 0.5:
        # Skew dương → tăng k để giữ nhiều điểm cao hiếm
        return min(n, base_k + 2)
    elif skewness < -0.5:
        # Skew âm → giảm k vì điểm cao nhiều hơn
        return min(n, max(base_k - 1, 1))


# ✅ Reranking
def rerank_results(query: str, candidates: list):
    pairs = [(query, item["text"]) for item in candidates]
    scores = reranker.predict(pairs)
    for item, score in zip(candidates, scores):
        item["rerank_score"] = float(score)
    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)


# ✅ Main search logic
def search_with_rerank_and_threshold(query: str, top_k: int = TOP_K):
    query_vec = embed_dense(query)
    scores, indices = index.search(query_vec, top_k)

    candidates = []
    for idx, score in zip(indices[0], scores[0]):
        if idx >= 0:
            meta = metadata[idx]
            candidates.append(
                {
                    "id": meta["id"],
                    "text": meta["text"],
                    "dense_score": float(score),
                    "chunk_index": meta.get("chunk_index", -1),
                    "url": meta.get("url", []),
                }
            )

    # ✅ Rerank trước
    reranked = rerank_results(query, candidates)

    # ✅ Threshold trên rerank score
    rerank_scores = np.array([doc["rerank_score"] for doc in reranked])
    threshold = determine_threshold_strategy(rerank_scores)

    selected = [doc for doc in reranked if doc["rerank_score"] >= threshold]

    return {
        "query": query,
        "threshold": round(threshold, 4),
        "num_total": len(candidates),
        "num_reranked": len(reranked),
        "num_selected": len(selected),
        "results": selected,
    }


# ✅ Demo
if __name__ == "__main__":
    query = input("🔍 Nhập truy vấn: ").strip()
    start = time.time()
    result = search_with_rerank_and_threshold(query)

    print(f"\n📊 Query: {result['query']}")
    print(
        f"🔢 Total FAISS: {result['num_total']} → Reranked: {result['num_reranked']} → Selected: {result['num_selected']}"
    )
    print(f"📈 Threshold (Rerank score): {result['threshold']:.4f}\n")

    for r in result["results"]:
        print(
            f"🔹 Rerank Score: {r['rerank_score']:.4f} | Dense Score: {r['dense_score']:.4f}"
        )
        print(f"📄 Text: {r['text'][:200]}...")
        print(f"🌐 Source(s): {r['url']}")
        print("—" * 50)

    print(f"\n⏱️ Execution time: {time.time() - start:.2f} seconds.")

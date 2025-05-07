import json
import time

import faiss
import fasttext
import numpy as np

# ==== CONFIG ====
FASTTEXT_MODEL_PATH = "cc.vi.300.bin"
FAISS_INDEX_PATH = "/Users/Yuki/NLP/no_api/save_local_db/len/vector_index.faiss"
METADATA_PATH = "/Users/Yuki/NLP/no_api/save_local_db/len/vector_metadata.json"
TOP_K = 100  # lấy nhiều để chọn lọc lại
# ================

# ✅ Load mô hình, FAISS index và metadata
print("📦 Loading resources...")
ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
index = faiss.read_index(FAISS_INDEX_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)


# ✅ Embedding (FastText normalized vector)
def embed_dense(text: str) -> np.ndarray:
    vec = ft_model.get_sentence_vector(text)
    norm = np.linalg.norm(vec)
    return (
        (vec / norm).astype(np.float32).reshape(1, -1)
        if norm > 0
        else vec.astype(np.float32).reshape(1, -1)
    )


# ✅ Dynamic threshold theo phân phối similarity
def determine_threshold_strategy(similarities):
    mean_sim = np.mean(similarities)
    std_dev_sim = np.std(similarities)
    skewness = (3 * (mean_sim - np.median(similarities))) / (std_dev_sim + 1e-9)
    gm = np.exp(np.mean(np.log(similarities + 1e-9)))

    if abs(skewness) < 0.5:
        return mean_sim
    elif skewness > 0.5:
        return gm
    elif skewness < -0.5:
        return gm - 0.5 * std_dev_sim
    else:
        q3 = np.percentile(similarities, 75)
        q1 = np.percentile(similarities, 25)
        iqr = q3 - q1
        return q3 + 1.5 * iqr


# ✅ Truy vấn và tính threshold động
def dense_search_with_dynamic_threshold(
    query: str, top_k: int = TOP_K, final_k: int = 5
):
    query_vec = embed_dense(query)
    scores, indices = index.search(query_vec, top_k)

    raw_results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx >= 0:
            meta = metadata[idx]
            raw_results.append(
                {
                    "id": meta["id"],
                    "text": meta["text"],
                    "score": round(score, 4),
                    "chunk_index": meta.get("chunk_index", -1),
                    "url": meta.get("url", []),
                }
            )

    similarities = np.array([r["score"] for r in raw_results])
    threshold = determine_threshold_strategy(similarities)

    filtered = [r for r in raw_results if r["score"] >= threshold]
    filtered.sort(key=lambda x: x["score"], reverse=True)

    return {
        "query": query,
        "threshold": round(threshold, 4),
        "num_total": len(raw_results),
        "num_selected": len(filtered),
        "results": filtered[:final_k],  # 🟢 chỉ lấy top-N sau khi lọc
    }


# ✅ Demo
if __name__ == "__main__":
    query = input("🔍 Nhập truy vấn: ").strip()
    start = time.time()
    search_output = dense_search_with_dynamic_threshold(query)

    print(f"\n📊 Query: {search_output['query']}")
    print(
        f"🔢 Total: {search_output['num_total']}, Selected: {search_output['num_selected']}"
    )
    print(f"📈 Threshold used: {search_output['threshold']:.4f}\n")

    for r in search_output["results"]:
        print(f"🔹 Score: {r['score']:.4f}")
        print(f"📄 Text: {r['text'][:200]}...")
        print(f"🌐 Source(s): {r['url']}")
        print("—" * 50)

    print(f"\n⏱️ Execution time: {time.time() - start:.2f} seconds.")

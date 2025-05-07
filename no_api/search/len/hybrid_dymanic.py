import json
import time

import faiss
import fasttext
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==== CONFIG ====
FASTTEXT_MODEL_PATH = "cc.vi.300.bin"
FAISS_INDEX_PATH = "/Users/Yuki/NLP/no_api/save_local_db/len/vector_index.faiss"
METADATA_PATH = "/Users/Yuki/NLP/no_api/save_local_db/len/vector_metadata.json"
TOP_K = 100
# ================

# ✅ Load resources
print("📦 Loading FastText, FAISS index, metadata...")
ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
index = faiss.read_index(FAISS_INDEX_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)
corpus_texts = [m["text"] for m in metadata]

print("🧠 Fitting TF-IDF vectorizer...")
tfidf_vectorizer = TfidfVectorizer().fit(corpus_texts)
corpus_sparse = tfidf_vectorizer.transform(corpus_texts)


# ✅ Embedding
def embed_dense(text: str):
    vec = ft_model.get_sentence_vector(text)
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32).reshape(1, -1)


def embed_sparse(text: str):
    return tfidf_vectorizer.transform([text])


# ✅ Dense search
def dense_search(query: str, top_k: int):
    q_vec = embed_dense(query)
    scores, indices = index.search(q_vec, top_k)
    return [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]


# ✅ Sparse search
def sparse_search(query: str, top_k: int):
    q_sparse = embed_sparse(query)
    sims = cosine_similarity(q_sparse, corpus_sparse)[0]
    top_idxs = np.argsort(sims)[::-1][:top_k]
    return [(int(i), float(sims[i])) for i in top_idxs]


# ✅ Dynamic threshold
def determine_threshold(similarities):
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    skew = (3 * (mean_sim - np.median(similarities))) / (std_sim + 1e-9)
    gm = np.exp(np.mean(np.log(similarities + 1e-9)))
    if abs(skew) < 0.5:
        return mean_sim
    elif skew > 0.5:
        return gm
    elif skew < -0.5:
        return gm - 0.5 * std_sim
    else:
        q3 = np.percentile(similarities, 75)
        q1 = np.percentile(similarities, 25)
        return q3 + 1.5 * (q3 - q1)


# ✅ RRF
def rrf_fusion(dense_res, sparse_res, top_k):
    scores = {}
    for rank, (idx, _) in enumerate(dense_res):
        scores[idx] = scores.get(idx, 0) + 1 / (rank + 1)
    for rank, (idx, _) in enumerate(sparse_res):
        scores[idx] = scores.get(idx, 0) + 1 / (rank + 1)
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:top_k]


# ✅ Hybrid search
def hybrid_search_dynamic(query: str, top_k: int = TOP_K):
    dense_res = dense_search(query, top_k)
    sparse_res = sparse_search(query, top_k)

    # RRF
    fused = rrf_fusion(dense_res, sparse_res, top_k)
    similarities = np.array([score for _, score in fused])
    threshold = determine_threshold(similarities)

    final = []
    for idx, score in fused:
        if score >= threshold:
            meta = metadata[idx]
            final.append(
                {
                    "id": meta["id"],
                    "text": meta["text"],
                    "score": round(score, 4),
                    "chunk_index": meta.get("chunk_index", -1),
                    "url": meta.get("url", []),
                }
            )
    return {
        "query": query,
        "threshold": round(threshold, 4),
        "total": len(fused),
        "selected": len(final),
        "results": final,
    }


# ✅ Main
if __name__ == "__main__":
    query = input("🔍 Nhập truy vấn: ").strip()
    start = time.time()
    result = hybrid_search_dynamic(query)

    print(f"\n📊 Query: {result['query']}")
    print(f"🔢 Total RRF: {result['total']}, Selected: {result['selected']}")
    print(f"📈 Threshold used: {result['threshold']:.4f}")

    for r in result["results"]:
        print(f"\n🔹 Score: {r['score']}")
        print(f"📄 Text: {r['text'][:200]}...")
        print(f"🌐 Source: {r['url']}")

    print(f"\n⏱️ Execution time: {time.time() - start:.2f} seconds.")

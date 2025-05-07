import json
from operator import itemgetter

import faiss
import fasttext
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==== ✅ CONFIG ====
FASTTEXT_MODEL_PATH = "cc.vi.300.bin"
FAISS_INDEX_PATH = "/Users/Yuki/NLP/no_api/save_local_db/sem_len/vector_index.faiss"
METADATA_PATH = "/Users/Yuki/NLP/no_api/save_local_db/sem_len/vector_metadata.json"
TOP_K = 5
# ===================

# ✅ Load model, index, metadata
print("📦 Loading model/index/metadata...")
ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
index = faiss.read_index(FAISS_INDEX_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)
corpus_texts = [doc["text"] for doc in metadata]

# ✅ TF-IDF vectorizer
print("🧠 Building TF-IDF vectorizer...")
tfidf_vectorizer = TfidfVectorizer().fit(corpus_texts)
corpus_sparse = tfidf_vectorizer.transform(corpus_texts)


# ✅ Embedding query
def embed_dense(text: str) -> np.ndarray:
    vec = ft_model.get_sentence_vector(text)
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm > 0 else vec.astype(np.float32)


def embed_sparse(text: str):
    return tfidf_vectorizer.transform([text])


# ✅ Dense search
def dense_search(query: str, top_k: int):
    query_vec = embed_dense(query).reshape(1, -1)
    scores, indices = index.search(query_vec, top_k)
    return [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]


# ✅ Sparse search
def sparse_search(query: str, top_k: int):
    query_sparse = embed_sparse(query)
    similarities = cosine_similarity(query_sparse, corpus_sparse)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(int(i), float(similarities[i])) for i in top_indices]


# ✅ RRF fusion
def search_rrf(query: str, top_k: int = TOP_K):
    dense_res = dense_search(query, top_k * 2)
    sparse_res = sparse_search(query, top_k * 2)

    rrf_scores = {}
    for rank, (idx, _) in enumerate(dense_res):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rank + 1)
    for rank, (idx, _) in enumerate(sparse_res):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rank + 1)

    sorted_rrf = sorted(rrf_scores.items(), key=itemgetter(1), reverse=True)[:top_k]

    results = []
    for idx, score in sorted_rrf:
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


# ✅ Demo
if __name__ == "__main__":
    query = input("🔍 Enter your query: ").strip()
    top_results = search_rrf(query)
    print("\n📊 Top results:\n")
    for r in top_results:
        print(f"🔹 Score: {r['score']}")
        print(f"📄 Text: {r['text'][:200]}...")
        print(f"🌐 Source(s): {r['url']}")
        print("—" * 50)

import json
import os
from operator import itemgetter

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==== ✅ CONFIG ====
MODEL_NAME = "intfloat/multilingual-e5-large"
TOP_K = 5

# ✅ Lấy đường dẫn tuyệt đối đến thư mục hiện tại
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Chuyển sang folder `save_local_db/sem_len` đúng tuyệt đối
BASE_DB_PATH = os.path.join(CURRENT_DIR, "save_local_db", "sem_len")
FAISS_INDEX_PATH = os.path.join(BASE_DB_PATH, "vector_index.faiss")
METADATA_PATH = os.path.join(BASE_DB_PATH, "vector_metadata.json")

# ✅ Kiểm tra tồn tại
if not os.path.exists(FAISS_INDEX_PATH):
    raise FileNotFoundError(f"Không tìm thấy FAISS index tại: {FAISS_INDEX_PATH}")
if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(f"Không tìm thấy metadata tại: {METADATA_PATH}")
# ===================

# ✅ Load model, index, metadata

# model_embed = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(FAISS_INDEX_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

corpus_texts = [doc["text"] for doc in metadata]


# ✅ Embedding query
def embed_dense(model, text: str) -> np.ndarray:
    vec = model.encode(text, normalize_embeddings=True)
    return np.array(vec, dtype=np.float32).reshape(1, -1)


def embed_sparse(text: str):
    print("🧠 Building TF-IDF vectorizer...")
    tfidf_vectorizer = TfidfVectorizer().fit(corpus_texts)
    corpus_sparse = tfidf_vectorizer.transform(corpus_texts)
    return corpus_sparse, tfidf_vectorizer.transform([text])


# ✅ Dense search
def dense_search(model, query: str, top_k: int):
    query_vec = embed_dense(model, query)
    scores, indices = index.search(query_vec, top_k)
    return [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]


# ✅ Sparse search
def sparse_search(query: str, top_k: int):
    corpus_sparse, query_sparse = embed_sparse(query)
    similarities = cosine_similarity(query_sparse, corpus_sparse)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(int(i), float(similarities[i])) for i in top_indices]


# ✅ RRF fusion
def search_rrf(model, query: str, top_k: int = TOP_K):
    dense_res = dense_search(model, query, top_k * 2)
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

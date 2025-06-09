import json
import os
from operator import itemgetter

import faiss
import numpy as np
from langdetect import detect
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==== ✅ CONFIG ====
MODEL_NAME = "intfloat/multilingual-e5-large"
TOP_K = 5

# ✅ Lấy đường dẫn tuyệt đối đến thư mục hiện tại
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DB_PATH = os.path.join(CURRENT_DIR, "save_local_db", "sem_len")
FAISS_INDEX_PATH = os.path.join(BASE_DB_PATH, "vector_index.faiss")
METADATA_PATH = os.path.join(BASE_DB_PATH, "vector_metadata.json")

if not os.path.exists(FAISS_INDEX_PATH):
    raise FileNotFoundError(f"Không tìm thấy FAISS index tại: {FAISS_INDEX_PATH}")
if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(f"Không tìm thấy metadata tại: {METADATA_PATH}")

# ==== ✅ LOAD MODEL, INDEX, METADATA ====
model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(FAISS_INDEX_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)
corpus_texts = [doc["text"] for doc in metadata]

# ✅ TF-IDF vectorizer chỉ build 1 lần
print("🧠 Building TF-IDF vectorizer...")
tfidf_vectorizer = TfidfVectorizer().fit(corpus_texts)
corpus_sparse = tfidf_vectorizer.transform(corpus_texts)

# ✅ Utility


def is_vietnamese(text):
    try:
        return detect(text) == "vi"
    except:
        return False


def is_valid_text(text):
    invalid_keywords = [
        "not found",
        "404",
        "chưa kích hoạt",
        "The requested URL",
        "trang thông tin này chưa kích hoạt",
    ]
    return not any(kw in text.lower() for kw in invalid_keywords)


def extract_keywords_by_tfidf(query: str, top_n: int = 5):
    query_vec = tfidf_vectorizer.transform([query])
    feature_array = np.array(tfidf_vectorizer.get_feature_names_out())
    tfidf_scores = query_vec.toarray().flatten()
    top_n_indices = tfidf_scores.argsort()[::-1][:top_n]
    return [feature_array[i] for i in top_n_indices if tfidf_scores[i] > 0]


# ✅ Embedding


def embed_dense(model, text: str) -> np.ndarray:
    vec = model.encode(text, normalize_embeddings=True)
    return np.array(vec, dtype=np.float32).reshape(1, -1)


def embed_sparse(text: str):
    return corpus_sparse, tfidf_vectorizer.transform([text])


# ✅ Search methods


def dense_search(model, query: str, top_k: int):
    query_vec = embed_dense(model, query)
    scores, indices = index.search(query_vec, top_k)
    return [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]


def sparse_search(query: str, top_k: int):
    corpus, query_sparse = embed_sparse(query)
    similarities = cosine_similarity(query_sparse, corpus)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(int(i), float(similarities[i])) for i in top_indices]


# ✅ Final RRF Search with boost


def search_rrf(model, query: str, top_k: int = TOP_K, keyword_boost: float = 0.5):
    query_keywords = extract_keywords_by_tfidf(query)

    dense_res = dense_search(model, query, top_k * 4)
    sparse_res = sparse_search(query, top_k * 4)

    rrf_scores = {}
    for rank, (idx, _) in enumerate(dense_res):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rank + 1)
    for rank, (idx, _) in enumerate(sparse_res):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rank + 1)

    sorted_rrf = sorted(rrf_scores.items(), key=itemgetter(1), reverse=True)

    filtered_results = []
    fallback_results = []

    for idx, score in sorted_rrf:
        meta = metadata[idx]
        text = meta["text"]
        item = {
            "id": meta["id"],
            "text": text,
            "score": score,
            "chunk_index": meta.get("chunk_index", -1),
            "url": meta.get("url", []),
        }

        if is_vietnamese(text) and is_valid_text(text):
            filtered_results.append(item)
        else:
            fallback_results.append(item)

        if len(filtered_results) >= top_k:
            break

    if len(filtered_results) < top_k:
        filtered_results.extend(fallback_results[: top_k - len(filtered_results)])

    for item in filtered_results:
        if any(kw in item["text"].lower() for kw in query_keywords):
            item["score"] += keyword_boost
        item["score"] = round(item["score"], 4)

    return sorted(filtered_results, key=lambda x: x["score"], reverse=True)

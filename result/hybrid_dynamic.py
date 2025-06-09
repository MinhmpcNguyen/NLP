import csv
import json

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# === Config ===
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
CSV_PATH = "crawl_data/GT_Q_A_Context.csv"
FAISS_INDEX_PATH = "NLP/save_local_db_copy/sem_len/vector_index.faiss"
METADATA_PATH = "NLP/save_local_db_copy/sem_len/vector_metadata.json"
TOP_K = 100
SIM_THRESHOLD = 0.70
OUTPUT_CSV = "crawl_data/retrieval_result_hybrid.csv"

# === Load model, index, metadata ===
print("📦 Loading model, index and metadata...")
model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(FAISS_INDEX_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)
corpus_texts = [m["text"] for m in metadata]

print("🧠 Fitting TF-IDF vectorizer...")
tfidf_vectorizer = TfidfVectorizer().fit(corpus_texts)
corpus_sparse = tfidf_vectorizer.transform(corpus_texts)


# === Embedding functions ===
def embed_dense(text: str):
    vec = model.encode(text, normalize_embeddings=True)
    return np.array(vec, dtype=np.float32).reshape(1, -1)


def embed_sparse(text: str):
    return tfidf_vectorizer.transform([text])


def embed_tensor(text: str):
    return model.encode(text, convert_to_tensor=True, normalize_embeddings=True)


# === Search functions ===
def dense_search(query: str, top_k: int):
    q_vec = embed_dense(query)
    scores, indices = index.search(q_vec, top_k)
    return [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]


def sparse_search(query: str, top_k: int):
    q_sparse = embed_sparse(query)
    sims = cosine_similarity(q_sparse, corpus_sparse)[0]
    top_idxs = np.argsort(sims)[::-1][:top_k]
    return [(int(i), float(sims[i])) for i in top_idxs]


def rrf_fusion(dense_res, sparse_res, top_k):
    scores = {}
    for rank, (idx, _) in enumerate(dense_res):
        scores[idx] = scores.get(idx, 0) + 1 / (rank + 1)
    for rank, (idx, _) in enumerate(sparse_res):
        scores[idx] = scores.get(idx, 0) + 1 / (rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


# === Evaluation loop ===
print("🔍 Evaluating hybrid retrieval with ground truth...")
df = pd.read_csv(CSV_PATH, quoting=csv.QUOTE_MINIMAL, on_bad_lines="skip")
df = df.dropna(subset=["Question", "Context"])

results = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    question = row["Question"]
    true_context = row["Context"]

    dense_res = dense_search(question, TOP_K)
    sparse_res = sparse_search(question, TOP_K)
    fused = rrf_fusion(dense_res, sparse_res, TOP_K)

    top_k_texts = [metadata[idx]["text"] for idx, _ in fused]

    true_emb = embed_tensor(true_context)
    retrieved_embs = [embed_tensor(t) for t in top_k_texts if t.strip()]
    if not retrieved_embs:
        max_sim = 0.0
    else:
        sims = [float(util.cos_sim(true_emb, emb)[0][0]) for emb in retrieved_embs]
        max_sim = max(sims)

    is_correct = "Correct" if max_sim >= SIM_THRESHOLD else "Incorrect"

    results.append(
        {
            "Question": question,
            "Context": true_context,
            "max_similarity": round(max_sim, 4),
            "Correctness": is_correct,
            "Top_k_texts": " ||| ".join(top_k_texts),
        }
    )

# === Save results ===
pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Evaluation completed. Results saved to: {OUTPUT_CSV}")

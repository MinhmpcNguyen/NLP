import csv
import json

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# === Config ===
MODEL_NAME = "intfloat/multilingual-e5-large"
CSV_PATH = "crawl_data/GT_Q_A_Context.csv"  # đường dẫn file GT
FAISS_INDEX_PATH = "NLP/save_local_db/sem_len/vector_index.faiss"
METADATA_PATH = "NLP/save_local_db/sem_len/vector_metadata.json"
TOP_K = 5
SIM_THRESHOLD = 0.80
OUTPUT_CSV = "crawl_data/retrieval_result_dense_norm.csv"

# === Load model, index, metadata ===
print("📦 Loading model, index and metadata...")
model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(FAISS_INDEX_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# === Load CSV and clean ===
df = pd.read_csv(CSV_PATH, quoting=csv.QUOTE_MINIMAL, on_bad_lines="skip")
df = df.dropna(subset=["Question", "Context"])


# === Embedding function ===
def embed(text: str):
    if not isinstance(text, str) or not text.strip():
        return None
    return model.encode(text, convert_to_tensor=True, normalize_embeddings=True)


# === Retrieval function ===
def retrieve_top_k(query: str, top_k=TOP_K):
    query_vec = model.encode(query, normalize_embeddings=True)
    scores, indices = index.search(np.array([query_vec], dtype=np.float32), top_k)

    results = []
    for idx in indices[0]:
        if idx >= 0:
            results.append(metadata[idx]["text"])
    return results


# === Evaluation loop ===
print("🔍 Evaluating retrieval on ground truth...")
results = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    question = row["Question"]
    true_context = row["Context"]

    top_k_texts = retrieve_top_k(question, top_k=TOP_K)
    true_emb = embed(true_context)
    if true_emb is None:
        continue

    retrieved_embs = [embed(t) for t in top_k_texts if embed(t) is not None]
    if not retrieved_embs:
        avg_sim = 0.0
    else:
        sims = [float(util.cos_sim(true_emb, r)[0][0]) for r in retrieved_embs]
        avg_sim = np.mean(sims)

    is_correct = "Correct" if avg_sim >= SIM_THRESHOLD else "Incorrect"

    results.append(
        {
            "Question": question,
            "Context": true_context,
            "avg_similarity": round(avg_sim, 4),
            "Correctness": is_correct,
            "Top_k_texts": " ||| ".join(top_k_texts),
        }
    )

# === Save to CSV ===
pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Evaluation completed. Results saved to: {OUTPUT_CSV}")

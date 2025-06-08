import faulthandler

faulthandler.enable()

import json
import time

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ==== CONFIG ====
MODEL_NAME = "intfloat/multilingual-e5-large"
RERANKER_MODEL = "itdainb/PhoRanker"
TOP_K = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FAISS_INDEX_PATH = "/Users/Yuki/NLP/no_api/save_local_db/len/vector_index.faiss"
METADATA_PATH = "/Users/Yuki/NLP/no_api/save_local_db/len/vector_metadata.json"
# ===================

print("📦 Loading models and FAISS index...")
dense_model = SentenceTransformer(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
reranker = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL).to(DEVICE)

index = faiss.read_index(FAISS_INDEX_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# ===== DEBUG INFO =====
print("✅ FAISS index dimension:", index.d)
print(
    "✅ Dense model output dimension:", dense_model.get_sentence_embedding_dimension()
)
# =======================


def embed_dense(text: str) -> np.ndarray:
    vec = dense_model.encode(text, normalize_embeddings=True)
    print("🔍 [DEBUG] Query vector shape:", vec.shape)  # thêm debug
    return np.array(vec, dtype=np.float32).reshape(1, -1)


def dense_search(query: str, top_k: int = TOP_K):
    query_vec = embed_dense(query)
    if query_vec.shape[1] != index.d:
        raise ValueError(
            f"❌ Dimension mismatch: Query vector dim {query_vec.shape[1]} vs FAISS index dim {index.d}"
        )

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
    return results


def rerank(query: str, candidates: list, batch_size: int = 4):
    reranked = []
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        pairs = [(query, item["text"]) for item in batch]
        inputs = tokenizer(
            pairs, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            logits = reranker(**inputs).logits
            scores = logits[:, 1] if logits.shape[1] > 1 else logits.squeeze()

        for item, score in zip(batch, scores):
            item["rerank_score"] = float(score)
            reranked.append(item)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked


def determine_threshold(similarities):
    mean_sim = np.mean(similarities)
    std_dev = np.std(similarities)
    skew = (3 * (mean_sim - np.median(similarities))) / (std_dev + 1e-9)
    gm = np.exp(np.mean(np.log(similarities + 1e-9)))

    if abs(skew) < 0.5:
        return mean_sim
    elif skew > 0.5:
        return gm
    elif skew < -0.5:
        return gm - 0.5 * std_dev
    else:
        q3 = np.percentile(similarities, 75)
        q1 = np.percentile(similarities, 25)
        return q3 + 1.5 * (q3 - q1)


def search_with_rerank_and_threshold(query: str):
    initial_results = dense_search(query)
    reranked = rerank(query, initial_results)

    rerank_scores = np.array([r["rerank_score"] for r in reranked])
    threshold = determine_threshold(rerank_scores)

    selected = [r for r in reranked if r["rerank_score"] >= threshold]

    return {
        "query": query,
        "threshold": round(threshold, 4),
        "num_total": len(initial_results),
        "num_selected": len(selected),
        "results": selected,
    }


# ✅ Main run
if __name__ == "__main__":
    query = input("🔍 Nhập truy vấn: ").strip()
    start = time.time()

    output = search_with_rerank_and_threshold(query)

    print(f"\n📊 Query: {output['query']}")
    print(
        f"🔢 Total Dense: {output['num_total']} → Selected after Rerank+Threshold: {output['num_selected']}"
    )
    print(f"📈 Threshold: {output['threshold']:.4f}\n")

    for r in output["results"]:
        print(
            f"🔹 Rerank Score: {r['rerank_score']:.4f} | Dense Score: {r['dense_score']:.4f}"
        )
        print(f"📄 Text: {r['text'][:200]}...")
        print(f"🌐 Source(s): {r['url']}")
        print("—" * 50)

    print(f"\n⏱️ Execution time: {time.time() - start:.2f} seconds.")

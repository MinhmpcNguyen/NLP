import json
import uuid

import faiss
import fasttext
import numpy as np
from tqdm import tqdm

# ==== ✅ CONFIG ====
FASTTEXT_MODEL_PATH = "NLP/cc.vi.300.bin"
NDJSON_PATH = "no_api/chunking/cluster/sem_len_cluster_rechunked.json"
FAISS_INDEX_PATH = "no_api/save_local_db/sem_len_cluster_sem/rechunked_index.faiss"
METADATA_PATH = "no_api/save_local_db/sem_len_cluster_sem/rechunked_metadata.json"
# ===================

# ✅ Load FastText model
print("📦 Loading FastText model...")
ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)

# ✅ FAISS index (300 chiều, cosine similarity)
dimension = 300
index = faiss.IndexFlatIP(dimension)  # dùng dot product, cần normalize

# ✅ Metadata list
metadata_store = []


# ✅ Hàm tạo embedding từ FastText
def get_embedding(text: str):
    vec = ft_model.get_sentence_vector(text)
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm > 0 else vec.astype(np.float32)


# ✅ Load và lưu vector + metadata
def upload_chunks_to_faiss(ndjson_path: str):
    with open(ndjson_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)  # Không phải NDJSON, là JSON object

    for cluster_name, cluster_data in tqdm(
        raw_data.items(), desc="🚀 Embedding & Saving clusters"
    ):
        chunks = cluster_data.get("Chunks", [])
        for i, chunk in enumerate(chunks):
            text = chunk.get("content", "").strip()
            if not text:
                continue

            vec = get_embedding(text)
            index.add(np.expand_dims(vec, axis=0))

            metadata_store.append(
                {
                    "id": str(uuid.uuid4()),
                    "cluster": cluster_name,
                    "chunk_index": i,
                    "text": text,
                    "source_urls": chunk.get("source_urls", []),
                }
            )


# ✅ Run & Save
if __name__ == "__main__":
    upload_chunks_to_faiss(NDJSON_PATH)

    # Lưu FAISS index
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"✅ FAISS index saved to {FAISS_INDEX_PATH}")

    # Lưu metadata song song (list of dicts, không phải NDJSON)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_store, f, indent=2, ensure_ascii=False)
    print(f"✅ Metadata saved to {METADATA_PATH}")

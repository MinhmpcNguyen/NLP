import json
import uuid

import faiss
import fasttext
import numpy as np
from tqdm import tqdm

# ==== ✅ CONFIG ====
FASTTEXT_MODEL_PATH = "cc.vi.300.bin"
NDJSON_PATH = "/Users/Yuki/NLP/no_api/chunking/sem_len/sem_len.json"
FAISS_INDEX_PATH = "vector_index.faiss"
METADATA_PATH = "vector_metadata.json"
# ===================

# ✅ Load FastText model
print("📦 Loading FastText model...")
ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)

# ✅ FAISS index (300 chiều, cosine similarity)
dimension = 300
index = faiss.IndexFlatIP(dimension)  # dùng dot product, cần normalize

# ✅ Metadata list (song song với vectors)
metadata_store = []


# ✅ Hàm tạo embedding từ FastText
def get_embedding(text: str):
    vec = ft_model.get_sentence_vector(text)
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm > 0 else vec.astype(np.float32)


# ✅ Load và lưu vector + metadata
def upload_chunks_to_faiss(ndjson_path: str):
    for line in tqdm(
        open(ndjson_path, "r", encoding="utf-8"), desc="🚀 Embedding & Saving"
    ):
        doc = json.loads(line)
        url = doc.get("Url", "")
        chunks = doc.get("Chunks", [])

        for i, chunk in enumerate(chunks):
            text = chunk.get("content", "").strip()
            if not text:
                continue

            vec = get_embedding(text)
            index.add(np.expand_dims(vec, axis=0))

            metadata_store.append(
                {"id": str(uuid.uuid4()), "url": url, "text": text, "chunk_index": i}
            )


# ✅ Run & Save
if __name__ == "__main__":
    upload_chunks_to_faiss(NDJSON_PATH)

    # Lưu FAISS index
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"✅ FAISS index saved to {FAISS_INDEX_PATH}")

    # Lưu metadata song song
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_store, f, indent=2, ensure_ascii=False)
    print(f"✅ Metadata saved to {METADATA_PATH}")

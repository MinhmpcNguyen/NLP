import json
import uuid

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ==== ✅ CONFIG ====
NDJSON_PATH = "NLP/chunking/sem_len/sem_len.json"
FAISS_INDEX_PATH = "NLP/save_local_db/len/vector_index.faiss"
METADATA_PATH = "NLP/save_local_db/len/vector_metadata.json"
MODEL_NAME = "intfloat/multilingual-e5-large"
# ===================

# ✅ Load SentenceTransformer model
print(f"📦 Loading SentenceTransformer model: {MODEL_NAME} ...")
sentence_model = SentenceTransformer(MODEL_NAME)

# ✅ FAISS index (embedding_dim, cosine similarity → normalize trước)
dimension = sentence_model.get_sentence_embedding_dimension()
index = faiss.IndexFlatIP(dimension)  # cosine similarity via inner product

# ✅ Metadata list (song song với vectors)
metadata_store = []


# ✅ Hàm tạo embedding từ SentenceTransformer
def get_embedding(text: str):
    vec = sentence_model.encode(text, normalize_embeddings=True)
    return np.array(vec, dtype=np.float32)


# ✅ Load và lưu vector + metadata
def upload_chunks_to_faiss(ndjson_path: str):
    with open(ndjson_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="🚀 Embedding & Saving"):
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
                    {
                        "id": str(uuid.uuid4()),
                        "url": url,
                        "text": text,
                        "chunk_index": i,
                    }
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

import json
import uuid

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ==== ✅ CONFIG ====
MODEL_NAME = "intfloat/multilingual-e5-large"
NDJSON_PATH = "/Users/Yuki/NLP/no_api/chunking/sem_len/sem_len.json"
FAISS_INDEX_PATH = "NLP/save_local_db/sem_len/vector_index.faiss"
METADATA_PATH = "NLP/save_local_db/sem_len/vector_metadata.json"
BATCH_SIZE = 64
# ===================

# ✅ Load SentenceTransformer model
print(f"📦 Loading SentenceTransformer model: {MODEL_NAME} ...")
model = SentenceTransformer(MODEL_NAME)
dimension = model.get_sentence_embedding_dimension()

# ✅ FAISS index (cosine similarity via normalized inner product)
index = faiss.IndexFlatIP(dimension)

# ✅ Metadata list
metadata_store = []


# ✅ Helper: tạo embedding với chuẩn hóa
def get_embeddings(texts: list[str]) -> np.ndarray:
    embeddings = model.encode(texts, normalize_embeddings=True)
    return np.array(embeddings, dtype=np.float32)


# ✅ Main function: xử lý NDJSON
def upload_chunks_to_faiss(ndjson_path: str):
    with open(ndjson_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    texts = []
    metas = []

    for line in tqdm(lines, desc="🚀 Collecting chunks"):
        doc = json.loads(line)
        url = doc.get("Url", "")
        chunks = doc.get("Chunks", [])

        for i, chunk in enumerate(chunks):
            text = chunk.get("content", "").strip()
            if not text:
                continue
            texts.append(text)
            metas.append(
                {"id": str(uuid.uuid4()), "url": url, "text": text, "chunk_index": i}
            )

            # Nếu đủ batch thì encode + add
            if len(texts) >= BATCH_SIZE:
                vecs = get_embeddings(texts)
                index.add(vecs)
                metadata_store.extend(metas)
                texts, metas = [], []

    # Xử lý batch cuối cùng
    if texts:
        vecs = get_embeddings(texts)
        index.add(vecs)
        metadata_store.extend(metas)


# ✅ Run & Save
if __name__ == "__main__":
    upload_chunks_to_faiss(NDJSON_PATH)

    # Lưu FAISS index
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"✅ FAISS index saved to {FAISS_INDEX_PATH}")

    # Lưu metadata
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_store, f, indent=2, ensure_ascii=False)
    print(f"✅ Metadata saved to {METADATA_PATH}")

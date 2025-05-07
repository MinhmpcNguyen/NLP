import json
import uuid

import faiss
import fasttext
import numpy as np
from tqdm import tqdm

# === ✅ CONFIG ===
FASTTEXT_MODEL_PATH = "cc.vi.300.bin"
INPUT_JSON = "/Users/Yuki/NLP/no_api/chunking/length/len.json"  # ← file bạn vừa cung cấp (dạng list of docs)
FAISS_INDEX_PATH = "vector_index.faiss"
METADATA_PATH = "vector_metadata.json"
# ==================

# ✅ Load FastText
print("📦 Loading FastText model...")
ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)

# ✅ Init FAISS index (cosine similarity = normalize + dot product)
dimension = 300
index = faiss.IndexFlatIP(dimension)

# ✅ Metadata lưu song song (id, url, text, chunk index)
metadata_store = []


def get_embedding(text: str):
    vec = ft_model.get_sentence_vector(text)
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm > 0 else vec.astype(np.float32)


# ✅ Load input + build FAISS
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    docs = json.load(f)

for doc in tqdm(docs, desc="🔄 Embedding and indexing"):
    url = doc.get("url", "")
    chunks = doc.get("chunks", [])
    for i, chunk in enumerate(chunks):
        text = chunk.get("content", "").strip()
        if not text:
            continue
        vec = get_embedding(text)
        index.add(np.expand_dims(vec, axis=0))
        metadata_store.append(
            {"id": str(uuid.uuid4()), "url": url, "text": text, "chunk_index": i}
        )

# ✅ Save index + metadata
faiss.write_index(index, FAISS_INDEX_PATH)
with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata_store, f, indent=2, ensure_ascii=False)

print(f"✅ FAISS saved to {FAISS_INDEX_PATH}")
print(f"✅ Metadata saved to {METADATA_PATH}")

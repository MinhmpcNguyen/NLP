import json
import uuid

import numpy as np
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ==== ✅ CONFIG ====
MODEL_NAME = "intfloat/multilingual-e5-large"
NDJSON_PATH = "/Users/Yuki/NLP/no_api/chunking/sem_len/sem_len.json"
PINECONE_INDEX_NAME = "sem-len-index"
PINECONE_API_KEY = "pcsk_3FE9xx_BzUsMz29JDSGS3Q8mKdTQy16SdtFeiwivhSxAWm9Xwh83fqhHhRBKXvc7FZa9bY"  # 🔐 replace with your key
PINECONE_CLOUD = "aws"  # or "gcp"
PINECONE_REGION = "us-east-1"  # ✅ check in Pinecone console
BATCH_SIZE = 64
# ===================

# ✅ Load model
print(f"📦 Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
dimension = model.get_sentence_embedding_dimension()

# ✅ Initialize Pinecone v3 client
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ Create index if not exists
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )

index = pc.Index(PINECONE_INDEX_NAME)


# ✅ Helper: encode + normalize
def get_embeddings(texts: list[str]) -> np.ndarray:
    return model.encode(texts, normalize_embeddings=True).tolist()


# ✅ Main upload logic
def upload_chunks_to_pinecone(ndjson_path: str):
    with open(ndjson_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    texts, metadatas, ids = [], [], []

    for line in tqdm(lines, desc="🚀 Processing chunks"):
        doc = json.loads(line)
        url = doc.get("Url", "")
        chunks = doc.get("Chunks", [])

        for i, chunk in enumerate(chunks):
            text = chunk.get("content", "").strip()
            if not text:
                continue

            uid = str(uuid.uuid4())
            texts.append(text)
            metadatas.append({"url": url, "text": text, "chunk_index": i})
            ids.append(uid)

            if len(texts) >= BATCH_SIZE:
                vecs = get_embeddings(texts)
                index.upsert(vectors=list(zip(ids, vecs, metadatas)))
                texts, metadatas, ids = [], [], []

    # ✅ Final batch
    if texts:
        vecs = get_embeddings(texts)
        index.upsert(vectors=list(zip(ids, vecs, metadatas)))

    print("✅ All vectors uploaded to Pinecone.")


# ✅ Main
if __name__ == "__main__":
    upload_chunks_to_pinecone(NDJSON_PATH)

import json

import faiss
import fasttext
import numpy as np

FASTTEXT_MODEL_PATH = "cc.vi.300.bin"
NDJSON_PATH = "sem_len_cluster_rechunked.ndjson"
FAISS_INDEX_PATH = "vector_index.faiss"
METADATA_PATH = "vector_metadata.json"
index = faiss.read_index(
    "/Users/Yuki/NLP/no_api/save_local_db/sem_len/vector_index.faiss"
)
with open(
    "/Users/Yuki/NLP/no_api/save_local_db/sem_len/vector_metadata.json",
    "r",
    encoding="utf-8",
) as f:
    metadata_store = json.load(f)
ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)


def get_embedding(text: str):
    vec = ft_model.get_sentence_vector(text)
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm > 0 else vec.astype(np.float32)


# Query
query = "tuyển sinh chương trình sau đại học"
query_vec = get_embedding(query)
scores, indices = index.search(np.expand_dims(query_vec, axis=0), k=5)

# In kết quả
for i, idx in enumerate(indices[0]):
    print(
        f"🔎 Match {i + 1}: {metadata_store[idx]['text']}  (score={scores[0][i]:.4f})"
    )

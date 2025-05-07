import asyncio
import json
from typing import Dict, List

import fasttext
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ====== GLOBAL STATE ======
embedding_cache = {}
ft_model = None
# ===========================


# ✅ Load FastText model
async def initialize_embedding_utils():
    global ft_model
    ft_model = fasttext.load_model("cc.vi.300.bin")
    return {"embedding": "FastText Vietnamese"}


# ✅ Embed một đoạn văn
async def create_embedding(paragraph: str):
    if paragraph in embedding_cache:
        return embedding_cache[paragraph]

    embedding = await asyncio.to_thread(ft_model.get_sentence_vector, paragraph)
    embedding = np.array(embedding)
    embedding_cache[paragraph] = embedding
    return embedding


# ✅ Gán đoạn văn vào các cụm (theo ngưỡng similarity và khoảng cách)
def clustering_paragraphs(
    embeddings: np.ndarray, similarity_threshold: float, distance_threshold: float
) -> Dict[int, List[int]]:
    clusters = {}
    cluster_centroids = []

    for idx, emb in enumerate(embeddings):
        assigned = False
        for cluster_id, centroid in enumerate(cluster_centroids):
            similarity = cosine_similarity([emb], [centroid])[0][0]
            distance = np.linalg.norm(emb - centroid)

            if similarity > similarity_threshold and distance < distance_threshold:
                clusters[cluster_id].append(idx)
                n = len(clusters[cluster_id])
                cluster_centroids[cluster_id] = centroid * (n - 1) / n + emb / n
                assigned = True
                break

        if not assigned:
            new_cluster_id = len(cluster_centroids)
            clusters[new_cluster_id] = [idx]
            cluster_centroids.append(emb)

    return clusters, cluster_centroids


# ✅ Đọc NDJSON và trả về list [{text, url}]
def load_paragraphs_with_source(file_path: str) -> List[Dict[str, str]]:
    paragraphs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            url = doc.get("Url", "")
            chunks = doc.get("Chunks", [])
            for chunk in chunks:
                if isinstance(chunk, dict) and "content" in chunk:
                    paragraphs.append({"text": chunk["content"].strip(), "url": url})
    return paragraphs


# ✅ Thực hiện cluster và gắn URL lại
async def cluster_paragraphs_only(
    paragraphs: List[Dict[str, str]],
    similarity_threshold: float = 0.75,
    distance_threshold: float = 1.5,
):
    texts = [p["text"] for p in paragraphs]
    embeddings = await asyncio.gather(*[create_embedding(text) for text in texts])
    embeddings = np.vstack(embeddings)

    clusters, _ = clustering_paragraphs(
        embeddings, similarity_threshold, distance_threshold
    )

    cluster_info = {}
    for cluster_id, indices in clusters.items():
        cluster_name = f"cluster_{cluster_id}"
        cluster_info[cluster_name] = {
            "paragraphs": [
                {"text": paragraphs[i]["text"], "url": paragraphs[i]["url"]}
                for i in indices
            ]
        }

    return cluster_info


# ✅ Main runner
async def main():
    await initialize_embedding_utils()

    file_path = "/Users/Yuki/NLP/no_api/chunking/sem_len/sem_len.json"  # NDJSON input
    output_path = "/Users/Yuki/NLP/no_api/chunking/cluster/sem_len_cluster.json"

    paragraphs = load_paragraphs_with_source(file_path)
    clustered = await cluster_paragraphs_only(paragraphs)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clustered, f, indent=4, ensure_ascii=False)

    print(f"✅ Clustering completed. Output saved to {output_path}")


# ✅ Entry point
if __name__ == "__main__":
    asyncio.run(main())

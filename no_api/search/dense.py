# dense_search.py

import pinecone
from sentence_transformers import SentenceTransformer

# Init Pinecone
pinecone.init(
    api_key="YOUR_API_KEY",  # <-- điền vào đây
    environment="YOUR_ENVIRONMENT",  # ex: "gcp-starter"
)
index = pinecone.Index("your-index-name")  # <-- điền tên index

# Dense model
dense_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed_dense(texts):
    """Embed dense vectors"""
    return dense_model.encode(texts, normalize_embeddings=True)


def upsert_dense(texts, ids):
    """Upsert dense vectors vào Pinecone"""
    dense_vectors = embed_dense(texts)
    vectors = []
    for idx, vec in enumerate(dense_vectors):
        vectors.append(
            {"id": ids[idx], "values": vec.tolist(), "metadata": {"text": texts[idx]}}
        )
    index.upsert(vectors=vectors)


def search_dense(query, top_k=5):
    """Search dense vectors"""
    dense_query = embed_dense([query])[0]
    res = index.query(vector=dense_query.tolist(), top_k=top_k, include_metadata=True)
    results = [
        {"id": match.id, "text": match.metadata["text"], "score": match.score}
        for match in res.matches
    ]
    return results


if __name__ == "__main__":
    # Ví dụ dùng
    texts = ["Hust university", "Pinecone vector search", "Machine learning is fun"]
    ids = ["1", "2", "3"]

    upsert_dense(texts, ids)

    query = "search in vector database"
    results = search_dense(query)
    for r in results:
        print(r)

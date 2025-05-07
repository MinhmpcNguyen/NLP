# hybrid_search.py

import pinecone
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# Init Pinecone
pinecone.init(
    api_key="YOUR_API_KEY",  # <-- điền vào
    environment="YOUR_ENVIRONMENT",
)
index = pinecone.Index("your-index-name")  # <-- điền tên index

# Dense model
dense_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Sparse model (TF-IDF)
tfidf_vectorizer = TfidfVectorizer()


def embed_dense(texts):
    return dense_model.encode(texts, normalize_embeddings=True)


def embed_sparse(texts):
    tfidf_sparse = tfidf_vectorizer.fit_transform(texts)
    sparse_dicts = []
    for vector in tfidf_sparse:
        indices = vector.indices
        values = vector.data
        sparse_dict = {"indices": indices.tolist(), "values": values.tolist()}
        sparse_dicts.append(sparse_dict)
    return sparse_dicts


def upsert_dense_sparse(texts, ids):
    dense_vectors = embed_dense(texts)
    sparse_vectors = embed_sparse(texts)

    vectors = []
    for idx, (dense_vec, sparse_vec) in enumerate(zip(dense_vectors, sparse_vectors)):
        vectors.append(
            {
                "id": ids[idx],
                "values": dense_vec.tolist(),
                "sparse_values": sparse_vec,
                "metadata": {"text": texts[idx]},
            }
        )

    index.upsert(vectors=vectors)


def search_hybrid_rrf(query, top_k=5):
    dense_query = embed_dense([query])[0]
    sparse_query = embed_sparse([query])[0]

    # Dense search
    res_dense = index.query(
        vector=dense_query.tolist(), top_k=top_k * 2, include_metadata=True
    )

    # Sparse search
    res_sparse = index.query(
        sparse_vector=sparse_query, top_k=top_k * 2, include_metadata=True
    )

    # RRF Fusion
    rrf_scores = {}
    for rank, match in enumerate(res_dense.matches):
        rrf_scores[match.id] = rrf_scores.get(match.id, 0) + 1 / (rank + 1)
    for rank, match in enumerate(res_sparse.matches):
        rrf_scores[match.id] = rrf_scores.get(match.id, 0) + 1 / (rank + 1)

    final_ranking = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for id_, score in final_ranking[:top_k]:
        for match in res_dense.matches + res_sparse.matches:
            if match.id == id_:
                results.append(
                    {"id": match.id, "text": match.metadata["text"], "score": score}
                )
                break

    return results


if __name__ == "__main__":
    # Ví dụ dùng
    texts = ["Hust university", "Pinecone vector search", "Machine learning is fun"]
    ids = ["1", "2", "3"]

    upsert_dense_sparse(texts, ids)

    query = "vector search hybrid"
    results = search_hybrid_rrf(query)
    for r in results:
        print(r)

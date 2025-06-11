import asyncio
import os
import time
from typing import List, Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import tiktoken
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

# === Global state ===
embedding_cache = {}
encoding = tiktoken.get_encoding("cl100k_base")


# === Embedding + similarity ===
async def create_embedding(model, paragraph: str):
    if paragraph in embedding_cache:
        return embedding_cache[paragraph]

    embedding = model.encode(paragraph, normalize_embeddings=True)
    embedding = np.array(embedding)
    embedding_cache[paragraph] = embedding
    return embedding


def cosine_sim(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    vec_a = vec_a.reshape(1, -1)
    vec_b = vec_b.reshape(1, -1)
    return float(cosine_similarity(vec_a, vec_b)[0][0])


async def compute_advanced_similarities(
    model,
    paragraphs: List[str],
    num_similarity_paragraphs_lookahead: int = 8,
    logging: bool = False,
):
    if len(paragraphs) < 2:
        return {"similarities": [], "average": 0.0, "variance": 0.0}

    embeddings = await asyncio.gather(
        *[create_embedding(model, paragraph) for paragraph in paragraphs]
    )
    similarities = []
    similarity_sum = 0

    for i in range(len(embeddings) - 1):
        max_similarity = cosine_sim(embeddings[i], embeddings[i + 1])

        if logging:
            print(f"\nSimilarity scores for paragraph {i}:")
            print(f"Base similarity with next paragraph: {max_similarity}")

        for j in range(
            i + 2, min(i + num_similarity_paragraphs_lookahead + 1, len(embeddings))
        ):
            sim = cosine_sim(embeddings[i], embeddings[j])
            if logging:
                print(f"Similarity with paragraph {j}: {sim}")
            max_similarity = max(max_similarity, sim)

        similarities.append(max_similarity)
        similarity_sum += max_similarity

    average = similarity_sum / len(similarities)
    variance = np.var(similarities)

    return {"similarities": similarities, "average": average, "variance": variance}


# === Threshold ===
def adjust_threshold(
    average: float,
    variance: float,
    base_threshold: float = 0.4,
    lower_bound: float = 0.2,
    upper_bound: float = 0.8,
    variance_lower: float = 0.01,
    variance_upper: float = 0.05,
    average_lower: float = 0.3,
    average_upper: float = 0.7,
    decrease_by: float = 0.1,
    increase_by: float = 0.1,
):
    if lower_bound >= upper_bound:
        raise ValueError("Invalid bounds: lower_bound must be less than upper_bound.")

    adjusted_threshold = base_threshold
    if variance < variance_lower:
        adjusted_threshold -= decrease_by
    elif variance > variance_upper:
        adjusted_threshold += increase_by

    if average < average_lower:
        adjusted_threshold += increase_by / 2
    elif average > average_upper:
        adjusted_threshold -= decrease_by / 2

    return min(max(adjusted_threshold, lower_bound), upper_bound)


# === Chunking ===
async def create_chunks(
    paragraphs: List[str],
    similarities: Optional[List[float]],
    similarity_threshold: float,
    max_length: int = 1024,
    min_length: int = 50,
    ultimate_max_length: int = 2048,
    logging: bool = False,
) -> List[str]:
    chunks = []
    current_chunk = [paragraphs[0]]
    chunk_length = len(encoding.encode(paragraphs[0]))

    for i in range(1, len(paragraphs)):
        next_paragraph = paragraphs[i]
        next_length = len(encoding.encode(next_paragraph))
        similarity = similarities[i - 1] if similarities else None
        combined_length = chunk_length + next_length

        should_merge = (
            (similarity is None or similarity >= similarity_threshold)
            and (
                combined_length <= max_length
                or chunk_length < min_length
                or next_length < min_length
            )
            and combined_length <= ultimate_max_length
        )

        if logging:
            print(f"\n--- Paragraph {i} ---")
            print(f"Similarity: {similarity}")
            print(f"Chunk length: {chunk_length}")
            print(f"Next length: {next_length}")
            print(f"Combined length: {combined_length}")
            print(f"Should merge: {should_merge}")

        if should_merge:
            current_chunk.append(next_paragraph)
            chunk_length += next_length
        else:
            chunks.append(" ".join(current_chunk).strip())
            current_chunk = [next_paragraph]
            chunk_length = next_length

    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return chunks


def remove_duplicate_chunks(data):
    seen_chunks = {}
    filtered_data = []

    for doc in data:
        new_chunks = []
        for chunk in doc["Chunks"]:
            content = chunk["content"].strip()
            if content in seen_chunks:
                print(
                    f"🗑️ Removing duplicate chunk in {doc['Url']} (already exists in {seen_chunks[content]})"
                )
                continue
            seen_chunks[content] = doc["Url"]
            new_chunks.append(chunk)

        if new_chunks:
            filtered_data.append({"Url": doc["Url"], "Chunks": new_chunks})

    return filtered_data


# === Public function to use from outside ===
async def extract_chunks_from_paragraphs(model, paragraphs: List[str]) -> List[str]:
    start_time = time.time()
    print("🚀 Starting chunking for multiple paragraphs...")

    if not isinstance(paragraphs, list) or not all(
        isinstance(p, str) for p in paragraphs
    ):
        print("⚠️ Input is not a valid list of strings.")
        return []

    # Gộp tất cả câu từ các đoạn
    sentences = []
    for para in paragraphs:
        para = para.strip()
        if para:
            sentences.extend(sent_tokenize(para))

    if not sentences:
        print("⚠️ No valid sentences extracted from input.")
        return []

    # Tính toán độ tương đồng
    similarity_results = await compute_advanced_similarities(model, sentences)
    threshold = adjust_threshold(
        similarity_results["average"], similarity_results["variance"]
    )

    # Chunking
    chunks = await create_chunks(
        sentences, similarity_results["similarities"], threshold
    )

    print(f"{len(chunks)} chunks extracted from paragraphs.")
    print(f"Time taken: {time.time() - start_time:.2f} seconds.")
    return chunks

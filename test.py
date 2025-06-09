from api_search import MODEL_NAME, search_rrf
from sentence_transformers import SentenceTransformer


def main():
    # Load model
    print("🚀 Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    # Nhập truy vấn
    query = input("🔍 Nhập truy vấn cần tìm: ").strip()

    # Gọi hàm search_rrf
    results = search_rrf(model, query, top_k=5)

    # In kết quả
    print(f"\n📊 Kết quả RRF cho truy vấn: {query}")
    for i, r in enumerate(results, 1):
        print(f"{i}. 🔹 Score: {r['score']:.4f}")
        print(f"📄 Text: {r['text'][:200]}...")
        print(f"🌐 URL: {r['url']}")
        print("—" * 50)


if __name__ == "__main__":
    main()

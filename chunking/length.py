import json


def chunk_by_length(content_list, max_words=1024, min_words=30):
    chunks = []
    current_chunk = []
    current_len = 0

    for para in content_list:
        para = para.strip()
        word_count = len(para.split())

        if word_count == 0:
            continue

        if current_len + word_count <= max_words:
            current_chunk.append(para)
            current_len += word_count
        else:
            if current_len >= min_words:
                chunks.append(" ".join(current_chunk))
                current_chunk = [para]
                current_len = word_count
            else:
                current_chunk.append(para)
                current_len += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ✅ File paths
input_file = "NLP/crawl_data/processed_results/final_result/final_unique.json"
output_file = "NLP/chunking/length/len.json"
skipped_file = "NLP/chunking/length/skipped_urls.json"

# ✅ Load data (check if flatten is needed)
with open(input_file, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

if isinstance(raw_data[0], list):
    flattened = [doc for group in raw_data for doc in group]
else:
    flattened = raw_data

# ✅ Chunking & tracking skipped URLs
output = []
skipped_urls = []

for doc in flattened:
    chunks = chunk_by_length(doc["content"], max_words=200, min_words=30)

    if not chunks:
        skipped_urls.append(doc["url"])
        print(f"⚠️ No valid chunks for URL: {doc['url']}")
        continue

    output.append(
        {"url": doc["url"], "chunks": [{"content": chunk} for chunk in chunks]}
    )

# ✅ Save valid output
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

# ✅ Save skipped URLs
with open(skipped_file, "w", encoding="utf-8") as f:
    json.dump(skipped_urls, f, indent=4, ensure_ascii=False)

# ✅ Summary
print(f"✅ Chunked data saved to {output_file}")
print(f"🗂️ Skipped URLs saved to {skipped_file}")
print(
    f"📄 Total input: {len(flattened)} | ✅ Kept: {len(output)} | ❌ Skipped: {len(skipped_urls)}"
)

import json

# ==== File paths ====
input_path = "NLP/crawl_data/processed_results/final_result/final_merge.json"
output_path = "NLP/crawl_data/processed_results/final_result/final_unique.json"
removed_path = "NLP/crawl_data/processed_results/final_result/removed_urls.txt"

# ==== Load and flatten ====
with open(input_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Flatten nested structure
flattened_data = []
for group in raw_data:
    if isinstance(group, list):
        flattened_data.extend(group)
    else:
        flattened_data.append(group)

# ==== Remove duplicate content globally ====
seen_contents = set()
unique_data = []
removed_urls = []

for doc in flattened_data:
    url = doc.get("url", "")
    contents = doc.get("content", [])

    # Remove duplicates
    unique_contents = []
    for para in contents:
        para_clean = para.strip()
        if para_clean and para_clean not in seen_contents:
            seen_contents.add(para_clean)
            unique_contents.append(para_clean)

    if unique_contents:
        unique_data.append({"url": url, "content": unique_contents})
    else:
        removed_urls.append(url)

# ==== Save output ====
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(unique_data, f, indent=4, ensure_ascii=False)

with open(removed_path, "w", encoding="utf-8") as f:
    f.write("\n".join(removed_urls))

print(f"✅ Saved {len(unique_data)} documents with unique contents to {output_path}")
print(
    f"🗑️ Removed {len(removed_urls)} URLs with no remaining unique content. Logged in {removed_path}"
)

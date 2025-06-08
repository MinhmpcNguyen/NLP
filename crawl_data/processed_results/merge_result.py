import json
from typing import Dict, List


def merge_url_data(data1: List[Dict], data2: List[Dict]) -> List[Dict]:
    merged = {}

    # Đưa tất cả URL từ data1 vào dict
    for entry in data1:
        merged[entry["url"]] = entry["content"]

    # So sánh với data2
    for entry in data2:
        url = entry["url"]
        content = entry["content"]

        if url not in merged:
            merged[url] = content
        else:
            # Giữ content có số lượng nhiều hơn
            if len(content) > len(merged[url]):
                merged[url] = content

    # Chuyển dict về list format
    merged_list = [{"url": url, "content": merged[url]} for url in sorted(merged)]
    return merged_list


# =======================================
# ✅ MAIN: Gộp file1 + file2 thành file output
# =======================================
def main():
    file1 = "NLP/crawl_data/processed_results/ts.json"
    file2 = "NLP/crawl_data/processed_results/final_result/merged_output1.json"
    output_file = "NLP/crawl_data/processed_results/final_result/final_merge.json"

    with open(file1, "r", encoding="utf-8") as f1:
        data1 = json.load(f1)

    with open(file2, "r", encoding="utf-8") as f2:
        data2 = json.load(f2)

    merged_data = merge_url_data(data1, data2)

    with open(output_file, "w", encoding="utf-8") as out:
        json.dump(merged_data, out, indent=4, ensure_ascii=False)

    print(f"✅ Merged output saved to: {output_file}")


if __name__ == "__main__":
    main()

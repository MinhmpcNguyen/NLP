import json
from typing import Any, Dict, List


def extract_content_from_markdown(markdown: List[Any]) -> List[str]:
    """
    Làm phẳng danh sách markdown và gắn prefix nếu là file (dict).
    """
    flat_content = []
    for item in markdown:
        if isinstance(item, list):
            flat_content.extend(item)
        elif isinstance(item, dict):
            for key, values in item.items():
                for v in values:
                    flat_content.append(f"{key}: {v}")
    return flat_content


def reformat_and_deduplicate(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Bước 1: gom tất cả content lại với thông tin URL + depth
    all_contents = {}  # content -> (url, depth)

    url_to_content_raw = {}  # url -> list of raw content
    url_to_depth = {}

    for entry in data:
        url = entry["url"]
        depth = entry.get("depth", 0)
        url_to_depth[url] = depth
        content_list = extract_content_from_markdown(entry.get("markdown", []))
        url_to_content_raw.setdefault(url, []).extend(content_list)

    # Bước 2: lọc content để giữ URL có depth nhỏ nhất
    for url, contents in url_to_content_raw.items():
        for content in contents:
            if content not in all_contents:
                all_contents[content] = (url, url_to_depth[url])
            else:
                existing_url, existing_depth = all_contents[content]
                # Giữ lại content tại URL có depth nhỏ hơn
                if url_to_depth[url] < existing_depth:
                    all_contents[content] = (url, url_to_depth[url])

    # Bước 3: gom lại content hợp lệ theo url
    filtered_result = {}
    for content, (url, _) in all_contents.items():
        filtered_result.setdefault(url, set()).add(content)

    # Bước 4: format lại thành list và remove trùng trong cùng URL
    final_result = []
    for url, content_set in filtered_result.items():
        final_result.append({"url": url, "content": sorted(content_set)})

    return final_result


# =============================================
# ✅ MAIN: Load input, process, and save output
# =============================================
def main():
    input_path = "crawl_data/crawl_results/results_http_hust.json"
    output_path = "NLP/crawl_data/processed_results/http_test.json"

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    reformatted = reformat_and_deduplicate(data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(reformatted, f, indent=4, ensure_ascii=False)

    print(f"✅ Reformatted output saved to {output_path}")


if __name__ == "__main__":
    main()

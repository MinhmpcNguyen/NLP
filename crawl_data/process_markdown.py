import re

import markdown
from bs4 import BeautifulSoup


def process_text(txt: str) -> list:
    """Processes markdown-style annotated text into key-value pairs."""
    segments = re.split(r"_\*\*", txt.strip())
    result = []

    for segment in segments:
        try:
            parts = re.split(r"\*\*_\n", segment, maxsplit=1)
            key = parts[0].strip()

            if len(parts) <= 1 or not parts[1].strip():
                processed = segment.strip()
            else:
                value = parts[1].strip()
                processed = {key: value}
        except Exception as e:
            processed = {"Error": str(e), "RawText": segment}

        result.append(processed)

    return result


def preprocess_markdown(md_text: str) -> list:
    """Converts structured markdown into a list of dictionaries by headers."""
    if "##" not in md_text:
        return process_text(md_text)

    sections = re.split(r"##\s*", md_text)
    result = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        parts = section.split("\n", 1)
        header = parts[0].strip()
        content = parts[1].strip() if len(parts) == 2 else ""

        if not header:
            continue

        clean_header = re.sub(r"\*+", "", header).strip()
        clean_content = (
            BeautifulSoup(markdown.markdown(content), "html.parser")
            .get_text(separator=" ")
            .strip()
        )

        result.append({clean_header: process_text(clean_content)})

    return result


def divide_to_paragraphs(url_content: str) -> list:
    """Splits Markdown content into logical sections, preserving pre-header content."""
    pattern = re.compile(r"^(#{1,})\s*(.*)", re.MULTILINE)
    lines = url_content.split("\n")
    sections = []

    first_header_index = next(
        (i for i, line in enumerate(lines) if pattern.match(line)), None
    )

    if first_header_index and first_header_index > 0:
        pre_header_content = [
            line for line in lines[:first_header_index] if line.strip()
        ]
        if pre_header_content:
            sections.append("\n".join(pre_header_content))

    current_headers = []
    current_content = []

    start_index = first_header_index if first_header_index is not None else 0

    for i in range(start_index, len(lines)):
        line = lines[i]
        match = pattern.match(line)

        if match:
            if current_content and current_headers:
                section = "\n".join(current_headers) + "\n" + "\n".join(current_content)
                sections.append(section)
                current_content = []
                current_headers = []

            current_headers = [line]
        else:
            if line.strip():
                current_content.append(line)

    if current_headers or current_content:
        if current_headers:
            section = "\n".join(current_headers) + "\n" + "\n".join(current_content)
        else:
            section = "\n".join(current_content)
        sections.append(section)

    return sections

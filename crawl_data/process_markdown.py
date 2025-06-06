import re

import markdown
from bs4 import BeautifulSoup


# Process paragraph text
def process_text(txt: str) -> list:
    # Split paragraph by highlight part `_**`
    h_p = re.split(r"_\*\*", txt.strip())
    t = []

    for text in h_p:
        try:
            # Split each section into key-value pairs at `**_`
            matches = re.split(r"\*\*_\n", text, maxsplit=1)
            key = matches[0].strip()

            # If there's no value, store raw text instead
            if len(matches) <= 1 or not matches[1].strip():
                section = text.strip()  # Store full raw text
            else:
                value = matches[1].strip()
                section = {key: value}
        except Exception as e:
            section = {"Error": str(e), "RawText": text}  # Error handling

        t.append(section)

    return t


# Process full markdown
def preprocess_markdown(md_text: str) -> list:
    # Check if markdown contains headers (##)
    if "##" not in md_text:
        return process_text(md_text)

    # Split by "##" but keep meaningful content
    raw_sections = re.split(r"##\s*", md_text)

    md = []

    for section in raw_sections:
        d = {}
        section = section.strip()
        if not section:  # Skip empty sections
            continue

        # Extract the first line as the header, remaining as content
        parts = section.split("\n", 1)
        if len(parts) == 2:
            header, content = parts[0].strip(), parts[1].strip()
        else:
            header, content = parts[0].strip(), ""

        # Ignore completely empty headers
        if not header:
            continue

        # Remove asterisks (**) from headers
        clean_header = re.sub(r"\*+", "", header).strip()

        # Convert markdown content to plain text
        clean_content = (
            BeautifulSoup(markdown.markdown(content), "html.parser")
            .get_text(separator=" ")
            .strip()
        )
        d[clean_header] = process_text(clean_content)
        md.append(d)

    return md


import re


def divide_to_paragraphs(url_content: str):
    """Splits Markdown content into sections based on headers and their content."""

    # Regex pattern to match Markdown headers
    pattern = re.compile(r"^(#{1,})\s*(.*)", re.MULTILINE)

    sections = []
    current_headers = []  # Stores accumulated headers without content
    current_content = []  # Stores content for current section

    for line in url_content.split("\n"):
        match = pattern.match(line)
        if match:  # If the line is a header
            header_level = len(match.group(1))

            # Check if we have content to save with accumulated headers
            if current_content:
                # Save section with all accumulated headers and content
                if current_headers:
                    section = (
                        "\n".join(current_headers) + "\n" + "\n".join(current_content)
                    )
                    sections.append(section)
                current_headers = []
                current_content = []

            # Add current header to accumulated headers
            current_headers.append(line)

        else:
            # Add content to current section
            if line.strip():  # Only add non-empty lines
                current_content.append(line)

    # Handle remaining headers and content
    if current_headers:
        if current_content:
            # Headers with content
            section = "\n".join(current_headers) + "\n" + "\n".join(current_content)
        else:
            # Headers without content - merge them all
            section = "\n".join(current_headers)
        sections.append(section)

    return sections

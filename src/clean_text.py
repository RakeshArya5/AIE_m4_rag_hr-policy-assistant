"""
clean_text.py - Clean the text of every loaded page.

RAG pipeline step 2 of 7: CLEANING.

Why: text that comes out of a PDF is messy. It has a header repeated on
every page, strange bullet symbols, extra spaces and many blank lines.
Clean text gives better chunks, better search results and better answers.

Cleaning steps (one small function each):
    1. remove the repeated header line "SPIL Corporate HR Policies"
    2. fix broken characters with the ftfy library
    3. replace odd bullet symbols and dashes with a simple "-"
    4. put every "Section N: Title" heading on its own line, in one standard format
    5. turn many spaces into one, and many blank lines into one
    6. drop pages that are empty after cleaning

The cleaned pages are saved to data/processed/clean_pages.json so you can open and read them.

How to run (from the project root):
    python -m src.clean_text
"""

import json
import re

import ftfy
from langchain_core.documents import Document

from src.config import PROCESSED_DATA_FOLDER
from src.load_documents import load_all_documents, find_page


# The header line that the PDF repeats at the top of every page.
HEADER_LINE = "SPIL Corporate HR Policies"

# Odd characters left over from the PDF, and what we replace them with.
# "\uf0b7" is the bullet from the Windows "Symbol" font: it shows up as a strange box,
# so we write it with its code number instead of pasting the invisible character.
ODD_CHARACTERS = {
    "\uf0b7": "-",   # Symbol-font bullet
    "•": "-",        # normal round bullet
    "▪": "-",        # small square bullet
    "–": "-",        # en dash
    "—": "-",        # em dash
}


def remove_header(text):
    """Step 1: remove every line that is exactly the repeated header."""
    kept_lines = []
    for line in text.split("\n"):
        # strip() removes the spaces around the line before we compare it.
        if line.strip() != HEADER_LINE:
            kept_lines.append(line)
    return "\n".join(kept_lines)


def fix_broken_characters(text):
    """Step 2: fix broken characters (for example 'â€™' becomes an apostrophe)."""
    # ftfy ("fixes text for you") repairs text that was decoded with the wrong encoding.
    # It also turns curly quotes like “ ” ’ into plain quotes " and '.
    return ftfy.fix_text(text)


def replace_odd_characters(text):
    """Step 3: replace odd bullets and dashes with a simple '-'."""
    for odd_character, simple_character in ODD_CHARACTERS.items():
        text = text.replace(odd_character, simple_character)
    return text


def standardize_section_headings(text):
    """Step 4: rewrite every heading as 'Section N: Title' on its own line."""
    # The PDF writes headings in different ways:
    #   "Section 6: Attendance", "Section 13:Disciplinary", "Section : 20  Review"
    # This pattern finds a line that STARTS with "Section", then an optional ":",
    # then the number, then another optional ":", then the title.
    heading_pattern = r"^[ \t]*Section[ \t]*:?[ \t]*(\d+)[ \t]*:?[ \t]*(.*)$"

    # We write it back in one standard format, with a blank line before and after,
    # so the heading is never glued to the paragraph around it.
    standard_heading = r"\nSection \1: \2\n"

    # re.MULTILINE makes "^" and "$" match at the start and end of EVERY line.
    return re.sub(heading_pattern, standard_heading, text, flags=re.MULTILINE)


def remove_extra_whitespace(text):
    """Step 5: one space between words, at most one blank line between paragraphs."""
    clean_lines = []
    for line in text.split("\n"):
        # Turn 2 or more spaces/tabs into a single space, then trim both ends.
        line = re.sub(r"[ \t]+", " ", line)
        clean_lines.append(line.strip())
    text = "\n".join(clean_lines)

    # Turn 3 or more line breaks (2+ blank lines) into exactly one blank line.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_page_text(text):
    """Run cleaning steps 1 to 5 on the text of one page, in order."""
    text = remove_header(text)
    text = fix_broken_characters(text)
    text = replace_odd_characters(text)
    text = standardize_section_headings(text)
    text = remove_extra_whitespace(text)
    return text


def clean_documents(documents):
    """Clean every Document. Step 6: drop pages that are empty after cleaning."""
    clean_pages = []
    for document in documents:
        clean_text = clean_page_text(document.page_content)

        # Step 6: skip the page if nothing is left.
        if clean_text == "":
            print("  Dropping empty page:", document.metadata)
            continue

        # Make a new Document with the clean text and a copy of the same metadata.
        clean_page = Document(page_content=clean_text, metadata=dict(document.metadata))
        clean_pages.append(clean_page)
    return clean_pages


def save_clean_pages(clean_pages):
    """Save the clean pages to data/processed/clean_pages.json. Returns the file path."""
    # Create the folder if it does not exist yet.
    PROCESSED_DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    # Turn each Document into a plain dictionary that JSON can store.
    pages_as_dictionaries = []
    for page in clean_pages:
        pages_as_dictionaries.append({
            "source": page.metadata["source"],
            "page": page.metadata["page"],
            "text": page.page_content,
        })

    # indent=2 makes the file easy to read; ensure_ascii=False keeps characters like ₹ as they are.
    output_path = PROCESSED_DATA_FOLDER / "clean_pages.json"
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(pages_as_dictionaries, output_file, indent=2, ensure_ascii=False)
    return output_path


def count_characters_per_file(documents):
    """Return a dictionary: file name -> total number of characters in that file."""
    characters_per_file = {}
    for document in documents:
        file_name = document.metadata["source"]
        old_total = characters_per_file.get(file_name, 0)
        characters_per_file[file_name] = old_total + len(document.page_content)
    return characters_per_file


def print_page_preview(title, page, number_of_characters):
    """Print the first characters of one page between two lines."""
    print()
    print(title)
    print("-" * 60)
    print(page.page_content[:number_of_characters])
    print("-" * 60)


# ---------- Run this file on its own to see what it does ----------
if __name__ == "__main__":
    # Step 1: load the raw pages (pipeline step 1).
    print("Step 1: Loading documents")
    handbook_pages = load_all_documents()

    # Step 2: clean every page.
    print()
    print("Step 2: Cleaning", len(handbook_pages), "pages")
    clean_pages = clean_documents(handbook_pages)
    print("  Pages kept after cleaning:", len(clean_pages))

    # Step 3: compare the number of characters before and after, per file.
    print()
    print("Step 3: Characters before and after cleaning")
    characters_before = count_characters_per_file(handbook_pages)
    characters_after = count_characters_per_file(clean_pages)
    for file_name in characters_before:
        before = characters_before[file_name]
        after = characters_after.get(file_name, 0)
        print("  " + file_name + ":", before, "->", after, "(removed", before - after, ")")

    # Step 4: show page 7 before and after, and page 20 (the table) after.
    pdf_name = "HR_Policies_Manuals_RAG.pdf"
    print_page_preview("Step 4: Page 7 BEFORE cleaning", find_page(handbook_pages, pdf_name, 7), 400)
    print_page_preview("        Page 7 AFTER cleaning", find_page(clean_pages, pdf_name, 7), 400)
    print_page_preview("        Page 20 AFTER cleaning (grade structure table)", find_page(clean_pages, pdf_name, 20), 1200)

    # Step 5: save the clean pages as JSON.
    print()
    output_path = save_clean_pages(clean_pages)
    print("Step 5: Saved clean pages to", output_path)

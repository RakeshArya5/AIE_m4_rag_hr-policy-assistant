"""
chunk_text.py - Split the clean, safe pages into small chunks.

RAG pipeline step 4 of 7: CHUNKING.

Why: a whole page is too big and talks about many things. Small chunks
(about 800 characters) are easier to search: the question is matched
against each chunk, and only the best few are sent to the LLM.

Each chunk gets this metadata, so we can cite it later:
    source          -> the file name
    page            -> the page number
    section         -> the latest "Section N: Title" heading seen so far
    effective_date  -> from "Effective date: ..." in the file, or the default from config

We first cut each page at its section headings, and then chunk each piece.
That way one chunk never mixes the end of one section with the start of the next.

The chunks are saved to data/processed/chunks.json so you can open and read them.

How to run (from the project root):
    python -m src.chunk_text
"""

import json
import re

from dateutil import parser as date_parser
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP, DEFAULT_EFFECTIVE_DATE, PROCESSED_DATA_FOLDER
from src.load_documents import load_all_documents
from src.clean_text import clean_documents
from src.privacy_check import hide_personal_data_in_documents


# The section name we use before the first "Section N:" heading (or in files that have none).
NO_SECTION = "General"


def is_section_heading(line):
    """Return True if the line looks like 'Section 6: Attendance & Leave Management'."""
    # clean_text.py already wrote every heading in this exact format.
    return re.match(r"^Section \d+: .+$", line) is not None


def find_effective_date(file_pages):
    """Find 'Effective date: ...' in a file and return it as YYYY-MM-DD (or the default)."""
    for page in file_pages:
        match = re.search(r"Effective date:\s*(.+)", page.page_content, re.IGNORECASE)
        if match:
            # dateutil understands many date styles, like "1 April 2026" or "01/04/2026".
            # dayfirst=True means "01/04/2026" is read as 1 April (the Indian style).
            date = date_parser.parse(match.group(1).strip(), dayfirst=True)
            return date.strftime("%Y-%m-%d")
    return DEFAULT_EFFECTIVE_DATE


def split_page_into_sections(page_text, current_section):
    """Cut one page into pieces at its headings. Returns a list of {"section", "text"}."""
    pieces = [{"section": current_section, "text": ""}]
    for line in page_text.split("\n"):
        # A heading starts a new piece that belongs to the new section.
        if is_section_heading(line):
            pieces.append({"section": line, "text": ""})
        pieces[-1]["text"] = pieces[-1]["text"] + line + "\n"

    # Drop the first piece if the page started directly with a heading (it is empty).
    non_empty_pieces = []
    for piece in pieces:
        if piece["text"].strip() != "":
            non_empty_pieces.append(piece)
    return non_empty_pieces


def chunk_one_file(file_pages, text_splitter):
    """Chunk all pages of ONE file, in page order. Returns a list of chunk Documents."""
    effective_date = find_effective_date(file_pages)
    current_section = NO_SECTION
    file_chunks = []

    for page in file_pages:
        for piece in split_page_into_sections(page.page_content, current_section):
            # split_text() returns a list of strings, each at most CHUNK_SIZE characters.
            for chunk_text in text_splitter.split_text(piece["text"]):
                metadata = {
                    "source": page.metadata["source"],
                    "page": page.metadata["page"],
                    "section": piece["section"],
                    "effective_date": effective_date,
                }
                file_chunks.append(Document(page_content=chunk_text, metadata=metadata))
            # Remember the section, because the next page continues it.
            current_section = piece["section"]
    return file_chunks


def chunk_documents(pages):
    """Chunk the pages of every file. Returns one list of chunk Documents."""
    # RecursiveCharacterTextSplitter tries to cut at blank lines first, then at
    # line breaks, then at spaces, and only as a last choice in the middle of a word.
    # So chunks usually end at the end of a paragraph or sentence.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # Find the file names, in the order they appear.
    file_names = []
    for page in pages:
        if page.metadata["source"] not in file_names:
            file_names.append(page.metadata["source"])

    # Chunk each file on its own, so sections never run from one file into another.
    all_chunks = []
    for file_name in file_names:
        file_pages = []
        for page in pages:
            if page.metadata["source"] == file_name:
                file_pages.append(page)
        all_chunks.extend(chunk_one_file(file_pages, text_splitter))
    return all_chunks


def save_chunks(chunks):
    """Save the chunks to data/processed/chunks.json. Returns the file path."""
    PROCESSED_DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    # Turn each chunk into a plain dictionary: its metadata plus its text.
    chunks_as_dictionaries = []
    for chunk in chunks:
        chunk_dictionary = dict(chunk.metadata)
        chunk_dictionary["text"] = chunk.page_content
        chunks_as_dictionaries.append(chunk_dictionary)

    output_path = PROCESSED_DATA_FOLDER / "chunks.json"
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(chunks_as_dictionaries, output_file, indent=2, ensure_ascii=False)
    return output_path


# ---------- Run this file on its own to see what it does ----------
if __name__ == "__main__":
    # Steps 1 to 3 of the pipeline: load, clean, hide personal data.
    print("Step 1: Loading, cleaning and hiding personal data")
    clean_pages = clean_documents(load_all_documents())
    safe_pages, privacy_report = hide_personal_data_in_documents(clean_pages)

    # Step 4: chunk.
    print()
    print("Step 2: Chunking with chunk size", CHUNK_SIZE, "and overlap", CHUNK_OVERLAP)
    chunks = chunk_documents(safe_pages)
    print("  Made", len(chunks), "chunks from", len(safe_pages), "pages")

    # Show one example chunk per interesting place: the start, and the leave rules on page 9.
    for chunk_number in [0, 1]:
        print()
        print("Step 3: Example chunk number", chunk_number)
        print("  Metadata:", chunks[chunk_number].metadata)
        print("  Text:", chunks[chunk_number].page_content[:300])
    for chunk in chunks:
        if "2 leaves in a month" in chunk.page_content:
            print()
            print("Step 3: The chunk with the monthly leave rule")
            print("  Metadata:", chunk.metadata)
            print("  Text:", chunk.page_content[:300])

    # Step 5: save the chunks as JSON.
    print()
    print("Step 4: Saved chunks to", save_chunks(chunks))

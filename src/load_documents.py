"""
load_documents.py - Load every HR document from data/raw into LangChain Documents.

RAG pipeline step 1 of 7: LOADING.

A LangChain "Document" is a simple object with two parts:
    page_content -> the text
    metadata     -> a small dictionary with extra info (here: "source" and "page")

We support three file types:
    .pdf  -> one Document per page
    .docx -> one Document for the whole file (page = 1)
    .md   -> one Document for the whole file (page = 1)

How to run (from the project root):
    python -m src.load_documents
"""

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

from src.config import RAW_DATA_FOLDER


def load_pdf(file_path):
    """Load a PDF file. Returns one Document per page."""
    # PyPDFLoader opens the PDF with the "pypdf" library and reads the text of each page.
    # It gives back a list with one Document per page.
    loader = PyPDFLoader(str(file_path))
    pdf_pages = loader.load()

    # PyPDFLoader counts pages from 0 and stores the full file path.
    # We want pages to start at 1 and "source" to be only the file name.
    for page in pdf_pages:
        page.metadata = {
            "source": file_path.name,
            "page": page.metadata["page"] + 1,
        }
    return pdf_pages


def load_word(file_path):
    """Load a Word (.docx) file. Returns a list with one Document."""
    # Docx2txtLoader uses the "docx2txt" library to pull all the text out of the Word file.
    # Word files have no fixed pages, so we get one Document for the whole file.
    loader = Docx2txtLoader(str(file_path))
    word_documents = loader.load()

    # Set our own simple metadata: the file name and page 1.
    for document in word_documents:
        document.metadata = {"source": file_path.name, "page": 1}
    return word_documents


def load_markdown(file_path):
    """Load a Markdown (.md) file. Returns a list with one Document."""
    # TextLoader simply reads the file as plain text.
    # We say encoding="utf-8" so special characters (like ₹ or é) are read correctly on Windows.
    loader = TextLoader(str(file_path), encoding="utf-8")
    markdown_documents = loader.load()

    # Set our own simple metadata: the file name and page 1.
    for document in markdown_documents:
        document.metadata = {"source": file_path.name, "page": 1}
    return markdown_documents


def load_one_file(file_path):
    """Pick the right loader based on the file extension."""
    # Look at the end of the file name (".pdf", ".docx", ".md") in lower case.
    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return load_pdf(file_path)
    if extension == ".docx":
        return load_word(file_path)
    if extension == ".md":
        return load_markdown(file_path)

    # Any other file type is skipped.
    print("  Skipping unsupported file:", file_path.name)
    return []


def load_all_documents(folder=RAW_DATA_FOLDER):
    """Load every supported file in the folder. Returns one list of Documents."""
    all_documents = []

    # sorted() gives the files in alphabetical order, so the output is always the same.
    for file_path in sorted(folder.iterdir()):
        if not file_path.is_file():
            continue

        # Load this file and add its Documents to the big list.
        file_documents = load_one_file(file_path)
        print("  Loaded", len(file_documents), "page(s) from", file_path.name)
        all_documents.extend(file_documents)

    return all_documents


def find_page(documents, file_name, page_number):
    """Return the Document with this file name and page number (or None)."""
    for document in documents:
        if document.metadata["source"] == file_name and document.metadata["page"] == page_number:
            return document
    return None


# ---------- Run this file on its own to see what it does ----------
if __name__ == "__main__":
    # Step 1: load everything from data/raw.
    print("Step 1: Loading documents from", RAW_DATA_FOLDER)
    handbook_pages = load_all_documents()
    print("Total Documents loaded:", len(handbook_pages))

    # Step 2: show a preview of two PDF pages so we can check the text looks right.
    pdf_name = "HR_Policies_Manuals_RAG.pdf"
    for page_number in [1, 7]:
        page = find_page(handbook_pages, pdf_name, page_number)
        print()
        print("Step 2: Preview of", pdf_name, "page", page_number)
        print("  Metadata:", page.metadata)
        print("  First 300 characters:")
        print("-" * 60)
        print(page.page_content[:300])
        print("-" * 60)

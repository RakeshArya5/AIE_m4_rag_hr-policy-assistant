"""
config.py - All project settings in ONE place.

RAG pipeline step: none (every other file reads its settings from here).

Why: if we want to change the chunk size or the model, we change it here
only, and every file picks up the new value automatically.

How to run (it just prints the settings):
    python -m src.config
"""

from pathlib import Path


# ---------- Folder paths ----------
# The project root is the folder that contains the "src" folder.
# We build every path from it, so the code works no matter where you run it from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_FOLDER = PROJECT_ROOT / "data" / "raw"              # documents in the knowledge base
UPDATES_FOLDER = PROJECT_ROOT / "data" / "updates"           # new files HR uploads later
PROCESSED_DATA_FOLDER = PROJECT_ROOT / "data" / "processed"  # cleaned pages and chunks saved as JSON
QDRANT_FOLDER = PROJECT_ROOT / "qdrant_data"                 # where Qdrant stores the vectors on disk


# ---------- OpenAI models ----------
CHAT_MODEL = "gpt-4o-mini"                     # the LLM that writes the answers
EMBEDDING_MODEL = "text-embedding-3-small"     # turns text into vectors (lists of numbers)
EMBEDDING_SIZE = 1536                          # how many numbers are in one vector from this embedding model


# ---------- Chunking ----------
CHUNK_SIZE = 800       # maximum number of characters in one chunk
CHUNK_OVERLAP = 150    # characters shared by two neighbouring chunks, so no sentence is lost at the cut
DEFAULT_EFFECTIVE_DATE = "2020-08-21"   # used when a document does not say "Effective date: ..." (the handbook's date)


# ---------- Retrieval ----------
TOP_K = 4                     # how many chunks we fetch from Qdrant for each question
SIMILARITY_THRESHOLD = 0.35   # chunks with a score below this are ignored (not relevant enough)


# ---------- Privacy ----------
ALLOWED_EMAILS = ["hr@sircapaints.com"]   # company addresses that are not personal data, so we do not hide them


# ---------- Vector database ----------
COLLECTION_NAME = "hr_policies"   # name of the "table" inside Qdrant that holds our chunks


# ---------- Run this file on its own to see the settings ----------
if __name__ == "__main__":
    print("Project settings:")
    print("  Raw data folder:      ", RAW_DATA_FOLDER)
    print("  Updates folder:       ", UPDATES_FOLDER)
    print("  Processed data folder:", PROCESSED_DATA_FOLDER)
    print("  Qdrant folder:        ", QDRANT_FOLDER)
    print("  Chat model:           ", CHAT_MODEL)
    print("  Embedding model:      ", EMBEDDING_MODEL, "(vector size", EMBEDDING_SIZE, ")")
    print("  Chunk size / overlap: ", CHUNK_SIZE, "/", CHUNK_OVERLAP)
    print("  Default effective date:", DEFAULT_EFFECTIVE_DATE)
    print("  Top-k / threshold:    ", TOP_K, "/", SIMILARITY_THRESHOLD)
    print("  Allowed emails:       ", ALLOWED_EMAILS)
    print("  Collection name:      ", COLLECTION_NAME)

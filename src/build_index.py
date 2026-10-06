"""
build_index.py - Turn the chunks into vectors and store them in Qdrant.

RAG pipeline step 5 of 7: INDEXING.

What happens here:
    1. run steps 1 to 4 (load, clean, hide personal data, chunk)
    2. send each chunk to the OpenAI embedding model, which turns it into a vector
       (a list of 1536 numbers that describes the MEANING of the text)
    3. store the vectors, text and metadata in Qdrant (a vector database)

Qdrant runs in "local folder mode": no server, it simply saves everything
in the qdrant_data/ folder. Only ONE program can open that folder at a time,
so stop the Streamlit app before running this file.

No duplicates: before adding the chunks of a file, we delete that file's
old chunks. So you can run this file again and again safely.

How to run (from the project root):
    python -m src.build_index
"""

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, FilterSelector

from src.config import QDRANT_FOLDER, COLLECTION_NAME, EMBEDDING_MODEL, EMBEDDING_SIZE
from src.load_documents import load_all_documents, load_one_file
from src.clean_text import clean_documents, save_clean_pages
from src.privacy_check import hide_personal_data_in_documents, print_privacy_report
from src.chunk_text import chunk_documents, save_chunks


# Read OPENAI_API_KEY from the .env file, so the OpenAI tools can find it.
load_dotenv()


def create_qdrant_client():
    """Open the Qdrant database that lives in the qdrant_data/ folder."""
    # path=... means "local folder mode": Qdrant stores its data in files on disk.
    return QdrantClient(path=str(QDRANT_FOLDER))


def create_collection_if_missing(qdrant_client):
    """Create the collection (like a table) the first time we run."""
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        # Each vector has EMBEDDING_SIZE numbers. COSINE means we compare vectors by
        # their angle: a score of 1 means "same meaning", 0 means "not related".
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE),
        )
        print("  Created new Qdrant collection:", COLLECTION_NAME)


def get_vector_store(qdrant_client):
    """Connect LangChain to our Qdrant collection."""
    # OpenAIEmbeddings calls the OpenAI API to turn text into vectors.
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # QdrantVectorStore is LangChain's wrapper around Qdrant. Behind the scenes it
    # embeds our text, and saves the text in "page_content" and our metadata in "metadata".
    return QdrantVectorStore(client=qdrant_client, collection_name=COLLECTION_NAME, embedding=embeddings)


def delete_old_chunks(qdrant_client, file_name):
    """Delete every chunk that came from this file, so we never store duplicates."""
    # LangChain saves our metadata under the key "metadata", so the file name is "metadata.source".
    same_file_filter = Filter(must=[FieldCondition(key="metadata.source", match=MatchValue(value=file_name))])
    qdrant_client.delete(collection_name=COLLECTION_NAME, points_selector=FilterSelector(filter=same_file_filter))


def store_chunks_of_one_file(qdrant_client, file_name, file_chunks):
    """Replace one file's chunks in Qdrant: delete the old ones, then add the new ones."""
    delete_old_chunks(qdrant_client, file_name)
    vector_store = get_vector_store(qdrant_client)
    # add_documents() embeds every chunk with OpenAI and stores it in Qdrant.
    vector_store.add_documents(file_chunks)
    print("  Stored", len(file_chunks), "chunks from", file_name)


def prepare_chunks(documents):
    """Pipeline steps 2 to 4: clean, hide personal data, chunk. Returns (chunks, privacy_report)."""
    clean_pages = clean_documents(documents)
    safe_pages, privacy_report = hide_personal_data_in_documents(clean_pages)
    chunks = chunk_documents(safe_pages)
    return chunks, privacy_report


def index_one_file(qdrant_client, file_path):
    """Index (or re-index) just ONE file. Used by the app when HR uploads a file."""
    create_collection_if_missing(qdrant_client)
    documents = load_one_file(file_path)
    chunks, privacy_report = prepare_chunks(documents)
    print_privacy_report(privacy_report)
    store_chunks_of_one_file(qdrant_client, file_path.name, chunks)
    return len(chunks), privacy_report


def chunks_of_file(chunks, file_name):
    """Return only the chunks that came from this file."""
    file_chunks = []
    for chunk in chunks:
        if chunk.metadata["source"] == file_name:
            file_chunks.append(chunk)
    return file_chunks


def build_full_index(qdrant_client):
    """Index every file in data/raw. Also saves clean_pages.json and chunks.json."""
    create_collection_if_missing(qdrant_client)

    # Steps 1 and 2: load and clean every file, and save the clean pages to look at.
    clean_pages = clean_documents(load_all_documents())
    print("  Saved clean pages to", save_clean_pages(clean_pages))

    # Steps 3 and 4: hide personal data, then chunk, and save the chunks to look at.
    safe_pages, privacy_report = hide_personal_data_in_documents(clean_pages)
    print_privacy_report(privacy_report)
    chunks = chunk_documents(safe_pages)
    print("  Saved", len(chunks), "chunks to", save_chunks(chunks))

    # Step 5: store the chunks in Qdrant, one file at a time.
    for file_name in privacy_report:
        store_chunks_of_one_file(qdrant_client, file_name, chunks_of_file(chunks, file_name))
    return len(chunks)


# ---------- Run this file on its own to see what it does ----------
if __name__ == "__main__":
    print("Step 1: Opening Qdrant in", QDRANT_FOLDER)
    qdrant_client = create_qdrant_client()

    print()
    print("Step 2: Running the pipeline and storing the chunks")
    build_full_index(qdrant_client)

    # count() tells us how many chunks (called "points" in Qdrant) are in the collection.
    print()
    total_points = qdrant_client.count(collection_name=COLLECTION_NAME).count
    print("Step 3: The collection", COLLECTION_NAME, "now holds", total_points, "chunks")

    # Close the database, so other programs (like the app) can open the folder.
    qdrant_client.close()

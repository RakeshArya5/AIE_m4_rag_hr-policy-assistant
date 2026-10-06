"""
retrieve.py - Find the chunks that best match a question.

RAG pipeline step 6 of 7: RETRIEVAL.

What happens here:
    1. the question is turned into a vector with the same embedding model
    2. Qdrant finds the TOP_K chunks whose vectors are closest to it
    3. we keep only chunks with a score at or above SIMILARITY_THRESHOLD,
       because a low score means "not really about this question"

The score is a number from 0 to 1 (cosine similarity): higher = more similar.

How to run (from the project root, with the app stopped):
    python -m src.retrieve
"""

from src.config import TOP_K, SIMILARITY_THRESHOLD
from src.build_index import create_qdrant_client, get_vector_store


def search_chunks(qdrant_client, question):
    """Return the TOP_K closest chunks as a list of (chunk, score) pairs, best first."""
    vector_store = get_vector_store(qdrant_client)
    # similarity_search_with_score() embeds the question with OpenAI, asks Qdrant
    # for the k nearest vectors, and gives back each chunk with its similarity score.
    return vector_store.similarity_search_with_score(question, k=TOP_K)


def retrieve_relevant_chunks(qdrant_client, question):
    """Search, then keep only the chunks whose score passes the threshold."""
    relevant_chunks = []
    for chunk, score in search_chunks(qdrant_client, question):
        if score >= SIMILARITY_THRESHOLD:
            relevant_chunks.append((chunk, score))
    return relevant_chunks


def print_results(results):
    """Print each chunk's score, source, page, section and the start of its text."""
    for chunk, score in results:
        metadata = chunk.metadata
        print("  score", round(score, 3), "|", metadata["source"], "| page", metadata["page"], "|", metadata["section"])
        print("     ", chunk.page_content[:120].replace("\n", " "), "...")


# ---------- Run this file on its own to see what it does ----------
if __name__ == "__main__":
    qdrant_client = create_qdrant_client()
    test_questions = [
        "How many leaves do I get per month?",
        "How often should I reset my password?",
    ]

    for question in test_questions:
        # First show ALL top-k results with their scores, then only the ones we keep.
        print()
        print("Question:", question)
        print("Step 1: Top", TOP_K, "results from Qdrant:")
        print_results(search_chunks(qdrant_client, question))

        print("Step 2: Results with score >=", SIMILARITY_THRESHOLD, ":")
        relevant_chunks = retrieve_relevant_chunks(qdrant_client, question)
        print_results(relevant_chunks)
        print("  Kept", len(relevant_chunks), "of", TOP_K, "chunks")

    qdrant_client.close()

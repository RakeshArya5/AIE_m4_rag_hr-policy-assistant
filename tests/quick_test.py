"""
quick_test.py - Ask the assistant a list of test questions and print the results.

RAG pipeline: this tests the whole pipeline end to end (steps 6 and 7 on
top of the index built in step 5).

Read each answer and check:
    - is it correct according to the handbook?
    - does it show the right source, page and section?
    - does the last question (not in the documents) say it couldn't find it?

How to run (from the project root, with the app stopped):
    python -m tests.quick_test
"""

from src.build_index import create_qdrant_client
from src.answer import answer_question, print_answer


# The questions to test. The last one is NOT in the documents on purpose.
TEST_QUESTIONS = [
    "How many leaves do I get per month?",
    "What are the office timings?",
    "How late can I come without losing leave?",
    "Can I encash unused leave?",
    "What do I get for lunch if I work on a Sunday?",
    "Which grade code is a Manager?",
    "When am I eligible for gratuity?",
    "Who do I contact for payroll questions?",
    "How often should I reset my password?",
]


# ---------- Run this file to test the assistant ----------
if __name__ == "__main__":
    qdrant_client = create_qdrant_client()

    # Ask every question and print the answer with its sources.
    question_number = 1
    for question in TEST_QUESTIONS:
        print()
        print("=" * 70)
        print("Test", question_number, "of", len(TEST_QUESTIONS))
        print_answer(question, answer_question(qdrant_client, question))
        question_number = question_number + 1

    print()
    print("Done: asked", len(TEST_QUESTIONS), "questions.")
    qdrant_client.close()

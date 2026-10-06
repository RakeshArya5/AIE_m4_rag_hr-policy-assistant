"""
answer.py - Answer a question using the retrieved chunks and the LLM.

RAG pipeline step 7 of 7: GENERATION (the "G" in RAG).

What happens here:
    1. hide personal data in the question (it goes to OpenAI too)
    2. retrieve the relevant chunks (step 6)
    3. if no chunk passes the threshold, say "I couldn't find this..." WITHOUT calling the LLM
    4. otherwise build a prompt: rules + context (chunks with source labels) + question
    5. call the LLM (gpt-4o-mini) and get the answer
    6. add the list of sources IN CODE, from the chunks' metadata
       (so the sources are always real, never invented by the LLM)

How to run (from the project root, with the app stopped):
    python -m src.answer
"""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import CHAT_MODEL
from src.build_index import create_qdrant_client
from src.retrieve import retrieve_relevant_chunks
from src.privacy_check import hide_personal_data


# Read OPENAI_API_KEY from the .env file.
load_dotenv()

# The polite reply we give when the answer is not in the documents.
NOT_FOUND_MESSAGE = "I couldn't find this in the HR policies. Please contact the HR department."

# The rules for the LLM. They are the most important part of the prompt.
SYSTEM_RULES = """You are the HR Policy Assistant of Sirca Paints India Ltd (SPIL).
Answer the employee's question using ONLY the context below.

Rules:
1. Use only facts from the context. Never use outside knowledge and never guess.
2. If the context does not contain the answer, reply with exactly this sentence and nothing else:
   """ + NOT_FOUND_MESSAGE + """
3. If two sources disagree, follow the one with the newest effective date,
   and say so, for example: "As per the update effective 2026-04-01, ..."
4. Start with the direct answer to the question (the general rule), then add
   any special cases or conditions. Keep it short, clear and friendly.
   Mention every source number you used, each in its own brackets, like [Source 1] [Source 2].
5. Personal data is hidden as labels like [EMAIL] or [PHONE]. Keep the labels; do not invent values."""

# ChatPromptTemplate is LangChain's fill-in-the-blanks prompt. {context} and {question}
# are the blanks. "system" holds our rules, "human" holds what the employee asked.
PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_RULES),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


def build_context(relevant_chunks):
    """Join the chunks into one text, each with a label saying where it came from."""
    context_parts = []
    source_number = 1
    for chunk, score in relevant_chunks:
        metadata = chunk.metadata
        label = ("[Source " + str(source_number) + "] " + metadata["source"]
                 + ", page " + str(metadata["page"]) + ", " + metadata["section"]
                 + ", effective date " + metadata["effective_date"])
        context_parts.append(label + "\n" + chunk.page_content)
        source_number = source_number + 1
    return "\n\n---\n\n".join(context_parts)


def list_sources(relevant_chunks, answer_text):
    """Make the sources list from the metadata of the chunks the answer used."""
    all_sources = []
    cited_sources = []
    source_number = 1
    for chunk, score in relevant_chunks:
        metadata = chunk.metadata
        source = {"source": metadata["source"], "page": metadata["page"], "section": metadata["section"]}
        # Skip a place we already listed (two chunks can come from the same page).
        if source not in all_sources:
            all_sources.append(source)
        # The LLM writes "[Source 2]" when it used chunk number 2.
        if "Source " + str(source_number) in answer_text and source not in cited_sources:
            cited_sources.append(source)
        source_number = source_number + 1

    # If the LLM forgot to cite, show every chunk we gave it, to be safe.
    if len(cited_sources) == 0:
        return all_sources
    return cited_sources


def ask_llm(context, question):
    """Fill in the prompt and send it to the chat model. Returns the answer text."""
    # temperature=0 makes the model stick to the most likely words: fewer made-up answers.
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    prompt_messages = PROMPT_TEMPLATE.invoke({"context": context, "question": question})
    response = llm.invoke(prompt_messages)
    return response.content.strip()


def answer_question(qdrant_client, question):
    """The full RAG answer. Returns {"answer": text, "sources": list of sources}."""
    # Step 1: hide personal data in the question before it leaves our computer.
    safe_question = hide_personal_data(question)

    # Steps 2 and 3: retrieve; if nothing is relevant, do not call the LLM at all.
    relevant_chunks = retrieve_relevant_chunks(qdrant_client, safe_question)
    if len(relevant_chunks) == 0:
        return {"answer": NOT_FOUND_MESSAGE, "sources": []}

    # Steps 4 and 5: ask the LLM with the rules, the context and the question.
    answer_text = ask_llm(build_context(relevant_chunks), safe_question)

    # Step 6: add the sources, unless the LLM said the answer is not in the context.
    if answer_text == NOT_FOUND_MESSAGE:
        return {"answer": answer_text, "sources": []}
    return {"answer": answer_text, "sources": list_sources(relevant_chunks, answer_text)}


def print_answer(question, result):
    """Print a question, its answer and its sources in a readable way."""
    print("Question:", question)
    print("Answer:  ", result["answer"])
    if len(result["sources"]) == 0:
        print("Sources:  (none)")
    for source in result["sources"]:
        print("Source:  ", source["source"], "| page", source["page"], "|", source["section"])


# ---------- Run this file on its own to see what it does ----------
if __name__ == "__main__":
    qdrant_client = create_qdrant_client()

    # One question that IS in the documents, and one that is NOT.
    for question in ["How many leaves do I get per month?", "How often should I reset my password?"]:
        print()
        print_answer(question, answer_question(qdrant_client, question))

    qdrant_client.close()

"""
app.py - The HR Policy Assistant web page, built with Streamlit.

RAG pipeline: this is the user interface on top of all 7 steps.
    Main page: type a question, see the answer and its sources (steps 6 and 7).
    Sidebar:   HR uploads a PDF, Word or Markdown file. It is saved into
               data/raw/ and ONLY that file is re-indexed (steps 1 to 5).

How Streamlit works: every time you click or type, Streamlit runs this
whole file again from top to bottom and redraws the page.

How to run (from the project root):
    streamlit run app.py
Note: stop the app (Ctrl+C) before running the src/ files, because only one
program at a time can open the qdrant_data/ folder.
"""

import streamlit as st

from src.config import RAW_DATA_FOLDER
from src.build_index import create_qdrant_client, index_one_file
from src.answer import answer_question


# Streamlit runs this file again on every click, but Qdrant's folder can be opened
# only once. st.cache_resource() wraps our function so it runs only the FIRST time;
# after that it hands back the same open Qdrant client.
get_shared_qdrant_client = st.cache_resource(create_qdrant_client)


def save_uploaded_file(uploaded_file):
    """Save the uploaded file into data/raw/ and return its path."""
    file_path = RAW_DATA_FOLDER / uploaded_file.name
    # getbuffer() gives the file's bytes; "wb" means "write bytes".
    with open(file_path, "wb") as output_file:
        output_file.write(uploaded_file.getbuffer())
    return file_path


def show_upload_sidebar(qdrant_client):
    """Sidebar: let HR upload a new or updated policy file and index it."""
    st.sidebar.header("Update the knowledge base")
    uploaded_file = st.sidebar.file_uploader("Upload a policy file", type=["pdf", "docx", "md"])

    # Only index when HR clicks the button, not on every rerun of the page.
    if uploaded_file is not None and st.sidebar.button("Add to knowledge base"):
        file_path = save_uploaded_file(uploaded_file)
        with st.sidebar.status("Indexing " + uploaded_file.name + " ..."):
            number_of_chunks, privacy_report = index_one_file(qdrant_client, file_path)
        st.sidebar.success("Added " + str(number_of_chunks) + " chunks from " + uploaded_file.name)

        # Show how many personal data items were hidden (never the values).
        counts = privacy_report[uploaded_file.name]
        st.sidebar.caption("Hidden: emails " + str(counts["EMAIL"]) + ", phones " + str(counts["PHONE"])
                           + ", PAN " + str(counts["PAN"]) + ", Aadhaar " + str(counts["AADHAAR"]))


def show_answer(result):
    """Show the answer and the list of sources."""
    st.subheader("Answer")
    st.write(result["answer"])

    if len(result["sources"]) > 0:
        st.subheader("Sources")
        for source in result["sources"]:
            st.markdown("- **" + source["source"] + "**, page " + str(source["page"]) + ", " + source["section"])


# ---------- The page itself ----------
st.set_page_config(page_title="HR Policy Assistant", page_icon="📘")
st.title("📘 SPIL HR Policy Assistant")
st.caption("Ask a question about the Sirca Paints HR policies. Every answer shows where it comes from.")

qdrant_client = get_shared_qdrant_client()
show_upload_sidebar(qdrant_client)

# The question box. When the employee presses Enter, the page reruns with the question filled in.
question = st.text_input("Your question", placeholder="For example: How many leaves do I get per month?")
if question.strip() != "":
    with st.spinner("Searching the HR policies..."):
        result = answer_question(qdrant_client, question)
    show_answer(result)

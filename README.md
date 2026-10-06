# HR Policy Assistant (Module 4 project)

A small web app that answers employee questions from the **SPIL (Sirca Paints) HR handbook**.
It uses RAG (Retrieval-Augmented Generation): it first finds the right parts of the documents, then asks an LLM to answer using only those parts.

What it does:
- Answers questions about the HR policies, and every answer shows its **source**: document name, page number and section.
- If the answer is not in the documents, it politely says: *"I couldn't find this in the HR policies. Please contact the HR department."*
- Hides personal data (emails, Indian mobile numbers, PAN, Aadhaar) **before** any text is sent to OpenAI.
- Lets HR upload a new or updated policy file (PDF, Word or Markdown) and updates the knowledge base straight away.
- If two documents disagree, it follows the one with the newest effective date and says so.

## Tech stack
Python, LangChain, OpenAI (`gpt-4o-mini` and `text-embedding-3-small`), Qdrant (local folder, no server), Streamlit.

## Setup (Windows)
1. Open a terminal in the project folder (`m4_hr-policy-assistant`).
2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install the libraries:
   ```
   pip install -r requirements.txt
   ```
4. Create a file called `.env` in the project folder with your OpenAI key:
   ```
   OPENAI_API_KEY=sk-...
   ```
5. Build the knowledge base (the Qdrant index):
   ```
   python -m src.build_index
   ```

## The pipeline: one file per step
Run every command from the project folder. Each file prints what it does, step by step.

| Step | File | What it does | Command |
|---|---|---|---|
| - | `src/config.py` | All settings in one place (paths, models, chunk size, top-k, threshold) | `python -m src.config` |
| 1 | `src/load_documents.py` | Loads every PDF, Word and Markdown file in `data/raw/` | `python -m src.load_documents` |
| 2 | `src/clean_text.py` | Removes the repeated header, odd characters and extra spaces; saves `data/processed/clean_pages.json` | `python -m src.clean_text` |
| 3 | `src/privacy_check.py` | Replaces emails, phones, PAN and Aadhaar with `[EMAIL]`, `[PHONE]`, ... and prints how many were hidden | `python -m src.privacy_check` |
| 4 | `src/chunk_text.py` | Splits the text into chunks with source, page, section and effective date; saves `data/processed/chunks.json` | `python -m src.chunk_text` |
| 5 | `src/build_index.py` | Runs steps 1-4, embeds the chunks and stores them in Qdrant (no duplicates) | `python -m src.build_index` |
| 6 | `src/retrieve.py` | Finds the chunks closest to a question and keeps those above the threshold | `python -m src.retrieve` |
| 7 | `src/answer.py` | Builds the prompt, calls the LLM, adds the sources | `python -m src.answer` |
| test | `tests/quick_test.py` | Asks 9 test questions and prints each answer with its sources | `python -m tests.quick_test` |

Steps 1 to 4 do not call OpenAI. Steps 5, 6 and 7 do (they need the `.env` key).

## Run the app
```
streamlit run app.py
```
Your browser opens the app. Type a question and press Enter.

To add or update a policy: in the sidebar, upload a PDF, Word or Markdown file and click **Add to knowledge base**.
The file is saved into `data/raw/` and only that file is re-indexed. Its old chunks are deleted first, so re-uploading never creates duplicates.

**Demo:** upload `data/updates/leave_policy_update_2026.docx` (2.5 leaves a month from 1 April 2026), then ask
*"How many leaves do I get per month?"* again. The answer should now follow the newer rule and say so.

## Important: only one program can open Qdrant at a time
Qdrant runs in local folder mode (`qdrant_data/`), and only one program can open that folder at a time.
**Stop the app (Ctrl+C in its terminal) before running `python -m src.build_index`, `src.retrieve`, `src.answer` or `tests.quick_test`.**
Otherwise you will see an error saying the storage folder is already being used.

## Folder structure
```text
m4_hr-policy-assistant/
├── app.py                # Streamlit app
├── data/
│   ├── raw/              # documents in the knowledge base
│   ├── updates/          # files HR uploads later (not indexed at the start)
│   └── processed/        # clean_pages.json and chunks.json, to inspect
├── qdrant_data/          # the vector database (created automatically)
├── src/                  # one file per pipeline step
└── tests/quick_test.py   # test questions
```

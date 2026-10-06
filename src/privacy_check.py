"""
privacy_check.py - Hide personal data before any text is sent to OpenAI.

RAG pipeline step 3 of 7: PRIVACY CHECK.

Why: our documents contain emails, phone numbers and ID numbers. We must
never send these to an outside service. So we replace each one with a label:
    priya@example.com  ->  [EMAIL]
    9876543210         ->  [PHONE]
    ABCDE1234F         ->  [PAN]
    2345 6789 0123     ->  [AADHAAR]

Company emails listed in ALLOWED_EMAILS (config.py) are not personal data, so we keep them.

We use "regular expressions" (regex): small patterns that describe what
an email or a phone number looks like.

The report only prints HOW MANY items were hidden, never the real values.

How to run (from the project root):
    python -m src.privacy_check
"""

import re

from langchain_core.documents import Document

from src.config import ALLOWED_EMAILS
from src.load_documents import load_all_documents
from src.clean_text import clean_documents


# Emails look like something@something.something
# They get their own function, because some company emails are allowed (see ALLOWED_EMAILS in config.py).
EMAIL_PATTERN = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"

# Each other label and the regex pattern that finds it.
PERSONAL_DATA_PATTERNS = {
    # Indian mobile: optional +91, then 10 digits starting with 6, 7, 8 or 9 (a space or dash allowed in the middle)
    "PHONE": r"(?:\+91[ -]?|\b)[6-9]\d{4}[ -]?\d{5}\b",
    # PAN card: 5 capital letters, 4 digits, 1 capital letter (like ABCDE1234F)
    "PAN": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    # Aadhaar: 12 digits starting with 2 to 9, often written in groups of 4
    "AADHAAR": r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b",
}


def hide_emails(text):
    """Replace every email with [EMAIL], except allowed company emails. Returns (text, count)."""
    number_hidden = 0
    # re.findall() gives a list of every email found in the text.
    for email in re.findall(EMAIL_PATTERN, text):
        # lower() so "HR@SircaPaints.com" also counts as allowed.
        if email.lower() in ALLOWED_EMAILS:
            continue
        text = text.replace(email, "[EMAIL]")
        number_hidden = number_hidden + 1
    return text, number_hidden


def hide_and_count(text):
    """Replace personal data with labels. Returns the new text and a count per label."""
    # Emails first, because an email can contain digits that look like a phone number.
    text, number_of_emails = hide_emails(text)
    counts = {"EMAIL": number_of_emails}
    for label, pattern in PERSONAL_DATA_PATTERNS.items():
        # re.subn() replaces every match AND tells us how many it replaced.
        text, number_hidden = re.subn(pattern, "[" + label + "]", text)
        counts[label] = number_hidden
    return text, counts


def hide_personal_data(text):
    """Replace personal data with labels and return only the safe text."""
    # We use this for the user's question too, before it goes to OpenAI.
    safe_text, counts = hide_and_count(text)
    return safe_text


def hide_personal_data_in_documents(documents):
    """Hide personal data in every Document. Returns (safe_documents, report)."""
    safe_documents = []
    # report looks like: {"file.md": {"EMAIL": 4, "PHONE": 3, "PAN": 1, "AADHAAR": 0}}
    report = {}

    for document in documents:
        safe_text, counts = hide_and_count(document.page_content)
        safe_documents.append(Document(page_content=safe_text, metadata=dict(document.metadata)))

        # Add this page's counts to the total for its file.
        file_name = document.metadata["source"]
        if file_name not in report:
            report[file_name] = {"EMAIL": 0, "PHONE": 0, "PAN": 0, "AADHAAR": 0}
        for label in counts:
            report[file_name][label] = report[file_name][label] + counts[label]

    return safe_documents, report


def print_privacy_report(report):
    """Print how many items of each kind were hidden in each file (never the values)."""
    print("Privacy report (number of items hidden):")
    for file_name in report:
        counts = report[file_name]
        print("  " + file_name + ":",
              "emails =", counts["EMAIL"],
              "| phones =", counts["PHONE"],
              "| PAN =", counts["PAN"],
              "| Aadhaar =", counts["AADHAAR"])


# ---------- Run this file on its own to see what it does ----------
if __name__ == "__main__":
    # Step 1 and 2: load and clean the pages (pipeline steps 1 and 2).
    print("Step 1: Loading and cleaning documents")
    clean_pages = clean_documents(load_all_documents())

    # Step 3: hide personal data and print the report.
    print()
    print("Step 2: Hiding personal data in", len(clean_pages), "pages")
    safe_pages, report = hide_personal_data_in_documents(clean_pages)
    print_privacy_report(report)

    # Step 4: show the helpdesk file AFTER hiding, to check that only labels are left.
    print()
    print("Step 3: The helpdesk file AFTER the privacy check:")
    print("-" * 60)
    for page in safe_pages:
        if page.metadata["source"] == "hr_helpdesk_contacts.md":
            print(page.page_content)
    print("-" * 60)

    # Step 5: a made-up example, so we can also see the Aadhaar pattern working.
    print()
    print("Step 4: Test on a made-up sentence (fake numbers):")
    print("  AFTER:", hide_personal_data("Call +91 98765 43210, Aadhaar 2345 6789 0123, PAN PQRST6789Z."))

"""Explore the FinanceBench open-source sample.

Prints basic statistics and a few full examples, so we can see
what the data looks like before building anything on top of it.

Run from the project root:
    python experiments/02_explore_financebench.py
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/financebench/data")
QUESTIONS_FILE = DATA_DIR / "financebench_open_source.jsonl"
DOCS_FILE = DATA_DIR / "financebench_document_information.jsonl"

N_EXAMPLES = 3
RANDOM_SEED = 42  # fixed seed: the same examples are picked on every run


def print_header(title):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_counts(df, column):
    """Print how many rows have each value of a column, if the column exists."""
    if column not in df.columns:
        print(f"\n(column '{column}' not found)")
        return
    print(f"\n{column}:")
    print(df[column].value_counts().to_string())


# Load both files. Each line of a .jsonl file is one record.
questions = pd.read_json(QUESTIONS_FILE, lines=True)
docs = pd.read_json(DOCS_FILE, lines=True)

# 1. How big is the dataset?
print_header("1. Size")
print(f"Questions: {len(questions)}")
print(f"Rows in document file: {len(docs)}")
print(f"Distinct documents used by the questions: {questions['doc_name'].nunique()}")

# 2. Which fields does each record have?
print_header("2. Columns")
print("Questions file:", list(questions.columns))
print("Documents file:", list(docs.columns))

# 3. How are questions and documents distributed?
print_header("3. Distributions")
print_counts(questions, "question_type")
print_counts(questions, "question_reasoning")
print_counts(docs, "company")
print_counts(docs, "doc_type")

# 4. A few complete examples.
print_header("4. Example questions")
examples = questions.sample(n=N_EXAMPLES, random_state=RANDOM_SEED)

for _, row in examples.iterrows():
    print(f"\n--- id {row['financebench_id']} | {row['doc_name']} ---")
    print("Question:", row["question"])
    print("Answer:  ", row["answer"])

    evidence = row.get("evidence")
    if isinstance(evidence, list):
        for item in evidence:
            print(f"\nEvidence (page {item.get('evidence_page_num')}):")
            print(item.get("evidence_text"))
    else:
        print("Evidence:", evidence)
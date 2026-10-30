from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

docs = [
    "Net income for fiscal year 2024 was $4.2 billion.",
    "Total revenue increased 12% compared to 2023.",
    "The company reported EBITDA of $7.1 billion in 2024.",
    "Operating expenses rose due to higher R&D spending.",
    "Profit after tax reached a record level last year.",
]
query = "How much money did the company earn in 2024?"

bm25 = BM25Okapi([d.lower().split() for d in docs])
print("BM25:", bm25.get_scores(query.lower().split()).round(2))

model = SentenceTransformer("all-MiniLM-L6-v2")
sims = util.cos_sim(model.encode(query), model.encode(docs))[0]
print("Dense:", sims.numpy().round(2))
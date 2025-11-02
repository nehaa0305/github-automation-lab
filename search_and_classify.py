from pathlib import Path
from linking_labeling.linking_model.retriever import RetrievalModel

# Initialize retriever
retriever = RetrievalModel(Path("index"))

# Search for similar PRs/issues
query = "Fix authentication bug"
results = retriever.retrieve(query, top_k=5, index_type="pr_issues")

print(f"Results for: '{query}'\n")
for i, r in enumerate(results, 1):
    print(f"{i}. {r.get('title', 'No title')}")
    print(f"   Score: {r.get('score', 0):.4f}")
    print(f"   Repo: {r.get('repo', 'Unknown')}\n")
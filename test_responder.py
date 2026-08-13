print("Script started")

from backend.agents.query_analyzer import analyze_query
from backend.agents.retriever import retrieve_context
from backend.agents.responder import generate_answer


repo_id = "f653b5ed"
repo_path = "./tmp/repos/f653b5ed"

query = "how does repository indexing work"


analysis = analyze_query(
    user_query=query,
    mode="chat"
)

print()
print("=" * 60)
print("ANALYSIS RESULT")
print("=" * 60)

print(analysis)


chunks = retrieve_context(
    repo_id=repo_id,
    repo_path=repo_path,
    retrieval_query=analysis["retrieval_query"],
)

print()
print("=" * 60)
print("TOP RETRIEVED CHUNKS")
print("=" * 60)

for i, chunk in enumerate(chunks[:5]):

    meta = chunk["metadata"]

    print()
    print(f"[{i+1}]")
    print(f"FILE   : {meta.get('file_path')}")
    print(f"SYMBOL : {meta.get('symbol_name')}")
    print(f"SCORE  : {chunk.get('score')}")

    print()
    print(chunk["text"][:300])


answer = generate_answer(
    user_query=query,
    retrieved_chunks=chunks,
)

print()
print("=" * 60)
print("FINAL ANSWER")
print("=" * 60)

print()
print(answer)

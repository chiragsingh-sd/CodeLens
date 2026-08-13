from backend.tools.search_tool import hybrid_search

repo_id = "f653b5ed"

tests = [
    ("SEMANTIC QUERY", "how does the critic safety check work"),
    ("EXACT NAME QUERY", "route_after_critic"),
]

for label, query in tests:
    print()
    print("=" * 50)
    print(label)
    print("=" * 50)

    results = hybrid_search(repo_id, query, k=3)

    print(f"Query: {query}")
    print(f"Results found: {len(results)}")

    for i, r in enumerate(results):
        print()
        print(f"[{i+1}] SCORE : {r['score']}")
        print(f"FILE     : {r['metadata']['file_path']}")
        print(f"SYMBOL   : {r['metadata']['symbol_name']}")
        print("PREVIEW:")
        print(r["text"][:250])

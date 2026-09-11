"""
Sanity test script for Corrected Hybrid Retrieval.
Validates behavior on q028 (Settings candidate participation),
as well as q008, q018, and q020.
"""

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EVAL_DIR = REPO_ROOT / "codelens-eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from harness.corrected_hybrid import CorrectedHybridRetriever

retriever = CorrectedHybridRetriever("3ef3e55d", k_dense=10, k_bm25=10, k_rrf=60)

with open(REPO_ROOT / "codelens-eval/data/eval_set.json", encoding="utf-8") as f:
    eval_set = json.load(f)

q_map = {q["id"]: q for q in eval_set}

test_qids = ["q028", "q008", "q018", "q020"]

for qid in test_qids:
    item = q_map[qid]
    query = item["query"]
    gt = item["ground_truth"]

    results, debug = retriever.retrieve(query, final_k=3)

    print("=" * 80)
    print(f"SANITY CHECK: {qid}")
    print(f"Query: {query}")
    print(f"Ground Truth: {gt['primary_symbols']}")
    print(f"\nDense Top 10 Candidates ({len(debug['dense_top_k_ids'])}):")
    for r, did in enumerate(debug["dense_top_k_ids"]):
        meta = retriever.all_metas[retriever.id_to_idx[did]]
        print(f"  [{r}] {meta.get('file_path')}::{meta.get('symbol_name')}")

    print(f"\nBM25 Top 10 Candidates ({len(debug['bm25_top_k_ids'])}):")
    for r, did in enumerate(debug["bm25_top_k_ids"]):
        meta = retriever.all_metas[retriever.id_to_idx[did]]
        print(f"  [{r}] {meta.get('file_path')}::{meta.get('symbol_name')}")

    print(f"\nUnion Candidate Count: {debug['union_candidate_count']}")

    print("\nFinal Top 3 Corrected Hybrid:")
    for r, res in enumerate(results, start=1):
        print(f"  Rank {r}: {res['file_path']}::{res['symbol_name']} | RRF Score: {res['score']}")

    # Check if target symbol is in final top 3
    gt_syms = {(ps['file'].replace('\\', '/').strip(), ps['symbol'].strip()) for ps in gt['primary_symbols']}
    hit_syms = [
        (res['file_path'], res['symbol_name'])
        for res in results
        if (res['file_path'], res['symbol_name']) in gt_syms
    ]
    print(f"\nTarget Symbol Hit in Top 3: {len(hit_syms) > 0} -> {hit_syms}")
    print(f"Latencies: {debug['latencies_ms']}")

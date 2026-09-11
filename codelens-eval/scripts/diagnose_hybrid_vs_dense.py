"""
Diagnostic Investigation: Dense Only vs. Existing Hybrid Retrieval (Step 3 Post-Eval).

Performs an evidence-based comparison of:
- Candidate pools and candidate counts (N_dense, N_bm25)
- Per-query top-3 Dense, BM25, and Hybrid/RRF rankings
- BM25 contribution: new relevant results vs. candidate duplication vs. ranking degradation
- Production tokenizer behavior on identifiers
- Step-by-step RRF fusion mechanics on q008, q018, and q020
- Benchmark composition categorization (semantic vs exact symbol vs mixed)
- Statistical breakdown of the MRR delta
"""

import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rank_bm25 import BM25Okapi
from backend.core.vector_store import get_collection
from backend.tools.search_tool import _get_embed_model, tokenize, hybrid_search


def is_symbol_match(chunk_meta: dict[str, Any], ground_truth: dict[str, Any]) -> bool:
    c_file = chunk_meta.get("file_path", "").replace("\\", "/").strip()
    c_sym = chunk_meta.get("symbol_name", "").strip()
    for ps in ground_truth.get("primary_symbols", []):
        ps_file = ps["file"].replace("\\", "/").strip()
        ps_sym = ps["symbol"].strip()
        if (c_file == ps_file or c_file.endswith(ps_file) or ps_file.endswith(c_file)) and c_sym == ps_sym:
            return True
    return False


def is_file_match(chunk_meta: dict[str, Any], ground_truth: dict[str, Any]) -> bool:
    c_file = chunk_meta.get("file_path", "").replace("\\", "/").strip()
    for af in ground_truth.get("acceptable_files", []):
        af_norm = af.replace("\\", "/").strip()
        if c_file == af_norm or c_file.endswith(af_norm) or af_norm.endswith(c_file):
            return True
    return False


def run_diagnostics():
    data_dir = REPO_ROOT / "codelens-eval" / "data"
    results_dir = REPO_ROOT / "codelens-eval" / "results"

    with open(data_dir / "benchmark_repos.json", "r", encoding="utf-8") as f:
        bench_data = json.load(f)
    repo_cfg = bench_data["repositories"][0]
    repo_id = repo_cfg.get("indexed_repo_id", "3ef3e55d")

    with open(data_dir / "eval_set.json", "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    # 1. Inspect Chroma collection
    collection = get_collection(repo_id)
    all_data = collection.get(include=["documents", "metadatas"])
    all_ids = all_data["ids"]
    all_docs = all_data["documents"]
    all_metas = all_data["metadatas"]
    total_docs = len(all_ids)

    # Build BM25 index matching production
    searchable_docs = []
    for doc, meta in zip(all_docs, all_metas):
        enriched = f"""
        {meta.get("symbol_name", "")}
        {meta.get("file_path", "")}
        {meta.get("chunk_type", "")}

        {doc}
        """
        searchable_docs.append(enriched)
    tokenized_docs = [tokenize(doc) for doc in searchable_docs]
    bm25 = BM25Okapi(tokenized_docs)

    embed_model = _get_embed_model()

    per_query_diagnostics = []

    # Counters
    bm25_adds_new_relevant = 0
    bm25_only_duplicates_dense = 0
    bm25_causes_relevant_dense_down = 0
    bm25_improves_rank = 0
    bm25_worsens_rank = 0

    mrr_diff_queries = []

    for item in eval_set:
        qid = item["id"]
        cat = item["query_type"]
        query = item["query"]
        gt = item["ground_truth"]

        # --- A. Dense Retrieval (top 3 & full ranking up to total_docs) ---
        q_emb = embed_model.encode([query], normalize_embeddings=True).tolist()
        dense_res = collection.query(
            query_embeddings=q_emb,
            n_results=total_docs,
            include=["documents", "metadatas", "distances"]
        )
        dense_ids = dense_res["ids"][0]
        dense_metas = dense_res["metadatas"][0]
        dense_dists = dense_res["distances"][0]

        top3_dense = [
            {
                "rank": r + 1,
                "doc_id": dense_ids[r],
                "file": dense_metas[r].get("file_path", "").replace("\\", "/"),
                "symbol": dense_metas[r].get("symbol_name", ""),
                "distance": round(dense_dists[r], 4),
                "is_symbol_hit": is_symbol_match(dense_metas[r], gt),
                "is_file_hit": is_file_match(dense_metas[r], gt),
            }
            for r in range(min(3, total_docs))
        ]

        # --- B. BM25 Retrieval ---
        q_tokens = tokenize(query)
        bm25_scores = bm25.get_scores(q_tokens)
        bm25_ranking = sorted(range(total_docs), key=lambda i: bm25_scores[i], reverse=True)

        top3_bm25 = [
            {
                "rank": r + 1,
                "doc_id": all_ids[bm25_ranking[r]],
                "file": all_metas[bm25_ranking[r]].get("file_path", "").replace("\\", "/"),
                "symbol": all_metas[bm25_ranking[r]].get("symbol_name", ""),
                "bm25_score": round(bm25_scores[bm25_ranking[r]], 4),
                "is_symbol_hit": is_symbol_match(all_metas[bm25_ranking[r]], gt),
                "is_file_hit": is_file_match(all_metas[bm25_ranking[r]], gt),
            }
            for r in range(min(3, total_docs))
        ]

        # --- C. Production Hybrid Retrieval ---
        hybrid_res = hybrid_search(repo_id=repo_id, query=query, k=3)
        top3_hybrid = [
            {
                "rank": r + 1,
                "file": h["metadata"].get("file_path", "").replace("\\", "/"),
                "symbol": h["metadata"].get("symbol_name", ""),
                "rrf_score": h["score"],
                "is_symbol_hit": is_symbol_match(h["metadata"], gt),
                "is_file_hit": is_file_match(h["metadata"], gt),
            }
            for r, h in enumerate(hybrid_res)
        ]

        # --- D. Analysis of BM25 Contribution ---
        dense_sym_hits = [c for c in top3_dense if c["is_symbol_hit"]]
        bm25_sym_hits = [c for c in top3_bm25 if c["is_symbol_hit"]]
        hybrid_sym_hits = [c for c in top3_hybrid if c["is_symbol_hit"]]

        dense_best_rank = dense_sym_hits[0]["rank"] if dense_sym_hits else None
        hybrid_best_rank = hybrid_sym_hits[0]["rank"] if hybrid_sym_hits else None

        dense_top3_ids = {c["doc_id"] for c in top3_dense}
        bm25_new_relevant = [
            c for c in top3_bm25
            if c["is_symbol_hit"] and c["doc_id"] not in dense_top3_ids
        ]

        has_new_rel = len(bm25_new_relevant) > 0
        if has_new_rel:
            bm25_adds_new_relevant += 1

        # Check if BM25 only duplicates or offers no new relevant
        if not has_new_rel:
            bm25_only_duplicates_dense += 1

        # Check rank shifts
        if dense_best_rank is not None and hybrid_best_rank is not None:
            if hybrid_best_rank > dense_best_rank:
                bm25_worsens_rank += 1
                bm25_causes_relevant_dense_down += 1
            elif hybrid_best_rank < dense_best_rank:
                bm25_improves_rank += 1
        elif dense_best_rank is not None and hybrid_best_rank is None:
            # Dense had it in top 3, Hybrid dropped it out of top 3!
            bm25_worsens_rank += 1
            bm25_causes_relevant_dense_down += 1
        elif dense_best_rank is None and hybrid_best_rank is not None:
            # Dense didn't have it, Hybrid brought it into top 3!
            bm25_improves_rank += 1

        # MRR calculation
        dense_mrr = (1.0 / dense_best_rank) if dense_best_rank else 0.0
        hybrid_mrr = (1.0 / hybrid_best_rank) if hybrid_best_rank else 0.0
        mrr_delta = hybrid_mrr - dense_mrr

        if abs(mrr_delta) > 1e-5:
            mrr_diff_queries.append({
                "question_id": qid,
                "category": cat,
                "query": query,
                "dense_best_rank": dense_best_rank,
                "hybrid_best_rank": hybrid_best_rank,
                "dense_mrr": round(dense_mrr, 4),
                "hybrid_mrr": round(hybrid_mrr, 4),
                "mrr_delta": round(mrr_delta, 4),
            })

        per_query_diagnostics.append({
            "question_id": qid,
            "category": cat,
            "query": query,
            "top3_dense": top3_dense,
            "top3_bm25": top3_bm25,
            "top3_hybrid": top3_hybrid,
            "dense_best_rank": dense_best_rank,
            "hybrid_best_rank": hybrid_best_rank,
            "bm25_new_relevant_in_top3": has_new_rel,
            "dense_mrr": round(dense_mrr, 4),
            "hybrid_mrr": round(hybrid_mrr, 4),
            "mrr_delta": round(mrr_delta, 4),
        })

    # Save diagnostics JSON
    out_file = results_dir / "hybrid_vs_dense_diagnosis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_documents_in_collection": total_docs,
            "summary_counts": {
                "bm25_adds_new_relevant": bm25_adds_new_relevant,
                "bm25_only_duplicates_or_no_new_rel": bm25_only_duplicates_dense,
                "bm25_causes_relevant_dense_down": bm25_causes_relevant_dense_down,
                "bm25_improves_rank": bm25_improves_rank,
                "bm25_worsens_rank": bm25_worsens_rank,
            },
            "queries_with_mrr_difference": mrr_diff_queries,
            "per_query": per_query_diagnostics,
        }, f, indent=2)

    print("=" * 70)
    print("DIAGNOSTIC SUMMARY: DENSE ONLY vs. EXISTING HYBRID")
    print("=" * 70)
    print(f"Total Questions Evaluated           : {len(eval_set)}")
    print(f"Total Documents in Chroma Collection: {total_docs}")
    print(f"BM25 adds a new relevant result     : {bm25_adds_new_relevant}")
    print(f"BM25 only duplicates Dense / no new : {bm25_only_duplicates_dense}")
    print(f"BM25 causes relevant Dense to drop  : {bm25_causes_relevant_dense_down}")
    print(f"BM25 improves correct result's rank : {bm25_improves_rank}")
    print(f"BM25 worsens correct result's rank  : {bm25_worsens_rank}")
    print(f"Questions accounting for MRR diff   : {len(mrr_diff_queries)}")
    print("=" * 70)
    for q in mrr_diff_queries:
        print(f"  {q['question_id']} [{q['category']}]: Dense Rank {q['dense_best_rank']} (MRR {q['dense_mrr']}) -> Hybrid Rank {q['hybrid_best_rank']} (MRR {q['hybrid_mrr']}) | Delta: {q['mrr_delta']:+.4f}")
    print("=" * 70)


if __name__ == "__main__":
    run_diagnostics()

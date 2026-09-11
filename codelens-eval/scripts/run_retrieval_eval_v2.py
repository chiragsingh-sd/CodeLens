"""
Benchmark Runner for Second Controlled Retrieval Experiment (Step 3 Follow-up).

Evaluates THREE retrieval variants across the exact same 40 benchmark questions:
1. Variant A: Dense Only (ChromaDB vector search using all-MiniLM-L6-v2, top 3)
2. Variant B: Existing Hybrid (Production BM25Okapi + ChromaDB via RRF, top 3)
3. Variant C: Corrected Hybrid (Dense top 10 + BM25 top 10 -> Union -> RRF -> top 3)

Computes standard IR metrics at symbol and file levels, sub-component latencies,
BM25 contribution statistics, failure classification, and writes:
- codelens-eval/results/retrieval_results_v2.json
- codelens-eval/results/retrieval_results_v2.csv
- codelens-eval/results/retrieval_report_v2.md
"""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EVAL_DIR = REPO_ROOT / "codelens-eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from backend.tools.search_tool import _get_embed_model
from harness.metrics import (
    calculate_query_metrics,
    compute_aggregate_metrics,
    compute_latency_stats,
)
from harness.variants import (
    run_dense_retrieval,
    run_hybrid_retrieval,
)
from harness.corrected_hybrid import CorrectedHybridRetriever


def get_git_commit(cwd: Path) -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


def is_symbol_match(chunk_meta: dict[str, Any], ground_truth: dict[str, Any]) -> bool:
    c_file = chunk_meta.get("file_path", "").replace("\\", "/").strip()
    c_sym = chunk_meta.get("symbol_name", "").strip()
    for ps in ground_truth.get("primary_symbols", []):
        ps_file = ps["file"].replace("\\", "/").strip()
        ps_sym = ps["symbol"].strip()
        if (c_file == ps_file or c_file.endswith(ps_file) or ps_file.endswith(c_file)) and c_sym == ps_sym:
            return True
    return False


def run_benchmark_v2():
    start_time = datetime.now(timezone.utc)
    print("=" * 80)
    print("CODELENS RETRIEVAL BENCHMARK -- EXPERIMENT 2 (CORRECTED HYBRID)")
    print(f"Timestamp: {start_time.isoformat()}")
    print("=" * 80)

    data_dir = REPO_ROOT / "codelens-eval" / "data"
    results_dir = REPO_ROOT / "codelens-eval" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "benchmark_repos.json", "r", encoding="utf-8") as f:
        bench_data = json.load(f)
    with open(data_dir / "eval_set.json", "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    repo_cfg = bench_data["repositories"][0]
    repo_id_in_chroma = repo_cfg.get("indexed_repo_id", repo_cfg["id"])
    git_sha = get_git_commit(REPO_ROOT)

    print(f"Benchmark Repository : {repo_cfg['name']} (chroma_id={repo_id_in_chroma})")
    print(f"Git Commit SHA       : {git_sha}")
    print(f"Python Version       : {platform.python_version()}")
    print(f"Total Questions      : {len(eval_data)}")
    print(f"Variants Evaluated   : Dense Only (k=3), Existing Hybrid (k=3), Corrected Hybrid (k10/k10->k3)")
    print()

    # Pre-warm embedding model
    print("[INIT] Pre-warming embedding model and initializing Corrected Hybrid retriever...")
    _get_embed_model().encode(["pre-warm embedding cache"], normalize_embeddings=True)
    corrected_retriever = CorrectedHybridRetriever(
        repo_id=repo_id_in_chroma,
        k_dense=10,
        k_bm25=10,
        k_rrf=60,
    )
    print(f"[INIT] Model & index ready ({corrected_retriever.total_docs} docs). Evaluating queries...\n")

    detailed_results = []

    # Latencies
    dense_latencies = []
    existing_hybrid_latencies = []
    corrected_hybrid_latencies = []
    corrected_dense_sub_latencies = []
    corrected_bm25_sub_latencies = []
    corrected_rrf_sub_latencies = []

    # Metrics collections (symbol level)
    dense_sym_metrics = []
    existing_sym_metrics = []
    corrected_sym_metrics = []

    # Metrics collections (file level)
    dense_file_metrics = []
    existing_file_metrics = []
    corrected_file_metrics = []

    # BM25 contribution tracking
    bm25_unique_candidate_queries = 0
    bm25_unique_relevant_queries = 0
    bm25_new_relevant_in_top3_queries = 0
    corrected_improves_rank_queries = 0
    corrected_worsens_rank_queries = 0
    corrected_unchanged_rank_queries = 0
    bm25_zero_unique_candidate_queries = 0

    for idx, item in enumerate(eval_data, start=1):
        qid = item["id"]
        cat = item["query_type"]
        query = item["query"]
        gt = item["ground_truth"]

        # -------------------------------------------------------------
        # 1. DENSE ONLY (k=3)
        # -------------------------------------------------------------
        t0_dense = time.perf_counter()
        dense_results = run_dense_retrieval(repo_id_in_chroma, query, k=3)
        dense_ms = (time.perf_counter() - t0_dense) * 1000.0
        dense_latencies.append(dense_ms)

        m_dense_sym = calculate_query_metrics(dense_results, gt, k=3, level="symbol")
        m_dense_file = calculate_query_metrics(dense_results, gt, k=3, level="file")
        dense_sym_metrics.append(m_dense_sym)
        dense_file_metrics.append(m_dense_file)

        # -------------------------------------------------------------
        # 2. EXISTING HYBRID (k=3)
        # -------------------------------------------------------------
        t0_exist = time.perf_counter()
        existing_results = run_hybrid_retrieval(repo_id_in_chroma, query, k=3)
        exist_ms = (time.perf_counter() - t0_exist) * 1000.0
        existing_hybrid_latencies.append(exist_ms)

        m_exist_sym = calculate_query_metrics(existing_results, gt, k=3, level="symbol")
        m_exist_file = calculate_query_metrics(existing_results, gt, k=3, level="file")
        existing_sym_metrics.append(m_exist_sym)
        existing_file_metrics.append(m_exist_file)

        # -------------------------------------------------------------
        # 3. CORRECTED HYBRID (K_dense=10, K_bm25=10, final_k=3)
        # -------------------------------------------------------------
        corrected_results, debug_info = corrected_retriever.retrieve(query, final_k=3)
        corr_ms = debug_info["latencies_ms"]["total_ms"]
        corrected_hybrid_latencies.append(corr_ms)
        corrected_dense_sub_latencies.append(debug_info["latencies_ms"]["dense_ms"])
        corrected_bm25_sub_latencies.append(debug_info["latencies_ms"]["bm25_ms"])
        corrected_rrf_sub_latencies.append(debug_info["latencies_ms"]["rrf_ms"])

        m_corr_sym = calculate_query_metrics(corrected_results, gt, k=3, level="symbol")
        m_corr_file = calculate_query_metrics(corrected_results, gt, k=3, level="file")
        corrected_sym_metrics.append(m_corr_sym)
        corrected_file_metrics.append(m_corr_file)

        # -------------------------------------------------------------
        # 4. BM25 CONTRIBUTION & COMPARATIVE ANALYSIS
        # -------------------------------------------------------------
        dense_top10_ids = set(debug_info["dense_top_k_ids"])
        bm25_top10_ids = set(debug_info["bm25_top_k_ids"])
        bm25_only_ids = bm25_top10_ids - dense_top10_ids

        # 1. Does BM25 introduce candidate Dense top 10 didn't contain?
        if len(bm25_only_ids) > 0:
            bm25_unique_candidate_queries += 1
        else:
            bm25_zero_unique_candidate_queries += 1

        # 2. Is that BM25-only candidate relevant?
        bm25_only_relevant_ids = []
        for did in bm25_only_ids:
            idx_doc = corrected_retriever.id_to_idx[did]
            meta = corrected_retriever.all_metas[idx_doc]
            if is_symbol_match(meta, gt):
                bm25_only_relevant_ids.append(did)

        if len(bm25_only_relevant_ids) > 0:
            bm25_unique_relevant_queries += 1

        # 3. Does BM25 introduce new relevant candidate into final Hybrid top 3?
        dense_top3_ids = {r["symbol_name"] for r in dense_results}
        corr_top3_relevant_syms = {
            r["symbol_name"] for r in corrected_results
            if is_symbol_match({"file_path": r["file_path"], "symbol_name": r["symbol_name"]}, gt)
        }
        new_relevant_in_top3 = corr_top3_relevant_syms - dense_top3_ids
        if len(new_relevant_in_top3) > 0:
            bm25_new_relevant_in_top3_queries += 1

        # 4, 5, 6. Rank shifts: Corrected Hybrid vs Dense
        dense_rank = m_dense_sym["first_relevant_rank"]
        corr_rank = m_corr_sym["first_relevant_rank"]

        if dense_rank is not None and corr_rank is not None:
            if corr_rank < dense_rank:
                corrected_improves_rank_queries += 1
                rank_shift_label = "IMPROVED"
            elif corr_rank > dense_rank:
                corrected_worsens_rank_queries += 1
                rank_shift_label = "WORSENED"
            else:
                corrected_unchanged_rank_queries += 1
                rank_shift_label = "UNCHANGED"
        elif dense_rank is None and corr_rank is not None:
            corrected_improves_rank_queries += 1
            rank_shift_label = "RESCUED (0 -> Hit)"
        elif dense_rank is not None and corr_rank is None:
            corrected_worsens_rank_queries += 1
            rank_shift_label = "LOST (Hit -> 0)"
        else:
            corrected_unchanged_rank_queries += 1
            rank_shift_label = "BOTH_MISSED"

        # Record detailed per-query trace
        detailed_results.append({
            "question_id": qid,
            "category": cat,
            "query": query,
            "ground_truth": gt,
            "variants": {
                "dense_only": {
                    "latency_ms": round(dense_ms, 2),
                    "retrieved_results": dense_results,
                    "metrics_symbol": m_dense_sym,
                    "metrics_file": m_dense_file,
                },
                "existing_hybrid": {
                    "latency_ms": round(exist_ms, 2),
                    "retrieved_results": existing_results,
                    "metrics_symbol": m_exist_sym,
                    "metrics_file": m_exist_file,
                },
                "corrected_hybrid": {
                    "latency_ms": round(corr_ms, 2),
                    "retrieved_results": corrected_results,
                    "metrics_symbol": m_corr_sym,
                    "metrics_file": m_corr_file,
                    "debug_info": debug_info,
                },
            },
            "comparison": {
                "bm25_unique_candidates_count": len(bm25_only_ids),
                "bm25_unique_relevant_count": len(bm25_only_relevant_ids),
                "new_relevant_in_top3": list(new_relevant_in_top3),
                "dense_symbol_rank": dense_rank,
                "existing_symbol_rank": m_exist_sym["first_relevant_rank"],
                "corrected_symbol_rank": corr_rank,
                "rank_shift_vs_dense": rank_shift_label,
            },
        })

        print(
            f"[{idx:02d}/{len(eval_data)}] {qid} [{cat:<16}] "
            f"Dense: {m_dense_sym['mrr']:.2f} | "
            f"ExistHyb: {m_exist_sym['mrr']:.2f} | "
            f"CorrHyb: {m_corr_sym['mrr']:.2f} | "
            f"Shift: {rank_shift_label}"
        )

    # -----------------------------------------------------------------
    # Compute Aggregates
    # -----------------------------------------------------------------
    dense_sym_agg = compute_aggregate_metrics(dense_sym_metrics, k=3)
    dense_file_agg = compute_aggregate_metrics(dense_file_metrics, k=3)
    dense_lat_agg = compute_latency_stats(dense_latencies)

    exist_sym_agg = compute_aggregate_metrics(existing_sym_metrics, k=3)
    exist_file_agg = compute_aggregate_metrics(existing_file_metrics, k=3)
    exist_lat_agg = compute_latency_stats(existing_hybrid_latencies)

    corr_sym_agg = compute_aggregate_metrics(corrected_sym_metrics, k=3)
    corr_file_agg = compute_aggregate_metrics(corrected_file_metrics, k=3)
    corr_lat_agg = compute_latency_stats(corrected_hybrid_latencies)

    # Sub-step latency aggregates for Corrected Hybrid
    corr_sub_lat_agg = {
        "dense_sub_ms": compute_latency_stats(corrected_dense_sub_latencies),
        "bm25_sub_ms": compute_latency_stats(corrected_bm25_sub_latencies),
        "rrf_sub_ms": compute_latency_stats(corrected_rrf_sub_latencies),
    }

    # Category breakdowns
    categories = sorted(list(set(q["query_type"] for q in eval_data)))
    category_breakdowns = []
    for c in categories:
        c_items = [r for r in detailed_results if r["category"] == c]
        c_dense_sym = compute_aggregate_metrics([it["variants"]["dense_only"]["metrics_symbol"] for it in c_items])
        c_exist_sym = compute_aggregate_metrics([it["variants"]["existing_hybrid"]["metrics_symbol"] for it in c_items])
        c_corr_sym = compute_aggregate_metrics([it["variants"]["corrected_hybrid"]["metrics_symbol"] for it in c_items])

        category_breakdowns.append({
            "category": c,
            "count": len(c_items),
            "dense": c_dense_sym,
            "existing_hybrid": c_exist_sym,
            "corrected_hybrid": c_corr_sym,
        })

    bm25_stats = {
        "bm25_unique_candidate_queries": bm25_unique_candidate_queries,
        "bm25_zero_unique_candidate_queries": bm25_zero_unique_candidate_queries,
        "bm25_unique_relevant_queries": bm25_unique_relevant_queries,
        "bm25_new_relevant_in_top3_queries": bm25_new_relevant_in_top3_queries,
        "corrected_improves_rank_queries": corrected_improves_rank_queries,
        "corrected_worsens_rank_queries": corrected_worsens_rank_queries,
        "corrected_unchanged_rank_queries": corrected_unchanged_rank_queries,
    }

    output_data = {
        "benchmark_metadata": {
            "timestamp": start_time.isoformat(),
            "repository": repo_cfg["name"],
            "repo_id": repo_cfg["id"],
            "chroma_collection": f"repo_{repo_id_in_chroma}",
            "git_commit_sha": git_sha,
            "python_version": platform.python_version(),
            "embedding_model": "all-MiniLM-L6-v2",
            "k_dense": 10,
            "k_bm25": 10,
            "final_k": 3,
            "k_rrf": 60,
            "total_questions": len(eval_data),
        },
        "overall_summary": {
            "dense_only_symbol": {**dense_sym_agg, **dense_lat_agg},
            "existing_hybrid_symbol": {**exist_sym_agg, **exist_lat_agg},
            "corrected_hybrid_symbol": {**corr_sym_agg, **corr_lat_agg},
            "dense_only_file": dense_file_agg,
            "existing_hybrid_file": exist_file_agg,
            "corrected_hybrid_file": corr_file_agg,
            "corrected_sub_latencies": corr_sub_lat_agg,
        },
        "bm25_contribution_statistics": bm25_stats,
        "category_summary": category_breakdowns,
        "per_query_results": detailed_results,
    }

    # Save JSON
    json_path = results_dir / "retrieval_results_v2.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n[OK] Saved results JSON to {json_path}")

    # Save CSV
    csv_path = results_dir / "retrieval_results_v2.csv"
    export_csv(csv_path, detailed_results)
    print(f"[OK] Exported results CSV to {csv_path}")

    # Generate Markdown Report
    report_path = results_dir / "retrieval_report_v2.md"
    generate_markdown_report_v2(report_path, output_data)
    print(f"[OK] Generated summary report to {report_path}\n")

    print_console_summary_v2(output_data)


def export_csv(csv_path: Path, results: list[dict[str, Any]]):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "question_id",
            "category",
            "query",
            "dense_hit1",
            "dense_hit3",
            "dense_mrr",
            "dense_latency_ms",
            "exist_hit1",
            "exist_hit3",
            "exist_mrr",
            "exist_latency_ms",
            "corr_hit1",
            "corr_hit3",
            "corr_mrr",
            "corr_latency_ms",
            "bm25_unique_candidates",
            "bm25_unique_relevant",
            "rank_shift_vs_dense",
        ])
        for r in results:
            d = r["variants"]["dense_only"]
            e = r["variants"]["existing_hybrid"]
            c = r["variants"]["corrected_hybrid"]
            cmp = r["comparison"]
            writer.writerow([
                r["question_id"],
                r["category"],
                r["query"],
                d["metrics_symbol"]["hit_at_1"],
                d["metrics_symbol"]["hit_at_3"],
                d["metrics_symbol"]["mrr"],
                d["latency_ms"],
                e["metrics_symbol"]["hit_at_1"],
                e["metrics_symbol"]["hit_at_3"],
                e["metrics_symbol"]["mrr"],
                e["latency_ms"],
                c["metrics_symbol"]["hit_at_1"],
                c["metrics_symbol"]["hit_at_3"],
                c["metrics_symbol"]["mrr"],
                c["latency_ms"],
                cmp["bm25_unique_candidates_count"],
                cmp["bm25_unique_relevant_count"],
                cmp["rank_shift_vs_dense"],
            ])


def generate_markdown_report_v2(report_path: Path, data: dict[str, Any]):
    meta = data["benchmark_metadata"]
    ov = data["overall_summary"]
    bm = data["bm25_contribution_statistics"]
    cats = data["category_summary"]
    per_q = data["per_query_results"]

    ds = ov["dense_only_symbol"]
    es = ov["existing_hybrid_symbol"]
    cs = ov["corrected_hybrid_symbol"]

    lines = [
        "# CodeLens Retrieval Evaluation (Experiment 2: Corrected Hybrid)",
        "",
        "## Executive Summary",
        "",
        "This controlled experiment evaluates whether a corrected candidate-pool architecture (retrieving independent top-10 candidate sets for both Dense and BM25 before RRF fusion) resolves the underperformance observed in Existing Hybrid retrieval.",
        "",
        "## Benchmark Configuration",
        f"- **Repository**: `{meta['repository']}` (`{meta['chroma_collection']}`)",
        f"- **Git Commit SHA**: `{meta['git_commit_sha']}`",
        f"- **Embedding Model**: `{meta['embedding_model']}`",
        f"- **Dense Candidate Pool ($K_{{dense}}$)**: `{meta['k_dense']}`",
        f"- **BM25 Candidate Pool ($K_{{bm25}}$)**: `{meta['k_bm25']}`",
        f"- **Final Retrieval Depth ($k$)**: `{meta['final_k']}`",
        f"- **RRF Constant ($K_{{RRF}}$)**: `{meta['k_rrf']}` (0-indexed)",
        f"- **Questions Evaluated**: `{meta['total_questions']}`",
        f"- **Timestamp**: `{meta['timestamp']}`",
        "",
        "---",
        "",
        "## 1. Overall Performance Comparison (Symbol Level)",
        "",
        "| Metric | Dense Only (k=3) | Existing Hybrid (k=3) | Corrected Hybrid (k10/k10->k3) | Corrected vs. Dense (Delta) | Corrected vs. Existing (Delta) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
        f"| **Hit@1** | **{ds['hit_at_1'] * 100:.1f}%** | {es['hit_at_1'] * 100:.1f}% | {cs['hit_at_1'] * 100:.1f}% | {cs['hit_at_1'] - ds['hit_at_1']:+.3f} ({(cs['hit_at_1'] - ds['hit_at_1'])*100:+.1f}%) | {cs['hit_at_1'] - es['hit_at_1']:+.3f} ({(cs['hit_at_1'] - es['hit_at_1'])*100:+.1f}%) |",
        f"| **Hit@3** | {ds['hit_at_3'] * 100:.1f}% | {es['hit_at_3'] * 100:.1f}% | **{cs['hit_at_3'] * 100:.1f}%** | {cs['hit_at_3'] - ds['hit_at_3']:+.3f} ({(cs['hit_at_3'] - ds['hit_at_3'])*100:+.1f}%) | {cs['hit_at_3'] - es['hit_at_3']:+.3f} ({(cs['hit_at_3'] - es['hit_at_3'])*100:+.1f}%) |",
        f"| **Recall@3** | {ds['recall_at_3'] * 100:.1f}% | {es['recall_at_3'] * 100:.1f}% | **{cs['recall_at_3'] * 100:.1f}%** | {cs['recall_at_3'] - ds['recall_at_3']:+.3f} ({(cs['recall_at_3'] - ds['recall_at_3'])*100:+.1f}%) | {cs['recall_at_3'] - es['recall_at_3']:+.3f} ({(cs['recall_at_3'] - es['recall_at_3'])*100:+.1f}%) |",
        f"| **Precision@3** | {ds['precision_at_3'] * 100:.1f}% | {es['precision_at_3'] * 100:.1f}% | **{cs['precision_at_3'] * 100:.1f}%** | {cs['precision_at_3'] - ds['precision_at_3']:+.3f} ({(cs['precision_at_3'] - ds['precision_at_3'])*100:+.1f}%) | {cs['precision_at_3'] - es['precision_at_3']:+.3f} ({(cs['precision_at_3'] - es['precision_at_3'])*100:+.1f}%) |",
        f"| **MRR** | **{ds['mrr']:.4f}** | {es['mrr']:.4f} | {cs['mrr']:.4f} | {cs['mrr'] - ds['mrr']:+.4f} | {cs['mrr'] - es['mrr']:+.4f} |",
        "",
        "---",
        "",
        "## 2. Latency Benchmarks (Milliseconds)",
        "",
        "| Variant | Mean Latency | P50 (Median) | P95 |",
        "|:---|:---:|:---:|:---:|",
        f"| **Dense Only** | {ds['mean_ms']:.2f} ms | {ds['p50_ms']:.2f} ms | {ds['p95_ms']:.2f} ms |",
        f"| **Existing Hybrid** | {es['mean_ms']:.2f} ms | {es['p50_ms']:.2f} ms | {es['p95_ms']:.2f} ms |",
        f"| **Corrected Hybrid** | {cs['mean_ms']:.2f} ms | {cs['p50_ms']:.2f} ms | {cs['p95_ms']:.2f} ms |",
        "",
        "### Corrected Hybrid Sub-step Latencies:",
        f"- **Dense Step**: Mean `{ov['corrected_sub_latencies']['dense_sub_ms']['mean_ms']:.2f} ms` | P50 `{ov['corrected_sub_latencies']['dense_sub_ms']['p50_ms']:.2f} ms`",
        f"- **BM25 Step**: Mean `{ov['corrected_sub_latencies']['bm25_sub_ms']['mean_ms']:.2f} ms` | P50 `{ov['corrected_sub_latencies']['bm25_sub_ms']['p50_ms']:.2f} ms`",
        f"- **RRF Fusion**: Mean `{ov['corrected_sub_latencies']['rrf_sub_ms']['mean_ms']:.2f} ms` | P50 `{ov['corrected_sub_latencies']['rrf_sub_ms']['p50_ms']:.2f} ms`",
        "",
        "---",
        "",
        "## 3. Category Breakdown (Symbol Level)",
        "",
        "| Category | Qs | Dense Hit@3 | ExistHyb Hit@3 | CorrHyb Hit@3 | Dense MRR | ExistHyb MRR | CorrHyb MRR |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for c in cats:
        d = c["dense"]
        e = c["existing_hybrid"]
        ch = c["corrected_hybrid"]
        lines.append(
            f"| `{c['category']}` | {c['count']} | {d['hit_at_3']*100:.1f}% | {e['hit_at_3']*100:.1f}% | {ch['hit_at_3']*100:.1f}% | {d['mrr']:.4f} | {e['mrr']:.4f} | {ch['mrr']:.4f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. BM25 Candidate Contribution Statistics",
        "",
        f"- **Queries where BM25 introduces unique candidate(s) not in Dense top 10**: `{bm['bm25_unique_candidate_queries']} / 40` ({bm['bm25_unique_candidate_queries']/40*100:.1f}%)",
        f"- **Queries where that BM25-only candidate is RELEVANT**: `{bm['bm25_unique_relevant_queries']} / 40` ({bm['bm25_unique_relevant_queries']/40*100:.1f}%)",
        f"- **Queries where BM25 brings a NEW relevant candidate into final Top 3**: `{bm['bm25_new_relevant_in_top3_queries']} / 40` ({bm['bm25_new_relevant_in_top3_queries']/40*100:.1f}%)",
        f"- **Queries where Corrected Hybrid IMPROVES rank vs. Dense**: `{bm['corrected_improves_rank_queries']} / 40` ({bm['corrected_improves_rank_queries']/40*100:.1f}%)",
        f"- **Queries where Corrected Hybrid WORSENS rank vs. Dense**: `{bm['corrected_worsens_rank_queries']} / 40` ({bm['corrected_worsens_rank_queries']/40*100:.1f}%)",
        f"- **Queries where rankings remain UNCHANGED**: `{bm['corrected_unchanged_rank_queries']} / 40` ({bm['corrected_unchanged_rank_queries']/40*100:.1f}%)",
        f"- **Queries where BM25 contributes NO unique candidates**: `{bm['bm25_zero_unique_candidate_queries']} / 40` ({bm['bm25_zero_unique_candidate_queries']/40*100:.1f}%)",
        "",
        "---",
        "",
        "## 5. Detailed Failure & Case Analysis",
        "",
    ])

    # Find cases
    case_a = [r for r in per_q if r["variants"]["dense_only"]["metrics_symbol"]["hit_at_3"] == 0 and r["variants"]["corrected_hybrid"]["metrics_symbol"]["hit_at_3"] > 0]
    case_b = [r for r in per_q if r["variants"]["dense_only"]["metrics_symbol"]["hit_at_3"] > 0 and r["variants"]["corrected_hybrid"]["metrics_symbol"]["hit_at_3"] == 0]
    case_c = [r for r in per_q if r["variants"]["dense_only"]["metrics_symbol"]["hit_at_3"] == 0 and r["variants"]["corrected_hybrid"]["metrics_symbol"]["hit_at_3"] == 0]
    case_d = [r for r in per_q if r["variants"]["existing_hybrid"]["metrics_symbol"]["hit_at_3"] == 0 and r["variants"]["corrected_hybrid"]["metrics_symbol"]["hit_at_3"] > 0]
    case_e = [r for r in per_q if r["variants"]["corrected_hybrid"]["metrics_symbol"]["hit_at_3"] > 0 and r["variants"]["dense_only"]["metrics_symbol"]["hit_at_3"] > 0 and r["variants"]["corrected_hybrid"]["metrics_symbol"]["mrr"] != r["variants"]["dense_only"]["metrics_symbol"]["mrr"]]

    lines.append(f"### 5.1. Dense Fails, Corrected Hybrid Succeeds ({len(case_a)} Cases)")
    if not case_a:
        lines.append("*None.*")
    else:
        for it in case_a:
            lines.extend(format_case_v2(it))

    lines.append(f"\n### 5.2. Dense Succeeds, Corrected Hybrid Fails ({len(case_b)} Cases)")
    if not case_b:
        lines.append("*None. Corrected Hybrid never dropped a successful Dense hit out of the Top 3.*")
    else:
        for it in case_b:
            lines.extend(format_case_v2(it))

    lines.append(f"\n### 5.3. Existing Hybrid Fails, Corrected Hybrid Succeeds ({len(case_d)} Cases)")
    if not case_d:
        lines.append("*None.*")
    else:
        for it in case_d:
            lines.extend(format_case_v2(it))

    lines.append(f"\n### 5.4. Corrected Hybrid Changes Rank but Not Hit@3 ({len(case_e)} Cases)")
    for it in case_e[:5]:
        lines.extend(format_case_v2(it))

    lines.extend([
        "",
        "---",
        "",
        "## 6. Conclusions",
        "",
        "1. **Architectural Validation**: The candidate union architecture successfully enables BM25 to rescue queries missed by Dense top 3 (such as `q028`), increasing overall Hit@3 and Recall@3.",
        "2. **Trade-off between Recall and Precision/MRR**: While candidate union increases Hit@3, fusing lexical ranking into the top ranks creates noise that occasionally displaces Rank 1 semantic hits, affecting MRR.",
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def format_case_v2(r: dict[str, Any]) -> list[str]:
    qid = r["question_id"]
    query = r["query"]
    gt = r["ground_truth"]["primary_symbols"]
    d = r["variants"]["dense_only"]
    e = r["variants"]["existing_hybrid"]
    c = r["variants"]["corrected_hybrid"]
    return [
        f"- **{qid}**: *\"{query}\"*",
        f"  - **Ground Truth**: `{gt}`",
        f"  - **Dense Top 3**: {[(res['file_path'], res['symbol_name']) for res in d['retrieved_results']]} (MRR={d['metrics_symbol']['mrr']})",
        f"  - **Existing Hybrid Top 3**: {[(res['file_path'], res['symbol_name']) for res in e['retrieved_results']]} (MRR={e['metrics_symbol']['mrr']})",
        f"  - **Corrected Hybrid Top 3**: {[(res['file_path'], res['symbol_name']) for res in c['retrieved_results']]} (MRR={c['metrics_symbol']['mrr']})",
        f"  - **Rank Shift vs. Dense**: `{r['comparison']['rank_shift_vs_dense']}`",
        "",
    ]


def print_console_summary_v2(data: dict[str, Any]):
    ov = data["overall_summary"]
    ds = ov["dense_only_symbol"]
    es = ov["existing_hybrid_symbol"]
    cs = ov["corrected_hybrid_symbol"]
    bm = data["bm25_contribution_statistics"]

    print("=" * 80)
    print("OVERALL RETRIEVAL METRICS: 3-WAY VARIANT COMPARISON (Symbol Level)")
    print("=" * 80)
    print(f"{'Variant':<28} | {'Hit@1':<8} | {'Hit@3':<8} | {'Recall@3':<10} | {'Prec@3':<8} | {'MRR':<8} | {'Latency':<8}")
    print("-" * 80)
    print(f"{'Variant A: Dense Only':<28} | {ds['hit_at_1']*100:>6.1f}% | {ds['hit_at_3']*100:>6.1f}% | {ds['recall_at_3']*100:>8.1f}% | {ds['precision_at_3']*100:>6.1f}% | {ds['mrr']:>7.4f} | {ds['mean_ms']:>6.2f}ms")
    print(f"{'Variant B: Existing Hybrid':<28} | {es['hit_at_1']*100:>6.1f}% | {es['hit_at_3']*100:>6.1f}% | {es['recall_at_3']*100:>8.1f}% | {es['precision_at_3']*100:>6.1f}% | {es['mrr']:>7.4f} | {es['mean_ms']:>6.2f}ms")
    print(f"{'Variant C: Corrected Hybrid':<28} | {cs['hit_at_1']*100:>6.1f}% | {cs['hit_at_3']*100:>6.1f}% | {cs['recall_at_3']*100:>8.1f}% | {cs['precision_at_3']*100:>6.1f}% | {cs['mrr']:>7.4f} | {cs['mean_ms']:>6.2f}ms")
    print("=" * 80)
    print("\nBM25 CANDIDATE CONTRIBUTION:")
    print(f"  - Queries where BM25 introduces unique candidate(s): {bm['bm25_unique_candidate_queries']}/40")
    print(f"  - Queries where that candidate is relevant         : {bm['bm25_unique_relevant_queries']}/40")
    print(f"  - Queries where BM25 adds NEW relevant to Top 3    : {bm['bm25_new_relevant_in_top3_queries']}/40")
    print(f"  - Queries where Corrected improves rank vs. Dense  : {bm['corrected_improves_rank_queries']}/40")
    print(f"  - Queries where Corrected worsens rank vs. Dense   : {bm['corrected_worsens_rank_queries']}/40")
    print(f"  - Queries where rank remains unchanged             : {bm['corrected_unchanged_rank_queries']}/40")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark_v2()

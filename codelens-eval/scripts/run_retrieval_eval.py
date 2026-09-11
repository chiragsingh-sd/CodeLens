"""
Benchmark Runner for CodeLens Retrieval Evaluation (Step 2).

Executes Variant A (Dense Only) vs Variant B (Hybrid Search) across all
40 benchmark questions, computes standard IR metrics (Hit@1, Hit@3, Recall@3,
Precision@3, MRR), records high-resolution latencies, identifies failure modes,
and outputs structured JSON, CSV, and a formatted Markdown report.
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

# Ensure repository root and codelens-eval directory are on sys.path
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


def load_dataset(data_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bench_file = data_dir / "benchmark_repos.json"
    eval_file = data_dir / "eval_set.json"

    with open(bench_file, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    return bench_data, eval_data


def run_benchmark():
    start_time = datetime.now(timezone.utc)
    print("=" * 80)
    print("CODELENS RETRIEVAL BENCHMARK -- STEP 2 EVALUATION")
    print(f"Timestamp: {start_time.isoformat()}")
    print("=" * 80)

    data_dir = REPO_ROOT / "codelens-eval" / "data"
    results_dir = REPO_ROOT / "codelens-eval" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    bench_data, eval_data = load_dataset(data_dir)
    repo_cfg = bench_data["repositories"][0]
    repo_id_in_chroma = repo_cfg.get("indexed_repo_id", repo_cfg["id"])
    git_sha = get_git_commit(REPO_ROOT)

    print(f"Benchmark Repository : {repo_cfg['name']} (id={repo_cfg['id']}, chroma_id={repo_id_in_chroma})")
    print(f"Git Commit SHA       : {git_sha}")
    print(f"Python Version       : {platform.python_version()}")
    print(f"Total Questions      : {len(eval_data)}")
    print(f"Retrieval Depth (K)  : 3")
    print()

    # Pre-warm embedding model to exclude one-time model initialization from query latencies
    print("[INIT] Pre-warming embedding model...")
    _get_embed_model().encode(["pre-warm embedding cache"], normalize_embeddings=True)
    print("[INIT] Model pre-warmed. Running retrieval queries...\n")

    detailed_results = []
    dense_latencies = []
    hybrid_latencies = []

    # Per-variant lists for metric calculation
    dense_metrics_list = []
    hybrid_metrics_list = []
    dense_file_metrics_list = []
    hybrid_file_metrics_list = []

    # By category: category -> list of metrics
    dense_by_cat: dict[str, list[dict[str, Any]]] = {}
    hybrid_by_cat: dict[str, list[dict[str, Any]]] = {}

    for idx, item in enumerate(eval_data, start=1):
        qid = item["id"]
        query = item["query"]
        category = item["query_type"]
        gt = item["ground_truth"]

        if category not in dense_by_cat:
            dense_by_cat[category] = []
            hybrid_by_cat[category] = []

        # -------------------------------------------------------------
        # Variant A: Dense Only
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        dense_chunks = run_dense_retrieval(repo_id=repo_id_in_chroma, query=query, k=3)
        dense_time_ms = (time.perf_counter() - t0) * 1000.0
        dense_latencies.append(dense_time_ms)

        dense_q_metrics = calculate_query_metrics(dense_chunks, gt, k=3, level="symbol")
        dense_file_metrics = calculate_query_metrics(dense_chunks, gt, k=3, level="file")
        dense_metrics_list.append(dense_q_metrics)
        dense_file_metrics_list.append(dense_file_metrics)
        dense_by_cat[category].append(dense_q_metrics)

        # -------------------------------------------------------------
        # Variant B: Existing Hybrid
        # -------------------------------------------------------------
        t1 = time.perf_counter()
        hybrid_chunks = run_hybrid_retrieval(repo_id=repo_id_in_chroma, query=query, k=3)
        hybrid_time_ms = (time.perf_counter() - t1) * 1000.0
        hybrid_latencies.append(hybrid_time_ms)

        hybrid_q_metrics = calculate_query_metrics(hybrid_chunks, gt, k=3, level="symbol")
        hybrid_file_metrics = calculate_query_metrics(hybrid_chunks, gt, k=3, level="file")
        hybrid_metrics_list.append(hybrid_q_metrics)
        hybrid_file_metrics_list.append(hybrid_file_metrics)
        hybrid_by_cat[category].append(hybrid_q_metrics)

        # Detailed record
        record = {
            "question_id": qid,
            "category": category,
            "query": query,
            "ground_truth": gt,
            "reference_answer": item.get("reference_answer", ""),
            "variants": {
                "dense_only": {
                    "latency_ms": round(dense_time_ms, 2),
                    "retrieved_results": dense_chunks,
                    "metrics_symbol": dense_q_metrics,
                    "metrics_file": dense_file_metrics,
                },
                "hybrid": {
                    "latency_ms": round(hybrid_time_ms, 2),
                    "retrieved_results": hybrid_chunks,
                    "metrics_symbol": hybrid_q_metrics,
                    "metrics_file": hybrid_file_metrics,
                },
            },
        }
        detailed_results.append(record)

        # Progress indicator
        d_hit = "HIT " if dense_q_metrics["hit_at_3"] > 0 else "MISS"
        h_hit = "HIT " if hybrid_q_metrics["hit_at_3"] > 0 else "MISS"
        print(f"[{idx:02d}/40] {qid} [{category:<18}] Dense(Hit@3): {d_hit} | Hybrid(Hit@3): {h_hit} | Dense:{dense_time_ms:.1f}ms Hybrid:{hybrid_time_ms:.1f}ms")

    # -----------------------------------------------------------------
    # Compute Aggregate Metrics
    # -----------------------------------------------------------------
    dense_overall = compute_aggregate_metrics(dense_metrics_list, k=3)
    hybrid_overall = compute_aggregate_metrics(hybrid_metrics_list, k=3)
    dense_file_overall = compute_aggregate_metrics(dense_file_metrics_list, k=3)
    hybrid_file_overall = compute_aggregate_metrics(hybrid_file_metrics_list, k=3)

    dense_latency_stats = compute_latency_stats(dense_latencies)
    hybrid_latency_stats = compute_latency_stats(hybrid_latencies)

    # Category aggregates
    cat_summary = []
    for cat in dense_by_cat:
        d_cat_agg = compute_aggregate_metrics(dense_by_cat[cat], k=3)
        h_cat_agg = compute_aggregate_metrics(hybrid_by_cat[cat], k=3)
        cat_summary.append({
            "category": cat,
            "count": d_cat_agg["count"],
            "dense": d_cat_agg,
            "hybrid": h_cat_agg,
        })

    # -----------------------------------------------------------------
    # Failure Analysis Categorization
    # -----------------------------------------------------------------
    dense_win = []   # Dense succeeds, Hybrid fails
    hybrid_win = []  # Hybrid succeeds, Dense fails
    both_fail = []   # Both fail
    both_succeed_diff = []  # Both succeed with different retrieved sources

    for rec in detailed_results:
        d_hit = rec["variants"]["dense_only"]["metrics_symbol"]["hit_at_3"] > 0
        h_hit = rec["variants"]["hybrid"]["metrics_symbol"]["hit_at_3"] > 0

        d_top = rec["variants"]["dense_only"]["retrieved_results"]
        h_top = rec["variants"]["hybrid"]["retrieved_results"]

        item_info = {
            "id": rec["question_id"],
            "category": rec["category"],
            "query": rec["query"],
            "ground_truth": rec["ground_truth"]["primary_symbols"],
            "dense_top": [(c["file_path"], c["symbol_name"], c.get("score")) for c in d_top],
            "hybrid_top": [(c["file_path"], c["symbol_name"], c.get("score")) for c in h_top],
        }

        if d_hit and not h_hit:
            dense_win.append(item_info)
        elif h_hit and not d_hit:
            hybrid_win.append(item_info)
        elif not d_hit and not h_hit:
            both_fail.append(item_info)
        else:
            # Both succeeded. Check if their top 1 result differs
            d_first = (d_top[0]["file_path"], d_top[0]["symbol_name"]) if d_top else None
            h_first = (h_top[0]["file_path"], h_top[0]["symbol_name"]) if h_top else None
            if d_first != h_first:
                both_succeed_diff.append(item_info)

    # -----------------------------------------------------------------
    # Save Machine-Readable Results (JSON)
    # -----------------------------------------------------------------
    full_output = {
        "benchmark_metadata": {
            "timestamp": start_time.isoformat(),
            "repository": repo_cfg["name"],
            "repo_id": repo_cfg["id"],
            "chroma_collection": f"repo_{repo_id_in_chroma}",
            "git_commit_sha": git_sha,
            "python_version": platform.python_version(),
            "embedding_model": "all-MiniLM-L6-v2",
            "retrieval_k": 3,
            "total_questions": len(eval_data),
        },
        "overall_summary": {
            "dense_only_symbol": {**dense_overall, **dense_latency_stats},
            "hybrid_symbol": {**hybrid_overall, **hybrid_latency_stats},
            "dense_only_file": dense_file_overall,
            "hybrid_file": hybrid_file_overall,
        },
        "category_summary": cat_summary,
        "failure_analysis": {
            "dense_wins_hybrid_fails": dense_win,
            "hybrid_wins_dense_fails": hybrid_win,
            "both_fail": both_fail,
            "both_succeed_different_top_sources": both_succeed_diff,
        },
        "per_query_results": detailed_results,
    }

    json_path = results_dir / "retrieval_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"\n[OK] Saved detailed JSON results to {json_path}")

    # -----------------------------------------------------------------
    # Save CSV Results
    # -----------------------------------------------------------------
    csv_path = results_dir / "retrieval_results.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "question_id", "category", "query",
            "dense_hit1", "dense_hit3", "dense_recall3", "dense_prec3", "dense_mrr", "dense_lat_ms",
            "hybrid_hit1", "hybrid_hit3", "hybrid_recall3", "hybrid_prec3", "hybrid_mrr", "hybrid_lat_ms",
            "mrr_delta"
        ])
        for rec in detailed_results:
            dm = rec["variants"]["dense_only"]["metrics_symbol"]
            hm = rec["variants"]["hybrid"]["metrics_symbol"]
            d_lat = rec["variants"]["dense_only"]["latency_ms"]
            h_lat = rec["variants"]["hybrid"]["latency_ms"]
            mrr_delta = round(hm["mrr"] - dm["mrr"], 4)
            writer.writerow([
                rec["question_id"], rec["category"], rec["query"],
                dm["hit_at_1"], dm["hit_at_3"], dm["recall_at_3"], dm["precision_at_3"], dm["mrr"], d_lat,
                hm["hit_at_1"], hm["hit_at_3"], hm["recall_at_3"], hm["precision_at_3"], hm["mrr"], h_lat,
                mrr_delta,
            ])
    print(f"[OK] Saved CSV results to {csv_path}")

    # -----------------------------------------------------------------
    # Generate Human-Readable Markdown Report
    # -----------------------------------------------------------------
    report_path = results_dir / "retrieval_report.md"
    generate_markdown_report(report_path, full_output)
    print(f"[OK] Generated summary report at {report_path}\n")

    # -----------------------------------------------------------------
    # Print Console Summary Tables
    # -----------------------------------------------------------------
    print_console_summary(full_output)


def generate_markdown_report(report_path: Path, data: dict[str, Any]):
    meta = data["benchmark_metadata"]
    dense = data["overall_summary"].get("dense_only_symbol", data["overall_summary"].get("dense_only", {}))
    hybrid = data["overall_summary"].get("hybrid_symbol", data["overall_summary"].get("hybrid", {}))
    dense_file = data["overall_summary"].get("dense_only_file", {})
    hybrid_file = data["overall_summary"].get("hybrid_file", {})
    cats = data["category_summary"]
    fails = data["failure_analysis"]

    # Improvements
    mrr_diff = hybrid["mrr"] - dense["mrr"]
    mrr_rel = (mrr_diff / dense["mrr"] * 100.0) if dense["mrr"] > 0 else 0.0

    hit3_diff = hybrid["hit_at_3"] - dense["hit_at_3"]
    hit3_rel = (hit3_diff / dense["hit_at_3"] * 100.0) if dense["hit_at_3"] > 0 else 0.0

    rec3_diff = hybrid["recall_at_3"] - dense["recall_at_3"]
    rec3_rel = (rec3_diff / dense["recall_at_3"] * 100.0) if dense["recall_at_3"] > 0 else 0.0

    prec3_diff = hybrid["precision_at_3"] - dense["precision_at_3"]
    prec3_rel = (prec3_diff / dense["precision_at_3"] * 100.0) if dense["precision_at_3"] > 0 else 0.0

    lines = [
        "# CodeLens Retrieval Evaluation (Step 2)",
        "",
        "## Benchmark Configuration",
        f"- **Repository**: `{meta['repository']}`",
        f"- **Git Commit SHA**: `{meta['git_commit_sha']}`",
        f"- **ChromaDB Collection**: `{meta['chroma_collection']}`",
        f"- **Embedding Model**: `{meta['embedding_model']}` (384 dim)",
        f"- **Number of Questions**: `{meta['total_questions']}`",
        f"- **Retrieval Depth (K)**: `{meta['retrieval_k']}`",
        f"- **Evaluation Timestamp**: `{meta['timestamp']}`",
        f"- **Python Version**: `{meta['python_version']}`",
        "",
        "---",
        "",
        "## 1. Overall Results (Symbol-Level Strict Match)",
        "",
        "| Variant | Hit@1 | Hit@3 | Recall@3 | Precision@3 | MRR | Mean Latency | P50 Latency | P95 Latency |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        f"| **Dense Only** | {dense['hit_at_1'] * 100:.1f}% | {dense['hit_at_3'] * 100:.1f}% | {dense['recall_at_3'] * 100:.1f}% | {dense['precision_at_3'] * 100:.1f}% | {dense['mrr']:.4f} | {dense['mean_ms']:.2f} ms | {dense['p50_ms']:.2f} ms | {dense['p95_ms']:.2f} ms |",
        f"| **Existing Hybrid** | {hybrid['hit_at_1'] * 100:.1f}% | {hybrid['hit_at_3'] * 100:.1f}% | {hybrid['recall_at_3'] * 100:.1f}% | {hybrid['precision_at_3'] * 100:.1f}% | {hybrid['mrr']:.4f} | {hybrid['mean_ms']:.2f} ms | {hybrid['p50_ms']:.2f} ms | {hybrid['p95_ms']:.2f} ms |",
        "",
        "### 1.1. Navigation-Level Results (File-Level Match)",
        "",
        "| Variant | Hit@1 | Hit@3 | Recall@3 | Precision@3 | MRR |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
        f"| **Dense Only** | {dense_file.get('hit_at_1', 0) * 100:.1f}% | {dense_file.get('hit_at_3', 0) * 100:.1f}% | {dense_file.get('recall_at_3', 0) * 100:.1f}% | {dense_file.get('precision_at_3', 0) * 100:.1f}% | {dense_file.get('mrr', 0):.4f} |",
        f"| **Existing Hybrid** | {hybrid_file.get('hit_at_1', 0) * 100:.1f}% | {hybrid_file.get('hit_at_3', 0) * 100:.1f}% | {hybrid_file.get('recall_at_3', 0) * 100:.1f}% | {hybrid_file.get('precision_at_3', 0) * 100:.1f}% | {hybrid_file.get('mrr', 0):.4f} |",
        "",
        "---",
        "",
        "## 2. Hybrid Improvement Over Dense",
        "",
        "| Metric | Dense Baseline | Hybrid | Absolute Δ | Relative Improvement |",
        "|:---|:---:|:---:|:---:|:---:|",
        f"| **MRR** | {dense['mrr']:.4f} | {hybrid['mrr']:.4f} | {mrr_diff:+.4f} | {mrr_rel:+.2f}% |",
        f"| **Hit@3** | {dense['hit_at_3'] * 100:.1f}% | {hybrid['hit_at_3'] * 100:.1f}% | {hit3_diff * 100:+.1f}% | {hit3_rel:+.2f}% |",
        f"| **Hit@1** | {dense['hit_at_1'] * 100:.1f}% | {hybrid['hit_at_1'] * 100:.1f}% | {(hybrid['hit_at_1'] - dense['hit_at_1']) * 100:+.1f}% | {((hybrid['hit_at_1'] - dense['hit_at_1']) / dense['hit_at_1'] * 100 if dense['hit_at_1'] > 0 else 0.0):+.2f}% |",
        f"| **Recall@3** | {dense['recall_at_3'] * 100:.1f}% | {hybrid['recall_at_3'] * 100:.1f}% | {rec3_diff * 100:+.1f}% | {rec3_rel:+.2f}% |",
        f"| **Precision@3** | {dense['precision_at_3'] * 100:.1f}% | {hybrid['precision_at_3'] * 100:.1f}% | {prec3_diff * 100:+.1f}% | {prec3_rel:+.2f}% |",
        "",
        "> [!NOTE]",
        "> These measurements reflect performance on the 40-question CodeLens repository benchmark. Due to the sample size, they should be evaluated as benchmark indicators rather than universal statistical generalizations.",
        "",
        "---",
        "",
        "## 3. Category Breakdown",
        "",
        "| Category | Qs | Dense Hit@3 | Hybrid Hit@3 | Dense MRR | Hybrid MRR | MRR Δ |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for c in cats:
        cat_name = c["category"]
        cnt = c["count"]
        d_h3 = c["dense"]["hit_at_3"] * 100
        h_h3 = c["hybrid"]["hit_at_3"] * 100
        d_mrr = c["dense"]["mrr"]
        h_mrr = c["hybrid"]["mrr"]
        delta_mrr = h_mrr - d_mrr
        lines.append(
            f"| `{cat_name}` | {cnt} | {d_h3:.1f}% | {h_h3:.1f}% | {d_mrr:.4f} | {h_mrr:.4f} | {delta_mrr:+.4f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Latency Breakdown",
        "",
        "| Retrieval Strategy | Mean Latency | Median (P50) | 95th Percentile (P95) |",
        "|:---|:---:|:---:|:---:|",
        f"| **Dense Only** | {dense['mean_ms']:.2f} ms | {dense['p50_ms']:.2f} ms | {dense['p95_ms']:.2f} ms |",
        f"| **Existing Hybrid (Dense + BM25 + RRF)** | {hybrid['mean_ms']:.2f} ms | {hybrid['p50_ms']:.2f} ms | {hybrid['p95_ms']:.2f} ms |",
        "",
        f"**Latency Cost of Hybrid Fusion**: +{hybrid['mean_ms'] - dense['mean_ms']:.2f} ms mean overhead (~{((hybrid['mean_ms'] - dense['mean_ms']) / dense['mean_ms'] * 100.0) if dense['mean_ms'] > 0 else 0.0:.1f}%).",
        "",
        "---",
        "",
        "## 5. Failure Analysis",
        "",
        f"### 5.1. Hybrid Succeeds, Dense Fails ({len(fails['hybrid_wins_dense_fails'])} Questions)",
        "Queries where exact lexical tokens and BM25 elevated the correct code which dense vector similarity alone missed:",
        "",
    ])

    if not fails["hybrid_wins_dense_fails"]:
        lines.append("*None.*")
    else:
        for it in fails["hybrid_wins_dense_fails"]:
            gt_str = ", ".join(f"`{sym['file']}::{sym['symbol']}`" for sym in it["ground_truth"])
            d_str = ", ".join(f"`{f}::{s}`" for f, s, sc in it["dense_top"])
            h_str = ", ".join(f"`{f}::{s}`" for f, s, sc in it["hybrid_top"])
            lines.extend([
                f"- **{it['id']}** [`{it['category']}`]: *\"{it['query']}\"*",
                f"  - **Ground Truth**: {gt_str}",
                f"  - **Dense Top-3**: {d_str}",
                f"  - **Hybrid Top-3**: {h_str}",
                "",
            ])

    lines.extend([
        f"### 5.2. Dense Succeeds, Hybrid Fails ({len(fails['dense_wins_hybrid_fails'])} Questions)",
        "Queries where semantic embeddings captured intent, but lexical RRF ranking diluted the relevant result out of Top-3:",
        "",
    ])

    if not fails["dense_wins_hybrid_fails"]:
        lines.append("*None.*")
    else:
        for it in fails["dense_wins_hybrid_fails"]:
            gt_str = ", ".join(f"`{sym['file']}::{sym['symbol']}`" for sym in it["ground_truth"])
            d_str = ", ".join(f"`{f}::{s}`" for f, s, sc in it["dense_top"])
            h_str = ", ".join(f"`{f}::{s}`" for f, s, sc in it["hybrid_top"])
            lines.extend([
                f"- **{it['id']}** [`{it['category']}`]: *\"{it['query']}\"*",
                f"  - **Ground Truth**: {gt_str}",
                f"  - **Dense Top-3**: {d_str}",
                f"  - **Hybrid Top-3**: {h_str}",
                "",
            ])

    lines.extend([
        f"### 5.3. Both Fail ({len(fails['both_fail'])} Questions)",
        "Queries where neither Dense nor Hybrid retrieved the target symbol in Top-3:",
        "",
    ])

    if not fails["both_fail"]:
        lines.append("*None.*")
    else:
        for it in fails["both_fail"]:
            gt_str = ", ".join(f"`{sym['file']}::{sym['symbol']}`" for sym in it["ground_truth"])
            d_str = ", ".join(f"`{f}::{s}`" for f, s, sc in it["dense_top"])
            h_str = ", ".join(f"`{f}::{s}`" for f, s, sc in it["hybrid_top"])
            lines.extend([
                f"- **{it['id']}** [`{it['category']}`]: *\"{it['query']}\"*",
                f"  - **Ground Truth**: {gt_str}",
                f"  - **Dense Top-3**: {d_str}",
                f"  - **Hybrid Top-3**: {h_str}",
                "",
            ])

    lines.extend([
        f"### 5.4. Both Succeed with Different Top Source ({len(fails['both_succeed_different_top_sources'])} Questions)",
        "Queries where both variants found relevant code in Top-3, but ranked different symbols at #1:",
        "",
    ])

    if not fails["both_succeed_different_top_sources"]:
        lines.append("*None.*")
    else:
        for it in fails["both_succeed_different_top_sources"]:
            gt_str = ", ".join(f"`{sym['file']}::{sym['symbol']}`" for sym in it["ground_truth"])
            d_first = it["dense_top"][0] if it["dense_top"] else ("None", "None", 0)
            h_first = it["hybrid_top"][0] if it["hybrid_top"] else ("None", "None", 0)
            lines.extend([
                f"- **{it['id']}** [`{it['category']}`]: *\"{it['query']}\"*",
                f"  - **Ground Truth**: {gt_str}",
                f"  - **Dense #1**: `{d_first[0]}::{d_first[1]}` (score {d_first[2]})",
                f"  - **Hybrid #1**: `{h_first[0]}::{h_first[1]}` (score {h_first[2]})",
                "",
            ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def print_console_summary(data: dict[str, Any]):
    dense = data["overall_summary"].get("dense_only_symbol", data["overall_summary"].get("dense_only", {}))
    hybrid = data["overall_summary"].get("hybrid_symbol", data["overall_summary"].get("hybrid", {}))
    dense_file = data["overall_summary"].get("dense_only_file", {})
    hybrid_file = data["overall_summary"].get("hybrid_file", {})
    cats = data["category_summary"]

    print("=" * 80)
    print("OVERALL RETRIEVAL PERFORMANCE (Symbol-Level Match)")
    print("=" * 80)
    print(f"{'Variant':<18} | {'Hit@1':<8} | {'Hit@3':<8} | {'Recall@3':<10} | {'Prec@3':<8} | {'MRR':<8} | {'Mean Latency':<12}")
    print("-" * 80)
    print(f"{'Dense Only':<18} | {dense['hit_at_1']*100:>6.1f}% | {dense['hit_at_3']*100:>6.1f}% | {dense['recall_at_3']*100:>8.1f}% | {dense['precision_at_3']*100:>6.1f}% | {dense['mrr']:>8.4f} | {dense['mean_ms']:>8.2f} ms")
    print(f"{'Existing Hybrid':<18} | {hybrid['hit_at_1']*100:>6.1f}% | {hybrid['hit_at_3']*100:>6.1f}% | {hybrid['recall_at_3']*100:>8.1f}% | {hybrid['precision_at_3']*100:>6.1f}% | {hybrid['mrr']:>8.4f} | {hybrid['mean_ms']:>8.2f} ms")
    print("-" * 80)

    mrr_diff = hybrid["mrr"] - dense["mrr"]
    mrr_rel = (mrr_diff / dense["mrr"] * 100.0) if dense["mrr"] > 0 else 0.0
    print(f"Hybrid MRR Gain : {mrr_diff:+.4f} ({mrr_rel:+.2f}%)")
    print(f"Latency Overhead: +{hybrid['mean_ms'] - dense['mean_ms']:.2f} ms\n")

    if dense_file and hybrid_file:
        print("=" * 80)
        print("OVERALL RETRIEVAL PERFORMANCE (File-Level Match)")
        print("=" * 80)
        print(f"{'Variant':<18} | {'Hit@1':<8} | {'Hit@3':<8} | {'Recall@3':<10} | {'Prec@3':<8} | {'MRR':<8}")
        print("-" * 80)
        print(f"{'Dense Only':<18} | {dense_file['hit_at_1']*100:>6.1f}% | {dense_file['hit_at_3']*100:>6.1f}% | {dense_file['recall_at_3']*100:>8.1f}% | {dense_file['precision_at_3']*100:>6.1f}% | {dense_file['mrr']:>8.4f}")
        print(f"{'Existing Hybrid':<18} | {hybrid_file['hit_at_1']*100:>6.1f}% | {hybrid_file['hit_at_3']*100:>6.1f}% | {hybrid_file['recall_at_3']*100:>8.1f}% | {hybrid_file['precision_at_3']*100:>6.1f}% | {hybrid_file['mrr']:>8.4f}")
        print("-" * 80 + "\n")

    print("=" * 80)
    print("CATEGORY-LEVEL RETRIEVAL BREAKDOWN")
    print("=" * 80)
    print(f"{'Category':<20} | {'Qs':<3} | {'Dense Hit@3':<12} | {'Hybrid Hit@3':<13} | {'Dense MRR':<10} | {'Hybrid MRR':<11} | {'MRR Diff':<9}")
    print("-" * 80)
    for c in cats:
        d_h3 = c["dense"]["hit_at_3"] * 100
        h_h3 = c["hybrid"]["hit_at_3"] * 100
        d_mrr = c["dense"]["mrr"]
        h_mrr = c["hybrid"]["mrr"]
        delta = h_mrr - d_mrr
        print(f"{c['category']:<20} | {c['count']:<3} | {d_h3:>10.1f}% | {h_h3:>11.1f}% | {d_mrr:>10.4f} | {h_mrr:>11.4f} | {delta:>+8.4f}")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()

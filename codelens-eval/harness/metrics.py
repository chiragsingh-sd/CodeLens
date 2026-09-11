"""
Metrics calculation module for CodeLens retrieval evaluation.

Implements standard Information Retrieval (IR) metrics:
- Hit@1
- Hit@3
- Recall@3
- Precision@3
- MRR (Mean Reciprocal Rank)
- Latency percentiles (mean, p50, p95)
"""

from typing import Any


def calculate_query_metrics(
    retrieved_items: list[dict[str, Any]],
    ground_truth: dict[str, Any],
    k: int = 3,
    level: str = "symbol",
) -> dict[str, Any]:
    """
    Computes retrieval metrics for a single query.

    level: 'symbol' for strict (file_path + symbol_name) matching
           'file' for relaxed (file_path) matching against acceptable_files
    """
    top_k = retrieved_items[:k]

    if level == "symbol":
        gt_symbols = {
            (ps["file"].replace("\\", "/").strip(), ps["symbol"].strip())
            for ps in ground_truth.get("primary_symbols", [])
        }
        total_gt = len(gt_symbols)

        # Identify which retrieved chunks match any ground truth symbol
        relevant_mask = []
        retrieved_gt_matches = set()

        for chunk in top_k:
            c_file = chunk.get("file_path", "").replace("\\", "/").strip()
            c_sym = chunk.get("symbol_name", "").strip()
            key = (c_file, c_sym)

            if key in gt_symbols:
                relevant_mask.append(True)
                retrieved_gt_matches.add(key)
            else:
                relevant_mask.append(False)

        num_retrieved_gt = len(retrieved_gt_matches)

    elif level == "file":
        gt_files = {
            f.replace("\\", "/").strip()
            for f in ground_truth.get("acceptable_files", [])
        }
        # Also include files from primary_symbols if any
        for ps in ground_truth.get("primary_symbols", []):
            gt_files.add(ps["file"].replace("\\", "/").strip())

        total_gt = len(gt_files)

        relevant_mask = []
        retrieved_gt_matches = set()

        for chunk in top_k:
            c_file = chunk.get("file_path", "").replace("\\", "/").strip()
            if c_file in gt_files:
                relevant_mask.append(True)
                retrieved_gt_matches.add(c_file)
            else:
                relevant_mask.append(False)

        num_retrieved_gt = len(retrieved_gt_matches)

    else:
        raise ValueError(f"Unknown evaluation level: {level}")

    # First relevant rank (1-indexed)
    first_relevant_rank = None
    for rank_idx, is_rel in enumerate(relevant_mask, start=1):
        if is_rel:
            first_relevant_rank = rank_idx
            break

    # 1. Hit@1
    hit_at_1 = 1.0 if len(relevant_mask) >= 1 and relevant_mask[0] else 0.0

    # 2. Hit@K (Hit@3)
    hit_at_k = 1.0 if any(relevant_mask) else 0.0

    # 3. Recall@K (Recall@3)
    # fraction of ground truth items retrieved in top K
    recall_at_k = (num_retrieved_gt / total_gt) if total_gt > 0 else 0.0

    # 4. Precision@K (Precision@3)
    # fraction of top K items that are relevant
    precision_at_k = sum(1 for is_rel in relevant_mask if is_rel) / float(k)

    # 5. MRR (Mean Reciprocal Rank contribution)
    mrr = (1.0 / first_relevant_rank) if first_relevant_rank is not None else 0.0

    return {
        "hit_at_1": hit_at_1,
        f"hit_at_{k}": hit_at_k,
        f"recall_at_{k}": round(recall_at_k, 4),
        f"precision_at_{k}": round(precision_at_k, 4),
        "mrr": round(mrr, 4),
        "first_relevant_rank": first_relevant_rank,
        "total_ground_truth": total_gt,
        "matched_ground_truth": num_retrieved_gt,
    }


def compute_aggregate_metrics(per_query_metrics: list[dict[str, Any]], k: int = 3) -> dict[str, float]:
    """
    Computes mean metrics across a list of per-query metric dicts.
    """
    if not per_query_metrics:
        return {
            "hit_at_1": 0.0,
            f"hit_at_{k}": 0.0,
            f"recall_at_{k}": 0.0,
            f"precision_at_{k}": 0.0,
            "mrr": 0.0,
            "count": 0,
        }

    n = len(per_query_metrics)
    return {
        "count": n,
        "hit_at_1": round(sum(m["hit_at_1"] for m in per_query_metrics) / n, 4),
        f"hit_at_{k}": round(sum(m[f"hit_at_{k}"] for m in per_query_metrics) / n, 4),
        f"recall_at_{k}": round(sum(m[f"recall_at_{k}"] for m in per_query_metrics) / n, 4),
        f"precision_at_{k}": round(sum(m[f"precision_at_{k}"] for m in per_query_metrics) / n, 4),
        "mrr": round(sum(m["mrr"] for m in per_query_metrics) / n, 4),
    }


def compute_latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    """
    Calculates mean, p50 (median), and p95 latency in milliseconds.
    """
    if not latencies_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}

    sorted_lat = sorted(latencies_ms)
    n = len(sorted_lat)

    mean_val = sum(sorted_lat) / n

    # Median (p50)
    mid = n // 2
    if n % 2 == 1:
        p50 = sorted_lat[mid]
    else:
        p50 = (sorted_lat[mid - 1] + sorted_lat[mid]) / 2.0

    # p95
    p95_idx = int(round(0.95 * (n - 1)))
    p95 = sorted_lat[min(p95_idx, n - 1)]

    return {
        "mean_ms": round(mean_val, 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
    }

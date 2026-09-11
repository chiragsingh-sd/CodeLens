"""
Dataset validation script for CodeLens evaluation benchmark.

Validates the integrity, schema, distributions, and source-code grounding
of eval_set.json and benchmark_repos.json without executing retrieval.
"""

import ast
import json
import os
import sys
from pathlib import Path

EXPECTED_CATEGORIES = {
    "symbol_lookup": 8,
    "subsystem_flow": 8,
    "api_routes": 8,
    "configuration": 6,
    "cross_component": 6,
    "error_handling": 4,
}

EXPECTED_TOTAL_QUESTIONS = 40


def extract_symbols_from_py_file(file_path: Path) -> set[str]:
    """
    Extracts defined symbols (classes, functions, async functions, module-level assignments)
    from a Python source file using AST.
    """
    if not file_path.exists():
        return set()

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(content)
    except Exception as e:
        print(f"  [WARN] AST parsing failed for {file_path}: {e}")
        return set()

    symbols = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    symbols.add(target.id)

    return symbols


def validate_dataset(workspace_root: Path) -> bool:
    data_dir = workspace_root / "codelens-eval" / "data"
    bench_file = data_dir / "benchmark_repos.json"
    eval_file = data_dir / "eval_set.json"

    errors = []
    warnings = []

    print("=" * 70)
    print("CODELENS EVALUATION DATASET VALIDATION")
    print("=" * 70)

    # 1. Check benchmark_repos.json
    if not bench_file.exists():
        errors.append(f"benchmark_repos.json does not exist at {bench_file}")
        valid_repo_ids = set()
    else:
        try:
            with open(bench_file, "r", encoding="utf-8") as f:
                bench_data = json.load(f)
            repos = bench_data.get("repositories", [])
            valid_repo_ids = {r["id"] for r in repos if "id" in r}
            print(f"[OK] benchmark_repos.json loaded successfully ({len(repos)} repository defined)")
        except Exception as e:
            errors.append(f"benchmark_repos.json failed to parse: {e}")
            valid_repo_ids = set()

    # 2. Check eval_set.json existence & JSON parsing
    if not eval_file.exists():
        errors.append(f"eval_set.json does not exist at {eval_file}")
        print(f"[FATAL] {eval_file} not found.")
        return False

    try:
        with open(eval_file, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        print(f"[OK] eval_set.json parsed as valid JSON ({len(eval_data)} entries)")
    except Exception as e:
        errors.append(f"eval_set.json failed to parse JSON: {e}")
        print(f"[FATAL] JSON parse error: {e}")
        return False

    # 3. Total question count
    if len(eval_data) != EXPECTED_TOTAL_QUESTIONS:
        errors.append(
            f"Question count mismatch: expected {EXPECTED_TOTAL_QUESTIONS}, got {len(eval_data)}"
        )
    else:
        print(f"[OK] Question count matches expected: {EXPECTED_TOTAL_QUESTIONS}")

    # 4. ID uniqueness & question uniqueness
    seen_ids = set()
    seen_queries = set()
    category_counts = {}

    file_symbol_cache = {}

    for idx, item in enumerate(eval_data):
        qid = item.get("id")
        query = item.get("query", "").strip()
        category = item.get("query_type")
        repo = item.get("repo")
        gt = item.get("ground_truth", {})
        ref_ans = item.get("reference_answer", "")

        entry_prefix = f"Entry [{idx}] ID={qid}:"

        # Unique ID
        if not qid:
            errors.append(f"{entry_prefix} missing 'id'")
        elif qid in seen_ids:
            errors.append(f"{entry_prefix} duplicate ID '{qid}'")
        else:
            seen_ids.add(qid)

        # Unique Query
        if not query:
            errors.append(f"{entry_prefix} missing 'query'")
        elif query.lower() in seen_queries:
            errors.append(f"{entry_prefix} duplicate exact query: '{query}'")
        else:
            seen_queries.add(query.lower())

        # Reference Answer
        if not ref_ans or not ref_ans.strip():
            errors.append(f"{entry_prefix} missing 'reference_answer'")

        # Category check
        if not category:
            errors.append(f"{entry_prefix} missing 'query_type'")
        elif category not in EXPECTED_CATEGORIES:
            errors.append(f"{entry_prefix} unknown category '{category}'")
        else:
            category_counts[category] = category_counts.get(category, 0) + 1

        # Repo reference check
        if not repo:
            errors.append(f"{entry_prefix} missing 'repo'")
        elif repo not in valid_repo_ids:
            errors.append(f"{entry_prefix} repo '{repo}' not registered in benchmark_repos.json")

        # Ground truth schema
        if not isinstance(gt, dict):
            errors.append(f"{entry_prefix} ground_truth must be a dictionary")
            continue

        primary_symbols = gt.get("primary_symbols")
        acceptable_files = gt.get("acceptable_files")

        if not isinstance(primary_symbols, list) or len(primary_symbols) == 0:
            errors.append(f"{entry_prefix} ground_truth.primary_symbols must be a non-empty list")
        else:
            for s_idx, sym_entry in enumerate(primary_symbols):
                if not isinstance(sym_entry, dict):
                    errors.append(f"{entry_prefix} primary_symbol[{s_idx}] must be a dict")
                    continue
                s_file = sym_entry.get("file")
                s_name = sym_entry.get("symbol")

                if not s_file or not s_name:
                    errors.append(f"{entry_prefix} primary_symbol[{s_idx}] missing 'file' or 'symbol'")
                    continue

                # Check file existence
                target_file_path = workspace_root / s_file
                if not target_file_path.exists():
                    errors.append(f"{entry_prefix} primary symbol file does not exist: {s_file}")
                else:
                    # Check symbol exists in source file via AST
                    if s_file not in file_symbol_cache:
                        file_symbol_cache[s_file] = extract_symbols_from_py_file(target_file_path)

                    known_symbols = file_symbol_cache[s_file]
                    if s_name not in known_symbols:
                        # Check fallback if symbol is in content (e.g. inner class or special assignment)
                        file_text = target_file_path.read_text(encoding="utf-8", errors="replace")
                        if s_name not in file_text:
                            errors.append(
                                f"{entry_prefix} symbol '{s_name}' could not be verified in {s_file}"
                            )
                        else:
                            warnings.append(
                                f"{entry_prefix} symbol '{s_name}' verified via text match in {s_file} (AST walk did not capture direct top-level node)"
                            )

        if not isinstance(acceptable_files, list) or len(acceptable_files) == 0:
            errors.append(f"{entry_prefix} ground_truth.acceptable_files must be a non-empty list")
        else:
            for a_file in acceptable_files:
                acc_path = workspace_root / a_file
                if not acc_path.exists():
                    errors.append(f"{entry_prefix} acceptable file does not exist: {a_file}")

    # 5. Category Distribution Check
    print("\n--- CATEGORY DISTRIBUTION CHECK ---")
    dist_mismatch = False
    for cat, exp_count in EXPECTED_CATEGORIES.items():
        actual_count = category_counts.get(cat, 0)
        status = "OK" if actual_count == exp_count else "FAIL"
        print(f"  [{status}] {cat:<22}: expected {exp_count}, got {actual_count}")
        if actual_count != exp_count:
            errors.append(
                f"Category distribution mismatch for '{cat}': expected {exp_count}, got {actual_count}"
            )
            dist_mismatch = True

    if not dist_mismatch:
        print("[OK] Category distribution matches exact specification.")

    print("\n--- VALIDATION SUMMARY ---")
    print(f"Total Questions Evaluated: {len(eval_data)}")
    print(f"Total Errors Found       : {len(errors)}")
    print(f"Total Warnings Found     : {len(warnings)}")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print("\nERRORS DETECTED:")
        for e in errors:
            print(f"  [ERROR] {e}")
        print("\nDATASET VALIDATION FAILED!")
        return False
    else:
        print("\nDATASET VALIDATION PASSED! ALL CHECKS SUCCESSFUL.")
        return True


if __name__ == "__main__":
    # Resolve repository workspace root
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    success = validate_dataset(repo_root)
    sys.exit(0 if success else 1)

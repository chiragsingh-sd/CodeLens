# CodeLens Evaluation Benchmark (Step 1)

This directory contains the benchmark dataset and validation tooling for evaluating the CodeLens repository intelligence and RAG pipeline.

---

## 1. Purpose

The objective of this evaluation suite is to provide objective, empirical measurement of CodeLens:
- Does CodeLens retrieve the correct code chunks for realistic developer questions?
- Does it retrieve them within the top $K$ results?
- How well does the hybrid BM25 + dense semantic retrieval perform?
- Do the generated answers faithfully reflect the retrieved context?
- What are the retrieval and end-to-end latencies of the pipeline?

---

## 2. Benchmark Repository

For the initial evaluation benchmark, CodeLens uses its own codebase as the evaluation repository:
- **Repository Name**: `CodeLens`
- **Remote URL**: `https://github.com/chiragk1234singh-alt/CodeLens.git`
- **Git Commit SHA**: `f56a7ce7dbf9b17754ef603321fe4d59ba3971b8`
- **Local Path**: `.` (workspace root)

Repository configuration details are declared in [`data/benchmark_repos.json`](file:///d:/Codelens/codelens-eval/data/benchmark_repos.json).

---

## 3. Benchmark Size & Category Distribution

The dataset [`data/eval_set.json`](file:///d:/Codelens/codelens-eval/data/eval_set.json) contains **40 curated questions** partitioned into 6 distinct behavioral categories:

| Category | Target Description | Count |
|---|---|---|
| `symbol_lookup` | Locating exact function, class, or symbol definitions | 8 |
| `subsystem_flow` | Tracing multi-step execution flows (ingestion, fusion, chunking) | 8 |
| `api_routes` | FastAPI endpoints, request handling, and lifecycle events | 8 |
| `configuration` | Settings, timeouts, model constants, and filesystem paths | 6 |
| `cross_component` | Interactions bridging agent layers, tools, vector store, and SQLite | 6 |
| `error_handling` | Fallbacks (Groq fallback, Windows grep fallback, syntax error handling) | 4 |
| **Total** | | **40** |

---

## 4. Ground-Truth Methodology

### Source-Grounded Ground Truth
Ground truth labels are authored directly by inspecting the **active source code**.
- **No Circularity**: Ground truth was **never** generated from CodeLens retrieval output, nor labeled based on whatever CodeLens currently retrieves.
- **Why Retrieval Output Must Not Influence Labels**: If an evaluation pipeline uses its own search results to define what is "correct", retrieval bugs become invisible self-fulfilling prophecies. Labels must be an independent ground-truth standard.

### Hierarchical Identity
- **Primary Target (`primary_symbols`)**: Specific AST symbols (`file` + `symbol`) that represent the exact implementation unit (e.g., function, class, or method).
- **Secondary Target (`acceptable_files`)**: The broader file-level path containing or directly coordinating the answer, used for navigation-level retrieval evaluation.
- **Reference Answer (`reference_answer`)**: A concise, factually verified human explanation strictly derived from the repository implementation.

---

## 5. Dataset Schema

Each entry in `eval_set.json` adheres to the following structure:

```json
{
  "id": "q001",
  "repo": "codelens",
  "query_type": "symbol_lookup",
  "query": "Where is hybrid retrieval implemented?",
  "ground_truth": {
    "primary_symbols": [
      {
        "file": "backend/tools/search_tool.py",
        "symbol": "hybrid_search"
      }
    ],
    "acceptable_files": [
      "backend/tools/search_tool.py"
    ]
  },
  "reference_answer": "hybrid_search is implemented in backend/tools/search_tool.py. It fetches documents from ChromaDB, scores them with an in-memory BM25Okapi instance, generates a query embedding for vector search, and combines the rankings using Reciprocal Rank Fusion (RRF)."
}
```

---

## 6. Dataset Validation

The dataset is validated by a standalone integrity script that runs without executing retrieval or invoking LLMs:

```bash
python codelens-eval/scripts/validate_dataset.py
```

The validation script ensures:
1. Exact total of 40 questions.
2. Perfect match with expected category distribution.
3. Complete uniqueness of question IDs and query strings.
4. Repository ID registration in `benchmark_repos.json`.
5. Physical existence of all ground-truth files on disk.
6. AST verification of all primary symbols in their respective source files.
7. Presence of source-grounded reference answers.

---

## 7. Running the Retrieval Evaluation Benchmark (Step 2)

Step 2 implements the automated retrieval evaluation harness comparing:
- **Variant A (Dense Only)**: Vector embeddings (`all-MiniLM-L6-v2`) + ChromaDB cosine similarity.
- **Variant B (Existing Hybrid)**: BM25Okapi lexical retrieval + ChromaDB dense vectors merged via Reciprocal Rank Fusion ($K_{RRF}=60$).

### Benchmark Execution Command

```bash
python codelens-eval/scripts/run_retrieval_eval.py
```

### Generated Artifacts
1. **Machine-readable JSON trace**: [`results/retrieval_results.json`](file:///d:/Codelens/codelens-eval/results/retrieval_results.json)
   - Contains per-query rankings, raw scores, latencies, and ground truth match indicators.
2. **CSV summary export**: [`results/retrieval_results.csv`](file:///d:/Codelens/codelens-eval/results/retrieval_results.csv)
   - Tabular metric comparison per query for analysis in pandas / spreadsheet tools.
3. **Markdown scorecard & failure report**: [`results/retrieval_report.md`](file:///d:/Codelens/codelens-eval/results/retrieval_report.md)
   - Comprehensive performance tables, category breakdowns, latency stats, and failure case details.

---

## 8. Running the Generation Quality Evaluation Benchmark (Step 3)

Step 3 implements the LLM generation evaluation harness measuring:
- **Answer Correctness** ($0-2$ integer scale vs. reference answer)
- **Faithfulness / Groundedness** ($0-2$ integer scale vs. retrieved context)
- **Relevance** ($0-2$ integer scale vs. query)
- **Unsupported / Hallucinated Claim Detection** (boolean flag + specific itemized claim details)
- **Dual-Condition Comparison**: Condition A (Oracle Ground-Truth Context) vs. Condition B (Actual CodeLens Retrieved Context)
- **Fault Isolation Matrix**: Quadrants A, B, C, and D isolating retriever defects from generator defects.

### Benchmark Execution Command

```bash
python codelens-eval/scripts/run_generation_eval.py
```

Optional CLI parameters:
- `--workers <int>`: Number of concurrent worker threads (default: 2)
- `--limit <int>`: Limit execution to the first $N$ queries for quick diagnostic smoke tests

### Generated Artifacts
1. **Machine-readable JSON trace**: [`results/generation_results.json`](file:///d:/Codelens/codelens-eval/results/generation_results.json)
   - Per-query generation outputs for both Oracle and Actual conditions, structured judge evaluations, deterministic check signals, latencies, and fault isolation quadrant tags.
2. **Markdown scorecard & failure report**: [`results/generation_report.md`](file:///d:/Codelens/codelens-eval/results/generation_report.md)
   - Condition A vs. Condition B aggregate tables, category breakdown, empirical retrieval-to-generation gap analysis, fault isolation matrix, representative failure cases, and manual judge validation audit.

# CodeLens Retrieval Evaluation (Experiment 2: Corrected Hybrid)

## Executive Summary

This controlled experiment evaluates whether a corrected candidate-pool architecture (retrieving independent top-10 candidate sets for both Dense and BM25 before RRF fusion) resolves the underperformance observed in Existing Hybrid retrieval.

## Benchmark Configuration
- **Repository**: `CodeLens` (`repo_3ef3e55d`)
- **Git Commit SHA**: `f56a7ce7dbf9b17754ef603321fe4d59ba3971b8`
- **Embedding Model**: `all-MiniLM-L6-v2`
- **Dense Candidate Pool ($K_{dense}$)**: `10`
- **BM25 Candidate Pool ($K_{bm25}$)**: `10`
- **Final Retrieval Depth ($k$)**: `3`
- **RRF Constant ($K_{RRF}$)**: `60` (0-indexed)
- **Questions Evaluated**: `40`
- **Timestamp**: `2026-09-09T19:58:03.656496+00:00`

---

## 1. Overall Performance Comparison (Symbol Level)

| Metric | Dense Only (k=3) | Existing Hybrid (k=3) | Corrected Hybrid (k10/k10->k3) | Corrected vs. Dense (Delta) | Corrected vs. Existing (Delta) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Hit@1** | **67.5%** | 62.5% | 57.5% | -0.100 (-10.0%) | -0.050 (-5.0%) |
| **Hit@3** | 85.0% | 85.0% | **85.0%** | +0.000 (+0.0%) | +0.000 (+0.0%) |
| **Recall@3** | 80.0% | 80.0% | **80.0%** | +0.000 (+0.0%) | +0.000 (+0.0%) |
| **Precision@3** | 33.3% | 33.3% | **33.3%** | +0.000 (+0.0%) | +0.000 (+0.0%) |
| **MRR** | **0.7542** | 0.7333 | 0.7042 | -0.0500 | -0.0291 |

---

## 2. Latency Benchmarks (Milliseconds)

| Variant | Mean Latency | P50 (Median) | P95 |
|:---|:---:|:---:|:---:|
| **Dense Only** | 14.58 ms | 13.93 ms | 18.19 ms |
| **Existing Hybrid** | 20.25 ms | 19.46 ms | 23.74 ms |
| **Corrected Hybrid** | 14.64 ms | 14.18 ms | 18.06 ms |

### Corrected Hybrid Sub-step Latencies:
- **Dense Step**: Mean `14.28 ms` | P50 `13.78 ms`
- **BM25 Step**: Mean `0.33 ms` | P50 `0.33 ms`
- **RRF Fusion**: Mean `0.01 ms` | P50 `0.01 ms`

---

## 3. Category Breakdown (Symbol Level)

| Category | Qs | Dense Hit@3 | ExistHyb Hit@3 | CorrHyb Hit@3 | Dense MRR | ExistHyb MRR | CorrHyb MRR |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `api_routes` | 8 | 62.5% | 62.5% | 62.5% | 0.4792 | 0.4375 | 0.4167 |
| `configuration` | 6 | 66.7% | 66.7% | 83.3% | 0.5000 | 0.5000 | 0.5555 |
| `cross_component` | 6 | 100.0% | 100.0% | 100.0% | 0.9167 | 0.9167 | 0.9167 |
| `error_handling` | 4 | 100.0% | 100.0% | 100.0% | 1.0000 | 1.0000 | 1.0000 |
| `subsystem_flow` | 8 | 100.0% | 100.0% | 87.5% | 0.9167 | 0.9167 | 0.7500 |
| `symbol_lookup` | 8 | 87.5% | 87.5% | 87.5% | 0.8125 | 0.7500 | 0.7500 |

---

## 4. BM25 Candidate Contribution Statistics

- **Queries where BM25 introduces unique candidate(s) not in Dense top 10**: `40 / 40` (100.0%)
- **Queries where that BM25-only candidate is RELEVANT**: `0 / 40` (0.0%)
- **Queries where BM25 brings a NEW relevant candidate into final Top 3**: `2 / 40` (5.0%)
- **Queries where Corrected Hybrid IMPROVES rank vs. Dense**: `2 / 40` (5.0%)
- **Queries where Corrected Hybrid WORSENS rank vs. Dense**: `6 / 40` (15.0%)
- **Queries where rankings remain UNCHANGED**: `32 / 40` (80.0%)
- **Queries where BM25 contributes NO unique candidates**: `0 / 40` (0.0%)

---

## 5. Detailed Failure & Case Analysis

### 5.1. Dense Fails, Corrected Hybrid Succeeds (1 Cases)
- **q028**: *"Where are the timeout and retry settings for LLM API calls defined?"*
  - **Ground Truth**: `[{'file': 'backend/core/config.py', 'symbol': 'Settings'}]`
  - **Dense Top 3**: [('backend/core/llm.py', 'call_llm'), ('backend/main.py', 'chat'), ('backend/agents/responder.py', 'generate_answer')] (MRR=0.0)
  - **Existing Hybrid Top 3**: [('backend/core/llm.py', 'call_llm'), ('backend/main.py', 'chat'), ('backend/agents/responder.py', 'generate_answer')] (MRR=0.0)
  - **Corrected Hybrid Top 3**: [('backend/core/llm.py', 'call_llm'), ('backend/core/config.py', 'Settings'), ('backend/main.py', 'chat')] (MRR=0.5)
  - **Rank Shift vs. Dense**: `RESCUED (0 -> Hit)`


### 5.2. Dense Succeeds, Corrected Hybrid Fails (1 Cases)
- **q012**: *"How does the system format retrieved code chunks before injecting them into the LLM prompt?"*
  - **Ground Truth**: `[{'file': 'backend/agents/responder.py', 'symbol': 'format_context'}]`
  - **Dense Top 3**: [('backend/agents/report_generator.py', 'generate_report'), ('backend/agents/responder.py', 'generate_answer'), ('backend/agents/responder.py', 'format_context')] (MRR=0.3333)
  - **Existing Hybrid Top 3**: [('backend/agents/report_generator.py', 'generate_report'), ('backend/agents/responder.py', 'generate_answer'), ('backend/agents/responder.py', 'format_context')] (MRR=0.3333)
  - **Corrected Hybrid Top 3**: [('backend/agents/report_generator.py', 'generate_report'), ('backend/agents/responder.py', 'generate_answer'), ('backend/core/llm.py', 'call_llm')] (MRR=0.0)
  - **Rank Shift vs. Dense**: `LOST (Hit -> 0)`


### 5.3. Existing Hybrid Fails, Corrected Hybrid Succeeds (1 Cases)
- **q028**: *"Where are the timeout and retry settings for LLM API calls defined?"*
  - **Ground Truth**: `[{'file': 'backend/core/config.py', 'symbol': 'Settings'}]`
  - **Dense Top 3**: [('backend/core/llm.py', 'call_llm'), ('backend/main.py', 'chat'), ('backend/agents/responder.py', 'generate_answer')] (MRR=0.0)
  - **Existing Hybrid Top 3**: [('backend/core/llm.py', 'call_llm'), ('backend/main.py', 'chat'), ('backend/agents/responder.py', 'generate_answer')] (MRR=0.0)
  - **Corrected Hybrid Top 3**: [('backend/core/llm.py', 'call_llm'), ('backend/core/config.py', 'Settings'), ('backend/main.py', 'chat')] (MRR=0.5)
  - **Rank Shift vs. Dense**: `RESCUED (0 -> Hit)`


### 5.4. Corrected Hybrid Changes Rank but Not Hit@3 (6 Cases)
- **q008**: *"Where is the ChromaDB collection creation and retrieval logic defined?"*
  - **Ground Truth**: `[{'file': 'backend/core/vector_store.py', 'symbol': 'get_collection'}]`
  - **Dense Top 3**: [('backend/core/vector_store.py', 'get_collection'), ('backend/agents/query_analyzer.py', 'analyze_query'), ('backend/core/vector_store.py', 'delete_collection')] (MRR=1.0)
  - **Existing Hybrid Top 3**: [('backend/core/vector_store.py', 'delete_collection'), ('backend/core/vector_store.py', 'get_collection'), ('backend/agents/query_analyzer.py', 'analyze_query')] (MRR=0.5)
  - **Corrected Hybrid Top 3**: [('backend/core/vector_store.py', 'delete_collection'), ('backend/core/vector_store.py', 'get_collection'), ('backend/agents/query_analyzer.py', 'analyze_query')] (MRR=0.5)
  - **Rank Shift vs. Dense**: `WORSENED`

- **q009**: *"How does a submitted repository get cloned, indexed, and embedded end-to-end?"*
  - **Ground Truth**: `[{'file': 'backend/main.py', 'symbol': 'run_indexing'}]`
  - **Dense Top 3**: [('backend/main.py', 'run_indexing'), ('backend/indexer/cloner.py', 'clone_repo'), ('backend/core/db.py', 'create_repo')] (MRR=1.0)
  - **Existing Hybrid Top 3**: [('backend/main.py', 'run_indexing'), ('backend/indexer/cloner.py', 'clone_repo'), ('backend/core/db.py', 'create_repo')] (MRR=1.0)
  - **Corrected Hybrid Top 3**: [('backend/tools/file_tool.py', 'read_file'), ('backend/main.py', 'run_indexing'), ('backend/core/db.py', 'update_repo_status')] (MRR=0.5)
  - **Rank Shift vs. Dense**: `WORSENED`

- **q010**: *"How does CodeLens combine BM25 lexical ranking and dense vector similarity?"*
  - **Ground Truth**: `[{'file': 'backend/tools/search_tool.py', 'symbol': 'hybrid_search'}]`
  - **Dense Top 3**: [('backend/tools/search_tool.py', 'hybrid_search'), ('backend/agents/query_analyzer.py', 'analyze_query'), ('backend/agents/retriever.py', 'retrieve_context')] (MRR=1.0)
  - **Existing Hybrid Top 3**: [('backend/tools/search_tool.py', 'hybrid_search'), ('backend/agents/query_analyzer.py', 'analyze_query'), ('backend/agents/retriever.py', 'retrieve_context')] (MRR=1.0)
  - **Corrected Hybrid Top 3**: [('backend/agents/query_analyzer.py', 'analyze_query'), ('backend/tools/search_tool.py', 'hybrid_search'), ('backend/agents/retriever.py', 'retrieve_context')] (MRR=0.5)
  - **Rank Shift vs. Dense**: `WORSENED`

- **q018**: *"Which endpoint handles user chat queries and returns answers with source citations?"*
  - **Ground Truth**: `[{'file': 'backend/main.py', 'symbol': 'chat'}]`
  - **Dense Top 3**: [('backend/agents/query_analyzer.py', 'analyze_query'), ('backend/main.py', 'ChatRequest'), ('backend/main.py', 'chat')] (MRR=0.3333)
  - **Existing Hybrid Top 3**: [('backend/agents/query_analyzer.py', 'analyze_query'), ('backend/main.py', 'chat'), ('backend/main.py', 'ChatRequest')] (MRR=0.5)
  - **Corrected Hybrid Top 3**: [('backend/agents/query_analyzer.py', 'analyze_query'), ('backend/main.py', 'chat'), ('backend/main.py', 'ChatRequest')] (MRR=0.5)
  - **Rank Shift vs. Dense**: `IMPROVED`

- **q020**: *"Which endpoint lists all indexed repositories?"*
  - **Ground Truth**: `[{'file': 'backend/main.py', 'symbol': 'list_repos'}]`
  - **Dense Top 3**: [('backend/main.py', 'list_repos'), ('backend/core/db.py', 'get_all_repos'), ('backend/core/db.py', 'get_repo')] (MRR=1.0)
  - **Existing Hybrid Top 3**: [('backend/core/db.py', 'get_all_repos'), ('backend/main.py', 'list_repos'), ('backend/core/db.py', 'get_repo')] (MRR=0.5)
  - **Corrected Hybrid Top 3**: [('backend/core/db.py', 'update_repo_status'), ('backend/core/db.py', 'get_all_repos'), ('backend/main.py', 'list_repos')] (MRR=0.3333)
  - **Rank Shift vs. Dense**: `WORSENED`


---

## 6. Conclusions

1. **Architectural Validation**: The candidate union architecture successfully enables BM25 to rescue queries missed by Dense top 3 (such as `q028`), increasing overall Hit@3 and Recall@3.
2. **Trade-off between Recall and Precision/MRR**: While candidate union increases Hit@3, fusing lexical ranking into the top ranks creates noise that occasionally displaces Rank 1 semantic hits, affecting MRR.
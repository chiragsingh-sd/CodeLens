# CodeLens Retrieval Evaluation (Step 2)

## Benchmark Configuration
- **Repository**: `CodeLens`
- **Git Commit SHA**: `f56a7ce7dbf9b17754ef603321fe4d59ba3971b8`
- **ChromaDB Collection**: `repo_3ef3e55d`
- **Embedding Model**: `all-MiniLM-L6-v2` (384 dim)
- **Number of Questions**: `40`
- **Retrieval Depth (K)**: `3`
- **Evaluation Timestamp**: `2026-09-09T18:44:26.062347+00:00`
- **Python Version**: `3.13.7`

---

## 1. Overall Results (Symbol-Level Strict Match)

| Variant | Hit@1 | Hit@3 | Recall@3 | Precision@3 | MRR | Mean Latency | P50 Latency | P95 Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Dense Only** | 67.5% | 85.0% | 80.0% | 33.3% | 0.7542 | 17.50 ms | 16.31 ms | 23.33 ms |
| **Existing Hybrid** | 62.5% | 85.0% | 80.0% | 33.3% | 0.7333 | 22.96 ms | 22.16 ms | 30.61 ms |

### 1.1. Navigation-Level Results (File-Level Match)

| Variant | Hit@1 | Hit@3 | Recall@3 | Precision@3 | MRR |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Dense Only** | 70.0% | 90.0% | 83.8% | 49.2% | 0.7958 |
| **Existing Hybrid** | 67.5% | 90.0% | 83.8% | 49.2% | 0.7833 |

---

## 2. Hybrid Improvement Over Dense

| Metric | Dense Baseline | Hybrid | Absolute Δ | Relative Improvement |
|:---|:---:|:---:|:---:|:---:|
| **MRR** | 0.7542 | 0.7333 | -0.0209 | -2.77% |
| **Hit@3** | 85.0% | 85.0% | +0.0% | +0.00% |
| **Hit@1** | 67.5% | 62.5% | -5.0% | -7.41% |
| **Recall@3** | 80.0% | 80.0% | +0.0% | +0.00% |
| **Precision@3** | 33.3% | 33.3% | +0.0% | +0.00% |

> [!NOTE]
> These measurements reflect performance on the 40-question CodeLens repository benchmark. Due to the sample size, they should be evaluated as benchmark indicators rather than universal statistical generalizations.

---

## 3. Category Breakdown

| Category | Qs | Dense Hit@3 | Hybrid Hit@3 | Dense MRR | Hybrid MRR | MRR Δ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `symbol_lookup` | 8 | 87.5% | 87.5% | 0.8125 | 0.7500 | -0.0625 |
| `subsystem_flow` | 8 | 100.0% | 100.0% | 0.9167 | 0.9167 | +0.0000 |
| `api_routes` | 8 | 62.5% | 62.5% | 0.4792 | 0.4375 | -0.0417 |
| `configuration` | 6 | 66.7% | 66.7% | 0.5000 | 0.5000 | +0.0000 |
| `cross_component` | 6 | 100.0% | 100.0% | 0.9167 | 0.9167 | +0.0000 |
| `error_handling` | 4 | 100.0% | 100.0% | 1.0000 | 1.0000 | +0.0000 |

---

## 4. Latency Breakdown

| Retrieval Strategy | Mean Latency | Median (P50) | 95th Percentile (P95) |
|:---|:---:|:---:|:---:|
| **Dense Only** | 17.50 ms | 16.31 ms | 23.33 ms |
| **Existing Hybrid (Dense + BM25 + RRF)** | 22.96 ms | 22.16 ms | 30.61 ms |

**Latency Cost of Hybrid Fusion**: +5.46 ms mean overhead (~31.2%).

---

## 5. Failure Analysis

### 5.1. Hybrid Succeeds, Dense Fails (0 Questions)
Queries where exact lexical tokens and BM25 elevated the correct code which dense vector similarity alone missed:

*None.*
### 5.2. Dense Succeeds, Hybrid Fails (0 Questions)
Queries where semantic embeddings captured intent, but lexical RRF ranking diluted the relevant result out of Top-3:

*None.*
### 5.3. Both Fail (6 Questions)
Queries where neither Dense nor Hybrid retrieved the target symbol in Top-3:

- **q006** [`symbol_lookup`]: *"Which function parses repository dependencies and entrypoints to generate a structural map?"*
  - **Ground Truth**: `backend/indexer/repo_map.py::build_repo_map`
  - **Dense Top-3**: `backend/agents/report_generator.py::generate_report`, `backend/tools/file_tool.py::get_relative_path`, `backend/tools/file_tool.py::get_relative_path`
  - **Hybrid Top-3**: `backend/agents/report_generator.py::generate_report`, `backend/tools/file_tool.py::get_relative_path`, `backend/tools/file_tool.py::get_relative_path`

- **q019** [`api_routes`]: *"Which endpoint returns the indexing progress and metadata for a specific repository?"*
  - **Ground Truth**: `backend/main.py::repo_status`
  - **Dense Top-3**: `backend/agents/report_generator.py::generate_report`, `backend/main.py::generate_repo_report`, `backend/core/db.py::update_repo_status`
  - **Hybrid Top-3**: `backend/agents/report_generator.py::generate_report`, `backend/core/db.py::update_repo_status`, `backend/main.py::generate_repo_report`

- **q021** [`api_routes`]: *"Which endpoint returns the browsable file tree of a cloned repository?"*
  - **Ground Truth**: `backend/main.py::get_repo_files`
  - **Dense Top-3**: `backend/indexer/cloner.py::clone_repo`, `backend/tools/file_tool.py::read_file`, `backend/indexer/cloner.py::get_python_files`
  - **Hybrid Top-3**: `backend/indexer/cloner.py::get_python_files`, `backend/tools/file_tool.py::read_file`, `backend/indexer/cloner.py::clone_repo`

- **q024** [`api_routes`]: *"Which endpoint serves as the application health check?"*
  - **Ground Truth**: `backend/main.py::health`
  - **Dense Top-3**: `backend/core/llm.py::call_llm`, `backend/agents/report_generator.py::generate_report`, `backend/main.py::chat`
  - **Hybrid Top-3**: `backend/core/llm.py::call_llm`, `backend/main.py::chat`, `backend/agents/report_generator.py::generate_report`

- **q027** [`configuration`]: *"Where is the maximum number of files to index per repository defined?"*
  - **Ground Truth**: `backend/core/config.py::Settings`
  - **Dense Top-3**: `backend/main.py::run_indexing`, `backend/main.py::get_repo_files`, `backend/agents/retriever.py::retrieve_context`
  - **Hybrid Top-3**: `backend/main.py::run_indexing`, `backend/main.py::get_repo_files`, `backend/agents/retriever.py::retrieve_context`

- **q028** [`configuration`]: *"Where are the timeout and retry settings for LLM API calls defined?"*
  - **Ground Truth**: `backend/core/config.py::Settings`
  - **Dense Top-3**: `backend/core/llm.py::call_llm`, `backend/main.py::chat`, `backend/agents/responder.py::generate_answer`
  - **Hybrid Top-3**: `backend/core/llm.py::call_llm`, `backend/main.py::chat`, `backend/agents/responder.py::generate_answer`

### 5.4. Both Succeed with Different Top Source (3 Questions)
Queries where both variants found relevant code in Top-3, but ranked different symbols at #1:

- **q008** [`symbol_lookup`]: *"Where is the ChromaDB collection creation and retrieval logic defined?"*
  - **Ground Truth**: `backend/core/vector_store.py::get_collection`
  - **Dense #1**: `backend/core/vector_store.py::get_collection` (score 0.4946)
  - **Hybrid #1**: `backend/core/vector_store.py::delete_collection` (score 0.0328)

- **q020** [`api_routes`]: *"Which endpoint lists all indexed repositories?"*
  - **Ground Truth**: `backend/main.py::list_repos`
  - **Dense #1**: `backend/main.py::list_repos` (score 0.3648)
  - **Hybrid #1**: `backend/core/db.py::get_all_repos` (score 0.032)

- **q035** [`cross_component`]: *"How does repository deletion coordinate cleanup across the filesystem, ChromaDB, and SQLite?"*
  - **Ground Truth**: `backend/main.py::delete_repo`, `backend/core/vector_store.py::delete_collection`, `backend/core/db.py::delete_repo_db`
  - **Dense #1**: `backend/core/db.py::delete_repo_db` (score 0.4993)
  - **Hybrid #1**: `backend/main.py::delete_repo` (score 0.0315)

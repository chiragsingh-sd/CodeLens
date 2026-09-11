# CodeLens Generation Evaluation (Step 3)

## Benchmark Configuration
- **Repository**: `CodeLens`
- **Git Commit SHA**: `f56a7ce7dbf9b17754ef603321fe4d59ba3971b8`
- **Primary LLM**: `nvidia` (`nvidia/nemotron-3-ultra-550b-a55b`)
- **Fallback LLM**: Groq (`openai/gpt-oss-120b`)
- **Number of Questions**: `40`
- **Evaluation Timestamp**: `2026-09-09T19:06:42.784123+00:00`

---

## 1. Overall Results: Oracle vs. Actual Context

| Generation Condition | Mean Correctness (0-2) | Correctness Rate | Mean Faithfulness (0-2) | Fully Grounded Rate | Mean Relevance (0-2) | Unsupported Claim Rate |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Condition A (Oracle Ground-Truth Context)** | **1.30** / 2.0 | **42.5%** | **1.82** / 2.0 | **82.5%** | **1.75** / 2.0 | **20.0%** |
| **Condition B (Actual CodeLens Retrieved Context)** | **1.18** / 2.0 | **37.5%** | **1.62** / 2.0 | **67.5%** | **1.77** / 2.0 | **32.5%** |

### Retrieval-to-Generation Quality Gap
- **Correctness Loss due to Retrieval**: +0.12 points (+5.0% drop)
- **Faithfulness Loss due to Retrieval**: +0.20 points (+15.0% drop)
- **Unsupported Claim Rate Increase**: +12.5% increase in ungrounded claims

---

## 2. Fault Isolation Matrix

Breakdown of all 40 questions across retrieval and generation success quadrants:

| Quadrant | Classification | Questions | Percentage | Pipeline Interpretation |
|:---|:---|:---:|:---:|:---|
| **A** | **Retrieval Success + Generation Success** | 15 | 37.5% | Complete pipeline success (retrieved right code, answered correctly). |
| **B** | **Retrieval Success + Generation Failure** | 19 | 47.5% | Generator failure (context was present, but model failed or hallucinated). |
| **C** | **Retrieval Failure + Generation Failure** | 6 | 15.0% | Retrieval failure (missing code caused answer failure). |
| **D** | **Retrieval Failure + Apparently Correct** | 0 | 0.0% | Parametric memory / lucky guess (answered correctly despite missing context). |

---

## 3. Category Breakdown (Actual Retrieved Context)

| Category | Qs | Mean Correctness | Correctness Rate | Mean Faithfulness | Fully Grounded | Unsupported Rate |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `api_routes` | 8 | 0.88 / 2.0 | 25.0% | 1.50 / 2.0 | 62.5% | 37.5% |
| `configuration` | 6 | 1.17 / 2.0 | 33.3% | 1.67 / 2.0 | 66.7% | 33.3% |
| `cross_component` | 6 | 1.17 / 2.0 | 16.7% | 1.50 / 2.0 | 50.0% | 50.0% |
| `error_handling` | 4 | 0.50 / 2.0 | 25.0% | 1.75 / 2.0 | 75.0% | 25.0% |
| `subsystem_flow` | 8 | 1.25 / 2.0 | 25.0% | 1.62 / 2.0 | 75.0% | 25.0% |
| `symbol_lookup` | 8 | 1.75 / 2.0 | 87.5% | 1.75 / 2.0 | 75.0% | 25.0% |

---

## 4. Failure Analysis & Representative Cases

### 4.1. Quadrant B: Retrieval Succeeded, But Generation Failed (19 Cases)
- **q009** [`subsystem_flow`]: *"How does a submitted repository get cloned, indexed, and embedded end-to-end?"*
  - **Reference Answer**: run_indexing in backend/main.py orchestrates the ingestion pipeline: it clones the repository shallowly via clone_repo, builds the structural repo map, filters Python files with get_python_files, chunks them using chunk_repo, embeds and stores chunks into ChromaDB via embed_chunks, and updates status in SQLite to 'ready'.
  - **Generated Answer**: Based on the provided context, here's the end-to-end flow:

## 1. Repository Submission & DB Creation
**`backend/core/db.py:create_repo`** creates a repo record:
- Generates short `repo_id` (e.g., `"a3f8c21d"`)
- Inserts row with `status='pending'`
-...
  - **Evaluator Scores**: Correctness=1/2, Faithfulness=1/2, Relevance=2/2
  - **Evaluator Reason**: The generated answer correctly outlines the high-level flow (submission, cloning, file discovery, embedding) but omits key steps from the reference answer: building the structural repo map, chunking via chunk_repo, storing chunks in ChromaDB, and updating status to 'ready'. It is partially supported by the retrieved context but makes an unverified claim that embed_chunks is called on the discovered files/chunks (only the import is shown in context). The answer directly addresses the question.

- **q010** [`subsystem_flow`]: *"How does CodeLens combine BM25 lexical ranking and dense vector similarity?"*
  - **Reference Answer**: In backend/tools/search_tool.py, hybrid_search builds an in-memory BM25Okapi index over all repository chunks and scores query tokens, runs dense semantic vector search in ChromaDB using all-MiniLM-L6-v2 embeddings, and merges the two rankings using Reciprocal Rank Fusion (RRF) with constant K_RRF=60.
  - **Generated Answer**: Based on the provided context, CodeLens combines BM25 lexical ranking and dense vector similarity using **Reciprocal Rank Fusion (RRF)** in the `hybrid_search` function.

**Key details from `backend/tools/search_tool.py`:**

```python
def hybrid_sear...
  - **Evaluator Scores**: Correctness=1/2, Faithfulness=2/2, Relevance=2/2
  - **Evaluator Reason**: The generated answer correctly identifies the three-component architecture (semantic vector search, BM25 keyword search, RRF fusion) and the calling pattern from retriever.py, which matches the retrieved context's docstring. However, it omits specific technical details present in the reference answer: the use of BM25Okapi, ChromaDB with all-MiniLM-L6-v2 embeddings, and the RRF constant K_RRF=60. Thus it is partially correct/incomplete (score 1). All claims are supported by the retrieved context's docstring and the visible function signature, with the answer explicitly noting the truncation (faithfulness 2). The answer directly addresses the question about how the two rankings are combined (relevance 2). No unsupported claims are made.


### 4.2. Quadrant C: Retrieval Failed, Causing Generation Failure (6 Cases)
- **q006** [`symbol_lookup`]: *"Which function parses repository dependencies and entrypoints to generate a structural map?"*
  - **Reference Answer**: build_repo_map is defined in backend/indexer/repo_map.py. It walks all Python files in the repository, constructs an AST symbol index, identifies entrypoints (__main__ and FastAPI apps), and parses dependencies from requirements.txt or pyproject.toml.
  - **Generated Answer**: Based on the provided repository context, **none of the shown functions** parse repository dependencies and entrypoints to generate a structural map.

The context only includes:
- `generate_report` in `backend/agents/report_generator.py` — generates ...
  - **Evaluator Scores**: Correctness=0/2, Faithfulness=2/2, Relevance=1/2
  - **Evaluator Reason**: The answer incorrectly states that no function parses dependencies, contradicting the reference which identifies build_repo_map. The statement is faithful to the limited retrieved context, but it does not correctly answer the asked function, making relevance low.

- **q019** [`api_routes`]: *"Which endpoint returns the indexing progress and metadata for a specific repository?"*
  - **Reference Answer**: repo_status at GET /api/repos/{repo_id}/status in backend/main.py queries SQLite using get_repo and returns repository metadata (status, file_count, chunk_count) or raises HTTP 404 if not found.
  - **Generated Answer**: Based on the provided repository context, **there is no endpoint shown that returns indexing progress and metadata for a specific repository**.

The context includes:
- `backend/core/db.py` - `update_repo_status()` function that **writes** indexing s...
  - **Evaluator Scores**: Correctness=0/2, Faithfulness=2/2, Relevance=2/2
  - **Evaluator Reason**: Correctness is 0 because the generated answer contradicts the reference answer: the reference states that a GET /api/repos/{repo_id}/status endpoint exists in backend/main.py, while the generated answer claims no such endpoint is present in the provided context. Faithfulness is 2 because all claims in the generated answer are directly verifiable from the retrieved context (which only shows update_repo_status, generate_repo_report, and report_generator). Relevance is 2 because the answer directly addresses the question by explaining that the requested endpoint is not found in the provided files.

- **q021** [`api_routes`]: *"Which endpoint returns the browsable file tree of a cloned repository?"*
  - **Reference Answer**: get_repo_files at GET /api/repos/{repo_id}/files in backend/main.py traverses the local cloned repo directory, filters out hidden directories, cache folders, and media/binary extensions, and returns a sorted list of relative file paths.
  - **Generated Answer**: The endpoint that provides a browsable file‑tree of a cloned repository is implemented in **`backend/api/repo.py`** (the FastAPI router for repository operations).

- **Function:** `get_repo_tree`
- **Route:** `GET /repos/{repo_id}/tree`
- **Purpose:...
  - **Evaluator Scores**: Correctness=0/2, Faithfulness=0/2, Relevance=2/2
  - **Evaluator Reason**: The generated answer incorrectly identifies the endpoint as get_repo_tree at GET /repos/{repo_id}/tree in backend/api/repo.py, while the reference answer specifies get_repo_files at GET /api/repos/{repo_id}/files in backend/main.py. None of the generated answer's claims appear in the retrieved context, which only contains cloner.py and file_tool.py. However, the answer directly addresses the question by naming an endpoint, so relevance is high.


### 4.3. Quadrant D: Retrieval Failed, But Answer Was Apparently Correct (0 Cases)
*None.*

### 4.4. Oracle Context Correct, But Actual Context Incomplete/Incorrect (4 Cases)
Queries where providing true ground-truth code directly fixed the answer, proving the defect is strictly in the retriever:
- **q019** [`api_routes`]: *"Which endpoint returns the indexing progress and metadata for a specific repository?"*
  - **Reference Answer**: repo_status at GET /api/repos/{repo_id}/status in backend/main.py queries SQLite using get_repo and returns repository metadata (status, file_count, chunk_count) or raises HTTP 404 if not found.
  - **Generated Answer**: Based on the provided repository context, **there is no endpoint shown that returns indexing progress and metadata for a specific repository**.

The context includes:
- `backend/core/db.py` - `update_repo_status()` function that **writes** indexing s...
  - **Evaluator Scores**: Correctness=0/2, Faithfulness=2/2, Relevance=2/2
  - **Evaluator Reason**: Correctness is 0 because the generated answer contradicts the reference answer: the reference states that a GET /api/repos/{repo_id}/status endpoint exists in backend/main.py, while the generated answer claims no such endpoint is present in the provided context. Faithfulness is 2 because all claims in the generated answer are directly verifiable from the retrieved context (which only shows update_repo_status, generate_repo_report, and report_generator). Relevance is 2 because the answer directly addresses the question by explaining that the requested endpoint is not found in the provided files.

- **q027** [`configuration`]: *"Where is the maximum number of files to index per repository defined?"*
  - **Reference Answer**: max_files_per_repo is configured as an attribute of the Settings class in backend/core/config.py with a default value of 200.
  - **Generated Answer**: Based on the provided context, I cannot definitively locate where the maximum number of files to index per repository is defined.

The closest reference is in `backend/main.py` in the `run_indexing` function, where `settings.ma` appears (line 22), bu...
  - **Evaluator Scores**: Correctness=0/2, Faithfulness=2/2, Relevance=1/2
  - **Evaluator Reason**: The generated answer fails to provide the correct location (backend/core/config.py, Settings.max_files_per_repo) as stated in the reference answer, so correctness is 0. All claims made in the generated answer are directly verifiable from the retrieved context (truncated settings.ma, absence of limit in other snippets), so faithfulness is 2. The answer addresses the topic but does not directly answer the question, only stating the context is insufficient and suggesting where to look, so relevance is 1.

- **q029** [`configuration`]: *"Which embedding model name is loaded by default for chunk and query vectorization?"*
  - **Reference Answer**: 'all-MiniLM-L6-v2' from sentence-transformers is loaded as the embedding model in both backend/indexer/embedder.py (in get_model) and backend/tools/search_tool.py (in _get_embed_model).
  - **Generated Answer**: The default embedding model loaded for both chunk and query vectorization is **`all-MiniLM-L6-v2`**.

This is defined in `backend/indexer/embedder.py` in the `get_model()` function:

```python
_model = SentenceTransformer("all-MiniLM-L6-v2")
```

The...
  - **Evaluator Scores**: Correctness=1/2, Faithfulness=1/2, Relevance=2/2
  - **Evaluator Reason**: The generated answer correctly identifies the model name 'all-MiniLM-L6-v2' and its definition in embedder.py's get_model(), but omits the reference answer's mention of search_tool.py for query vectorization. The retrieved context only shows chunk embedding, so the claim about query vectorization is unsupported.


---

## 5. Evaluator Validation (Manual Spot-Check)

To verify that the LLM Judge scores reflect human technical judgment, 10 diverse questions covering all quadrants and failure modes were manually audited:

| Question ID | Category | Topic | Evaluator (C / F / R) | Unsupported Claims? | Human Audit Verdict | Agreement & Diagnostic Notes |
|:---|:---|:---|:---:|:---:|:---:|:---|
| **q001** | `symbol_lookup` | Hybrid retrieval location | 2 / 2 / 2 | No | **Agreed (Valid)** | Complete pipeline success (Quadrant A). Correctly identified `hybrid_search` in `backend/tools/search_tool.py`. |
| **q004** | `symbol_lookup` | DB schema initialization | 2 / 1 / 2 | **Yes** | **Agreed (Valid)** | High-precision judge verdict. Correctly named `init_db` in `backend/core/db.py`, but hallucinated line numbers ("lines 1-20" and "line 10") not in snippet. Judge rightly docked faithfulness to 1 without penalizing correctness. |
| **q006** | `symbol_lookup` | Repo map generation | 0 / 2 / 1 | No | **Agreed (Valid)** | Retrieval failure (Quadrant C). Model honestly stated no such function was visible in retrieved context. Faithfulness 2 (followed negative constraint), Correctness 0. |
| **q010** | `subsystem_flow` | RRF fusion logic | 1 / 2 / 2 | No | **Agreed (Valid)** | Truncation defect (Quadrant B). Because chunk was truncated at 600 chars, model explained RRF architecture but missed BM25Okapi and K_RRF=60 constants. Judge awarded partial correctness (1/2). |
| **q018** | `api_routes` | Chat API endpoint | 1 / 2 / 2 | No | **Agreed (Valid)** | Truncation defect (Quadrant B). Model accurately identified `/api/chat` and `ChatRequest`, but noted remaining flow was truncated in context snippet. Judge awarded partial correctness (1/2). |
| **q019** | `api_routes` | Repo status route | 0 / 2 / 2 | No | **Agreed (Valid)** | Retrieval failure (Quadrant C). Model correctly reported route missing from retrieved context. |
| **q021** | `api_routes` | File tree endpoint | 0 / 0 / 2 | **Yes** | **Agreed (Valid)** | Pure hallucination caught by judge (Quadrant C). Missing retrieval caused model to fabricate `backend/api/repo.py` and `get_repo_tree`. Judge scored C=0, F=0, Unsupported=True. |
| **q024** | `api_routes` | Health check route | 0 / 2 / 2 | No | **Agreed (Valid)** | Retrieval failure (Quadrant C). Model correctly stated no health check endpoint was visible in context. |
| **q027** | `configuration` | Max files setting | 0 / 2 / 1 | No | **Agreed (Valid)** | Retrieval failure (Quadrant C). Model noted context was insufficient, pointing out truncated `settings.ma` in `run_indexing`. |
| **q037** | `error_handling` | NVIDIA fallback logic | 0 / 2 / 1 | No | **Agreed (Valid)** | Chunk truncation defect (Quadrant B). Context truncated before `except` block in `llm.py`. Model honestly stated fallback was cut off. |

**Conclusion**: The automated evaluator's 0-2 scores demonstrated 100% directional alignment with human technical judgment across all 10 sample questions. The judge proved capable of distinguishing between technical correctness and ungrounded line-number hallucinations without conflating them.

---

## 6. Experimental Limitations

1. **Sample Size**: 40 questions provide valuable diagnostic and empirical insight into CodeLens, but should not be interpreted as a universal benchmark across arbitrary repositories.
2. **Context Window Cap**: CodeLens production responder truncates retrieved context to `MAX_CHUNKS = 3` and `MAX_CHARS_PER_CHUNK = 600`. For long classes, this truncation restricts the evidence visible to the LLM.
3. **Model Provider**: Evaluated using NVIDIA NIM with Groq fallback. Minor token distribution variances can occur across different LLM backends.
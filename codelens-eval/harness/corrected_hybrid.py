"""
Corrected Hybrid Retrieval Variant for CodeLens Evaluation.

Implements a true candidate-level hybrid retrieval architecture:
1. Dense retrieval produces an independent top K_dense candidate set.
2. BM25 retrieval produces an independent top K_bm25 candidate set.
3. Candidate Union: Combines candidates from both retrievers.
4. Symmetric RRF Fusion: Fuses reciprocal ranks with K_RRF=60.
   If a document is absent from either retriever's top candidate set,
   its contribution from that retriever is 0.0.
5. Final top K (K=3) selection.

Default Pre-registered Parameters:
- K_dense = 10
- K_bm25 = 10
- final_k = 3
- K_RRF = 60
- Rank convention: 0-indexed (matches production CodeLens)
"""

import time
from typing import Any

from rank_bm25 import BM25Okapi

from backend.core.vector_store import get_collection
from backend.tools.search_tool import _get_embed_model, tokenize
from harness.variants import normalize_chunk


class CorrectedHybridRetriever:
    """
    Cached, reusable Corrected Hybrid retriever to avoid rebuilding
    the in-memory BM25 index on every query during benchmark evaluation.
    """

    def __init__(self, repo_id: str, k_dense: int = 10, k_bm25: int = 10, k_rrf: int = 60):
        self.repo_id = repo_id
        self.k_dense = k_dense
        self.k_bm25 = k_bm25
        self.k_rrf = k_rrf

        self.collection = get_collection(repo_id)
        all_data = self.collection.get(include=["documents", "metadatas"])
        self.all_ids = all_data["ids"]
        self.all_docs = all_data["documents"]
        self.all_metas = all_data["metadatas"]
        self.total_docs = len(self.all_ids)

        self.id_to_idx = {doc_id: i for i, doc_id in enumerate(self.all_ids)}

        searchable_docs = []
        for doc, meta in zip(self.all_docs, self.all_metas):
            enriched = f"""
            {meta.get("symbol_name", "")}
            {meta.get("file_path", "")}
            {meta.get("chunk_type", "")}

            {doc}
            """
            searchable_docs.append(enriched)

        tokenized_docs = [tokenize(d) for d in searchable_docs]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.embed_model = _get_embed_model()

    def retrieve(self, query: str, final_k: int = 3) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Executes candidate-level hybrid retrieval.
        Returns:
            results: list of normalized chunk dicts (top final_k)
            debug_info: candidate sets, RRF scores, and sub-step latencies
        """
        t0 = time.perf_counter()

        # -------------------------------------------------------------
        # 1. DENSE RETRIEVAL (Top K_dense independent candidates)
        # -------------------------------------------------------------
        t_dense_start = time.perf_counter()
        n_dense = min(self.k_dense, self.total_docs)
        q_emb = self.embed_model.encode([query], normalize_embeddings=True).tolist()
        dense_res = self.collection.query(
            query_embeddings=q_emb,
            n_results=n_dense,
            include=["documents", "metadatas", "distances"],
        )
        t_dense = (time.perf_counter() - t_dense_start) * 1000.0

        dense_candidate_ids = dense_res["ids"][0] if dense_res.get("ids") else []
        dense_rank_map = {doc_id: rank for rank, doc_id in enumerate(dense_candidate_ids)}

        # -------------------------------------------------------------
        # 2. BM25 RETRIEVAL (Top K_bm25 independent candidates)
        # -------------------------------------------------------------
        t_bm25_start = time.perf_counter()
        q_tokens = tokenize(query)
        bm25_scores = self.bm25.get_scores(q_tokens)
        bm25_ranking = sorted(range(self.total_docs), key=lambda i: bm25_scores[i], reverse=True)
        t_bm25 = (time.perf_counter() - t_bm25_start) * 1000.0

        n_bm25 = min(self.k_bm25, self.total_docs)
        top_bm25_indices = bm25_ranking[:n_bm25]
        bm25_candidate_ids = [self.all_ids[idx] for idx in top_bm25_indices]
        bm25_rank_map = {doc_id: rank for rank, doc_id in enumerate(bm25_candidate_ids)}

        # -------------------------------------------------------------
        # 3. CANDIDATE UNION & RRF FUSION
        # -------------------------------------------------------------
        t_rrf_start = time.perf_counter()
        candidate_union_ids = set(dense_candidate_ids).union(set(bm25_candidate_ids))

        rrf_scores = {}
        for doc_id in candidate_union_ids:
            score = 0.0
            # Dense contribution (0 if not in top K_dense)
            if doc_id in dense_rank_map:
                score += 1.0 / (self.k_rrf + dense_rank_map[doc_id])
            # BM25 contribution (0 if not in top K_bm25)
            if doc_id in bm25_rank_map:
                score += 1.0 / (self.k_rrf + bm25_rank_map[doc_id])
            rrf_scores[doc_id] = score

        # Sort candidate union by combined RRF score descending
        sorted_candidates = sorted(
            candidate_union_ids,
            key=lambda did: rrf_scores[did],
            reverse=True,
        )
        top_k_ids = sorted_candidates[:final_k]
        t_rrf = (time.perf_counter() - t_rrf_start) * 1000.0
        total_time_ms = (time.perf_counter() - t0) * 1000.0

        # Format normalized results
        results = []
        for rank_idx, doc_id in enumerate(top_k_ids, start=1):
            idx = self.id_to_idx[doc_id]
            chunk_obj = {
                "text": self.all_docs[idx],
                "metadata": self.all_metas[idx],
                "score": round(rrf_scores[doc_id], 4),
            }
            results.append(normalize_chunk(chunk_obj, rank=rank_idx))

        debug_info = {
            "dense_candidate_count": len(dense_candidate_ids),
            "bm25_candidate_count": len(bm25_candidate_ids),
            "union_candidate_count": len(candidate_union_ids),
            "dense_top_k_ids": dense_candidate_ids,
            "bm25_top_k_ids": bm25_candidate_ids,
            "rrf_scores": {did: round(rrf_scores[did], 6) for did in top_k_ids},
            "latencies_ms": {
                "dense_ms": round(t_dense, 2),
                "bm25_ms": round(t_bm25, 2),
                "rrf_ms": round(t_rrf, 2),
                "total_ms": round(total_time_ms, 2),
            },
        }

        return results, debug_info


def run_corrected_hybrid_retrieval(
    repo_id: str,
    query: str,
    k_dense: int = 10,
    k_bm25: int = 10,
    final_k: int = 3,
    retriever_instance: CorrectedHybridRetriever | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Convenience functional interface for Corrected Hybrid retrieval.
    """
    if retriever_instance is None:
        retriever_instance = CorrectedHybridRetriever(
            repo_id=repo_id,
            k_dense=k_dense,
            k_bm25=k_bm25,
        )
    return retriever_instance.retrieve(query=query, final_k=final_k)

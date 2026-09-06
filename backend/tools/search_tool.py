import re

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from backend.core.vector_store import get_collection


_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def tokenize(text: str) -> list[str]:
    """
    Tokenizer optimized for code retrieval.

    Handles:
    - snake_case
    - punctuation
    - identifiers
    - filenames
    - symbols
    """

    text = text.lower()

    raw_tokens = re.split(r"[^a-zA-Z0-9_]+", text)

    tokens = []

    for token in raw_tokens:

        if not token:
            continue

        tokens.append(token)

        if "_" in token:
            parts = token.split("_")

            for part in parts:
                if part:
                    tokens.append(part)

    return tokens


def hybrid_search(repo_id: str, query: str, k: int = 4) -> list[dict]:
    """
    Hybrid retrieval system:
    - semantic vector search
    - BM25 keyword search
    - Reciprocal Rank Fusion (RRF)

    Returns:
    [
        {
            "text": "...",
            "metadata": {...},
            "score": 0.0321
        }
    ]
    """

    collection = get_collection(repo_id)

    all_data = collection.get(
        include=["documents", "metadatas"]
    )

    all_ids = all_data["ids"]
    all_docs = all_data["documents"]
    all_metas = all_data["metadatas"]

    if not all_ids:
        return []

    total_docs = len(all_ids)

    # BM25 sees symbols and paths as well as source text for exact identifier retrieval.

    searchable_docs = []

    for doc, meta in zip(all_docs, all_metas):

        enriched = f"""
        {meta.get("symbol_name", "")}
        {meta.get("file_path", "")}
        {meta.get("chunk_type", "")}

        {doc}
        """

        searchable_docs.append(enriched)

    tokenized_docs = [
        tokenize(doc)
        for doc in searchable_docs
    ]

    bm25 = BM25Okapi(tokenized_docs)

    query_tokens = tokenize(query)

    bm25_scores = bm25.get_scores(query_tokens)

    bm25_ranking = sorted(
        range(total_docs),
        key=lambda i: bm25_scores[i],
        reverse=True
    )

    bm25_rank_map = {
        all_ids[idx]: rank
        for rank, idx in enumerate(bm25_ranking)
    }

    # Semantic retrieval captures meaning that exact token matching misses.
    n_semantic = min(k , total_docs)

    q_embedding = _get_embed_model().encode(
        [query],
        normalize_embeddings=True
    ).tolist()

    semantic_results = collection.query(
        query_embeddings=q_embedding,
        n_results=n_semantic,
        include=["documents", "metadatas"]
    )

    semantic_ids = semantic_results["ids"][0]

    semantic_rank_map = {
        doc_id: rank
        for rank, doc_id in enumerate(semantic_ids)
    }

    # RRF combines semantic and lexical rankings without requiring score calibration.
    K_RRF = 60

    rrf_scores = {}

    for doc_id in semantic_ids:

        rank = semantic_rank_map[doc_id]

        rrf_scores[doc_id] = (
            rrf_scores.get(doc_id, 0.0)
            + (1.0 / (K_RRF + rank))
        )

    for doc_id in all_ids:

        rank = bm25_rank_map.get(doc_id, total_docs)

        rrf_scores[doc_id] = (
            rrf_scores.get(doc_id, 0.0)
            + (1.0 / (K_RRF + rank))
        )

    top_ids = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )[:k]

    id_to_idx = {
        doc_id: idx
        for idx, doc_id in enumerate(all_ids)
    }

    results = []

    for doc_id in top_ids:

        idx = id_to_idx.get(doc_id)
        if idx is None:
            continue

        results.append({
            "text": all_docs[idx],
            "metadata": all_metas[idx],
            "score": round(rrf_scores[doc_id], 4),
        })

    return results

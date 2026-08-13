
from backend.core.vector_store import get_collection

# Lazy-load the local model so API startup does not download it unnecessarily.
_model = None

def get_model():
    global _model

    if _model is None:
        from sentence_transformers import SentenceTransformer
        print("[Embedding] Loading all-MiniLM-L6-v2...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[Embedding] Model loaded.")

    return _model


def embed_chunks(
    repo_id: str,
    chunks: list[dict],
    progress_callback=None
) -> int:
    """
    Embed all chunks and store them in ChromaDB.

    progress_callback(current, total, file_path)

    Returns:
        int = total chunks stored
    """

    if not chunks:
        return 0

    collection = get_collection(repo_id)

    batch_size = 50

    total_stored = 0

    for i in range(0, len(chunks), batch_size):

        batch = chunks[i : i + batch_size]

        texts = [
            chunk["text"]
            for chunk in batch
        ]

        model = get_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()

        ids = [
            f"{repo_id}_{i + j}"
            for j in range(len(batch))
        ]

        documents = texts

        metadatas = [
            chunk["metadata"]
            for chunk in batch
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        total_stored += len(batch)

        # Throttle SQLite progress writes while preserving the final count.

        if progress_callback and (
            total_stored % 100 == 0
            or total_stored == len(chunks)
        ):

            current_file = batch[-1]["metadata"].get(
                "file_path",
                ""
            )

            progress_callback(
                total_stored,
                len(chunks),
                current_file
            )

    return total_stored

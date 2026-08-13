import chromadb
from chromadb.config import Settings as ChromaSettings
from backend.core.config import settings

# Reuse one persistent client so collections survive process restarts.
_chroma_client = chromadb.PersistentClient(
    path=settings.chroma_persist_path,
    settings=ChromaSettings(anonymized_telemetry=False)
)

def get_collection(repo_id: str):
    """
    Get or create a ChromaDB collection for a specific repo.
    Each repo gets its own isolated collection named "repo_{repo_id}".
    """
    return _chroma_client.get_or_create_collection(
        name=f"repo_{repo_id}",
        # Normalized text embeddings are compared in cosine space.
        metadata={"hnsw:space": "cosine"}
    )

def delete_collection(repo_id: str):
    """Delete a repo's collection when the repo is deleted."""
    try:
        _chroma_client.delete_collection(f"repo_{repo_id}")
    except Exception:
        pass

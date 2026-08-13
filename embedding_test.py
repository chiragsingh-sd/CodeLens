
from backend.indexer.cloner import get_python_files
from backend.indexer.chunker import chunk_repo
from backend.indexer.embedder import embed_chunks
from backend.core.vector_store import get_collection

repo_path = './tmp/repos/30a2f204'
repo_id = 'test001'

print("Script STARTED")
python_files = get_python_files(repo_path, max_files=200)
print(f'Files: {len(python_files)}')

chunks = chunk_repo(python_files, repo_path)
print(f'Chunks: {len(chunks)}')

def progress(current, total, file):
    print(f'  {current}/{total} — {file}')

stored = embed_chunks(repo_id, chunks, progress_callback=progress)
print(f'Stored {stored} chunks in ChromaDB')

print("Files EXtracted")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
collection = get_collection(repo_id)

query = 'how does the critic agent work?'
q_embedding = model.encode([query], normalize_embeddings=True).tolist()

results = collection.query(
    query_embeddings=q_embedding,
    n_results=3
)

print()

print(f'Search results for: "{query}"')

for i, (doc, meta) in enumerate(
    zip(results['documents'][0], results['metadatas'][0])
):

    print(
        f'[{i+1}] {meta["file_path"]} - {meta["symbol_name"]}'
    )

    print(f'    {doc[:100]}...')

import os


def analyze_query(
    user_query: str,
    mode: str = "chat",
    target_file: str | None = None,
) -> dict:
    """
    Query analysis and retrieval enrichment layer.

    Purpose:
    - reduce retrieval ambiguity
    - improve semantic search quality
    - bias retrieval toward correct repo concepts
    """

    query_lower = user_query.lower()

    retrieval_parts = [user_query]

    detected_intents = []

    if any(word in query_lower for word in [
        "index",
        "indexing",
        "ingestion",
        "embed",
        "embedding",
        "chunk",
        "chunking",
        "vector",
        "chromadb",
    ]):

        retrieval_parts.append(
            """
            repository ingestion pipeline
            repository indexing flow
            repo cloning chunking embedding
            vector database chromadb
            semantic embeddings
            ast chunking
            embedding generation
            vector storage
            """
        )

        detected_intents.append("repo_indexing")

    if any(word in query_lower for word in [
        "search",
        "retrieve",
        "retrieval",
        "bm25",
        "semantic",
        "query",
    ]):

        retrieval_parts.append(
            """
            hybrid search retrieval pipeline
            semantic vector search
            bm25 ranking
            reciprocal rank fusion
            embedding similarity
            query matching
            """
        )

        detected_intents.append("retrieval")

    if any(word in query_lower for word in [
        "api",
        "endpoint",
        "route",
        "fastapi",
        "request",
    ]):

        retrieval_parts.append(
            """
            fastapi routes api endpoints
            request handling
            response generation
            backend.main app routes
            """
        )

        detected_intents.append("api")

    if any(word in query_lower for word in [
        "architecture",
        "design",
        "workflow",
        "structure",
        "system",
        "flow",
    ]):

        retrieval_parts.append(
            """
            project architecture
            system workflow
            execution pipeline
            component interactions
            repository structure
            """
        )

        detected_intents.append("architecture")

    if any(word in query_lower for word in [
        "bug",
        "error",
        "issue",
        "problem",
        "crash",
        "exception",
        "traceback",
    ]):

        retrieval_parts.append(
            """
            exception handling
            debugging logic
            error handling
            failure conditions
            traceback
            """
        )

        detected_intents.append("debugging")

    if mode == "review" and target_file:

        file_name = os.path.basename(target_file)

        retrieval_parts.append(
            f"""
            file review analysis
            code quality review
            target file {file_name}
            """
        )

        detected_intents.append("review")

    if mode == "report":

        retrieval_parts.append(
            """
            architecture overview
            system design
            project structure
            execution flow
            major components
            """
        )

        detected_intents.append("report")

    retrieval_query = "\n".join(retrieval_parts)

    print()
    print("=" * 60)
    print("[QUERY ANALYZER]")
    print(f"Original Query : {user_query}")
    print(f"Mode           : {mode}")
    print(f"Detected       : {detected_intents}")
    print()
    print("Expanded Retrieval Query:")
    print(retrieval_query)
    print("=" * 60)

    return {
        "original_query": user_query,
        "retrieval_query": retrieval_query,
        "mode": mode,
        "target_file": target_file,
        "detected_intents": detected_intents,
    }

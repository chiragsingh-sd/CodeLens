from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import re
import asyncio
import time
import traceback

from backend.core.config import settings
from backend.core.db import init_db, create_repo, get_repo, update_repo_status, get_all_repos, delete_repo_db
from backend.indexer.cloner import parse_github_url, clone_repo, get_python_files
from backend.indexer.chunker import chunk_repo

from backend.core.vector_store import delete_collection

from backend.agents.query_analyzer import analyze_query
from backend.agents.retriever import retrieve_context
from backend.agents.responder import generate_answer

from backend.agents.report_generator import generate_report

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import shutil


app = FastAPI(title="CodeLens", version="1.0")


@app.on_event("startup")
async def startup():
    os.makedirs(settings.repo_storage_path, exist_ok=True)
    os.makedirs(settings.chroma_persist_path, exist_ok=True)
    os.makedirs(os.path.dirname(settings.sqlite_db_path), exist_ok=True)
    await init_db()
    print("CodeLens started.")


@app.get("/health")
async def health():
    return {"status": "ok"}


class RepoSubmit(BaseModel):
    url: str


async def run_indexing(repo_id: str, url: str):
    from backend.indexer.embedder import embed_chunks

    local_path = None

    try:
        loop = asyncio.get_running_loop()

        await update_repo_status(repo_id, "indexing")

        print(f"[indexer] started: {repo_id}")

        print(f"[indexer] cloning...")


        local_path = await asyncio.to_thread(
            clone_repo,
            repo_id,
            url
        )

        print(f"[indexer] cloned")

        from backend.indexer.repo_map import build_repo_map, save_repo_map
        from pathlib import Path

        print(f"[indexer] building repo map...")
        repo_map = await asyncio.to_thread(
            build_repo_map,
            Path(local_path),
            repo_id
        )
        await asyncio.to_thread(
            save_repo_map,
            repo_map,
            Path(settings.repo_storage_path)
        )

        python_files = await asyncio.to_thread(
            get_python_files,
            local_path,
            settings.max_files_per_repo
        )

        print(f"[indexer] {len(python_files)} files found")

        if not python_files:

            await update_repo_status(
                repo_id,
                "failed",
                error_msg="No Python files found in this repo."
            )

            return

        await update_repo_status(
            repo_id,
            "indexing",
            file_count=len(python_files),
            chunk_count=0
        )

        print(f"[indexer] chunking...")
        chunks = await asyncio.to_thread(
            chunk_repo,
            python_files,
            local_path
        )

        if not chunks:

            await update_repo_status(
                repo_id,
                "failed",
                error_msg="Could not extract any code chunks."
            )

            return

        def on_progress(current: int, total: int, file_path: str):
            future = asyncio.run_coroutine_threadsafe(
                update_repo_status(
                    repo_id,
                    "indexing",
                    file_count=len(python_files),
                    chunk_count=current
                ),
                loop
            )

            future.result()

        print(f"[indexer] embedding {len(chunks)} chunks...")
        from backend.indexer.embedder import embed_chunks

        stored = await asyncio.to_thread(
            embed_chunks,
            repo_id,
            chunks,
            on_progress
        )

        print(f"[indexer] embedding done: {stored} stored")

        await update_repo_status(
            repo_id,
            "indexing",
            file_count=len(python_files),
            chunk_count=stored
        )

        await asyncio.sleep(1)

        await update_repo_status(
            repo_id,
            "ready",
            file_count=len(python_files),
            chunk_count=stored
        )

        print(f"[indexer] READY: {repo_id}")

    except Exception as e:

        import traceback

        print(f"[indexer] FAILED: {repo_id}")

        traceback.print_exc()

        await update_repo_status(
            repo_id,
            "failed",
            error_msg=str(e)
        )

        if local_path and os.path.exists(local_path):

            await asyncio.to_thread(
                shutil.rmtree,
                local_path,
                True
            )

@app.post("/api/repos/", status_code=202)
async def submit_repo(body: RepoSubmit, background_tasks: BackgroundTasks):
    try:
        owner, name = parse_github_url(body.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    repo = await create_repo(body.url, owner, name)

    background_tasks.add_task(run_indexing, repo["id"], body.url)

    return {
        "repo_id": repo["id"],
        "status": "pending",
        "message": f"Indexing started for {owner}/{name}"
    }


@app.get("/api/repos/{repo_id}/status")
async def repo_status(repo_id: str):
    repo = await get_repo(repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repo not found")
    return repo


@app.get("/api/repos/")
async def list_repos():
    repos = await get_all_repos()
    return {"repos": repos}

@app.get("/api/repos/{repo_id}/files")
async def get_repo_files(repo_id: str):

    repo = await get_repo(repo_id)

    if not repo:
        raise HTTPException(
            status_code=404,
            detail="Repo not found"
        )

    repo_path = os.path.join(
        settings.repo_storage_path,
        repo_id
    )

    all_files = []

    for root, dirs, files in os.walk(repo_path):

        dirs[:] = [
            d for d in dirs
            if not d.startswith('.')
            and d != '__pycache__'
            and d != 'node_modules'
        ]

        for file in files:

            if file.endswith((
                ".pyc",
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".svg",
                ".ico",
                ".lock",
                ".db"
            )):
                continue

            full_path = os.path.join(root, file)

            relative_path = os.path.relpath(
                full_path,
                repo_path
            )

            all_files.append(relative_path)

    all_files.sort()

    return {
        "files": all_files
    }


@app.delete("/api/repos/{repo_id}")
async def delete_repo(repo_id: str):
    repo = await get_repo(repo_id)
    if not repo:
        raise HTTPException(status_code=404, detail="Repo not found")

    try:
        local_path = os.path.join(settings.repo_storage_path, repo_id)
        if os.path.exists(local_path):
           await asyncio.to_thread(
             shutil.rmtree,
               local_path,
            ignore_errors=True
)
        print(f"[delete] removed files: {local_path}")
    except Exception as e:
        print(f"[delete] file deletion warning: {e}")

    try:
        delete_collection(repo_id)
        print(f"[delete] removed vector store for {repo_id}")
    except Exception as e:
        print(f"[delete] vector store warning: {e}")

    try:
        await delete_repo_db(repo_id)
        print(f"[delete] removed db record for {repo_id}")
    except Exception as e:
        print(f"[delete] db warning: {e}")

    return {"message": "Repo deleted"}
 
class ChatRequest(BaseModel):
    repo_id: str
    query: str
    mode: str = "chat"
    target_file: str | None = None

@app.post("/api/chat")
async def chat(body: ChatRequest):
    request_start = time.perf_counter()

    try:

        print("\n========== /api/chat ==========")
        print("query:", body.query)
        print("mode:", body.mode)
        print("repo_id:", body.repo_id)

        repo = await get_repo(body.repo_id)

        if not repo:
            raise HTTPException(
                status_code=404,
                detail="Repo not found"
            )

        if repo["status"] != "ready":
            raise HTTPException(
                status_code=400,
                detail=f"Repo status: {repo['status']}"
            )

        repo_path = os.path.join(
            settings.repo_storage_path,
            body.repo_id
        )

        if body.mode == "review" and body.target_file:

            full_path = os.path.join(
                repo_path,
                body.target_file
            )

            if not os.path.exists(full_path):

                raise HTTPException(
                    status_code=400,
                    detail=f"Target file not found: {body.target_file}"
                )

        print("[STEP] analyze_query")

        analyzed = analyze_query(
            user_query=body.query,
            mode=body.mode,
            target_file=body.target_file,
        )

        print("[OK] analyze_query")

        retrieval_start = time.perf_counter()
        print("[STEP] retrieve_context")

        chunks = await asyncio.to_thread(
            retrieve_context,
            repo_id=body.repo_id,
            repo_path=repo_path,
            retrieval_query=analyzed["retrieval_query"],
            mode=body.mode,
            target_file=body.target_file,
        )

        print("[OK] retrieve_context")
        print(f"[TIMING] chat retrieval: {time.perf_counter() - retrieval_start:.2f}s")

        answer_start = time.perf_counter()
        print("[STEP] generate_answer")

        answer = await asyncio.to_thread(
            generate_answer,
            user_query=body.query,
            retrieved_chunks=chunks,
        )

        print("[OK] generate_answer")
        print(f"[TIMING] chat answer: {time.perf_counter() - answer_start:.2f}s")
        print(f"[TIMING] chat total: {time.perf_counter() - request_start:.2f}s")

        print("========== SUCCESS ==========\n")

        return {
            "answer": answer,
            "sources": [
                {
                    "file": c["metadata"]["file_path"],
                    "symbol": c["metadata"]["symbol_name"],
                }
                for c in chunks[:5]
            ]
        }

    except HTTPException:
        raise

    except Exception as e:

        print("\n========== CHAT ERROR ==========")

        traceback.print_exc()

        print("================================\n")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
class ReportRequest(BaseModel):
    repo_id: str

@app.post("/api/report")
async def generate_repo_report(body: ReportRequest):
    request_start = time.perf_counter()

    repo = await get_repo(body.repo_id)

    if not repo:
        raise HTTPException(status_code=404, detail="Repo not found")

    repo_path = os.path.join(
        settings.repo_storage_path,
        body.repo_id
    )

    analysis = analyze_query(
        user_query="repository architecture overview",
        mode="report",
    )

    retrieval_start = time.perf_counter()
    chunks = await asyncio.to_thread(
        retrieve_context,
        repo_id=body.repo_id,
        repo_path=repo_path,
        retrieval_query=analysis["retrieval_query"],
        mode="report",
    )
    print(f"[TIMING] report retrieval: {time.perf_counter() - retrieval_start:.2f}s")

    report_start = time.perf_counter()
    report = await asyncio.to_thread(
        generate_report,
        repo_name=repo["name"],
        retrieved_chunks=chunks,
    )
    print(f"[TIMING] report LLM: {time.perf_counter() - report_start:.2f}s")
    print(f"[TIMING] report total: {time.perf_counter() - request_start:.2f}s")

    return {
        "repo": repo["name"],
        "report": report,
    }
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse("frontend/index.html")

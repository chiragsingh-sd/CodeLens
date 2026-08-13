import aiosqlite
import uuid
from datetime import datetime
from backend.core.config import settings

DB_PATH = settings.sqlite_db_path

async def init_db():
   

    async with aiosqlite.connect(DB_PATH) as db:
        

        await db.execute("""
            CREATE TABLE IF NOT EXISTS repos (
                id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                owner TEXT NOT NULL,
                name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                file_count INTEGER DEFAULT 0,
                chunk_count INTEGER DEFAULT 0,
                error_msg TEXT,
                created_at TEXT NOT NULL,
                indexed_at TEXT
            )
        """)

      
        await db.commit()

        print("SQLite database initialized.")

async def create_repo(url: str, owner: str, name: str) -> dict:
    """
    Insert a new repo row. Returns the created record.
    """
    repo_id = str(uuid.uuid4())[:8]     # short ID like "a3f8c21d"
    created_at = datetime.utcnow().isoformat()

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO repos (id, url, owner, name, status, created_at)
               VALUES (?, ?, ?, ?, 'pending', ?)""",
            (repo_id, url, owner, name, created_at)
        )
        await db.commit()

    return {"id": repo_id, "url": url, "owner": owner, "name": name,
            "status": "pending", "created_at": created_at}


async def get_repo(repo_id: str) -> dict | None:
    """
    Fetch one repo by ID. Returns None if not found.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row  # makes results behave like dicts
        async with db.execute(
            "SELECT * FROM repos WHERE id = ?", (repo_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None


async def update_repo_status(
    repo_id: str,
    status: str,
    file_count: int | None = None,
    chunk_count: int | None = None,
    current_file: str | None = None,
    error_msg: str | None = None,
):
    """
    Update the status of a repo during indexing.
    """
    indexed_at = datetime.utcnow().isoformat() if status == "ready" else None

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """UPDATE repos
               SET status = ?,
                   file_count = COALESCE(?, file_count),
                   chunk_count = COALESCE(?, chunk_count),
                   error_msg = ?,
                   indexed_at = COALESCE(?, indexed_at)
               WHERE id = ?""",
            (status, file_count, chunk_count, error_msg, indexed_at, repo_id)
        )
        await db.commit()


async def get_all_repos() -> list[dict]:
    """
    List all repos, newest first.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM repos ORDER BY created_at DESC"
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
        
async def delete_repo_db(repo_id: str) -> None:
    """Remove a repo record from SQLite permanently."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM repos WHERE id = ?", (repo_id,))
        await db.commit()

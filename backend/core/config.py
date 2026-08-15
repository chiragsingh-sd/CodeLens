from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    
    groq_api_key: str = os.getenv("groq_api_key")
    nvidia_api_key:  str = os.getenv("NVIDIA_API_KEY")
    llm_provider: str = "nvidia"
    llm_model: str = "deepseek-ai/deepseek-v4-flash"
    repo_storage_path: str = os.getenv(
    "repo_storage_path",
    "/tmp/codelens/repos"
)
    chroma_persist_path: str = os.getenv(
    "chroma_persist_path",
    "/tmp/codelens/chroma"
)
    sqlite_db_path: str = os.getenv(
    "sqlite_db_path",
    "/tmp/codelens/jobs.db"
)
    max_files_per_repo: int = 200
    groq_model: str = "openai/gpt-oss-120b"
    llm_timeout_seconds: float = 45.0
    llm_max_retries: int = 0

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"



settings = Settings()

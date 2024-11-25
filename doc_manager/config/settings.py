from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    CHROMA_DB_DIR: str = "./chroma_db"
    DOCUMENT_DIR: str = "./documents"
    COLLECTION_NAME: str = "documents"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    class Config:
        env_file = ".env"

settings = Settings()

# Ensure directories exist
Path(settings.CHROMA_DB_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.DOCUMENT_DIR).mkdir(parents=True, exist_ok=True)

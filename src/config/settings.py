from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppSettings:
    project_name: str = "Agentic AI-Driven Analytics and Decision Intelligence Platform"
    db_path: str = "analytics_audit.db"
    faiss_index_path: str = "data/faiss_index"
    max_preview_rows: int = 15
    min_confidence: float = 0.5

    @property
    def db_url(self) -> str:
        return f"sqlite:///{self.db_path}"

    @property
    def faiss_dir(self) -> Path:
        path = Path(self.faiss_index_path)
        path.mkdir(parents=True, exist_ok=True)
        return path


SETTINGS = AppSettings()

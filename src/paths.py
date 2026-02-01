"""Project paths. Data lives in data/ at repo root."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def db_path() -> str:
    return str(DATA_DIR / "ruokasuositusdata.db")

"""Caminhos centrais do project2_from_project1, independentes do diretório de execução."""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VOCAB_DIR = PROJECT_DIR / "vocabularies"
WEB_DIR = PROJECT_DIR / "src" / "visualization" / "web"
ASSETS_DIR = PROJECT_DIR / "assets"

CASES_PATH = RAW_DIR / "cases.csv"
NODES_PATH = PROCESSED_DIR / "nodes.csv"
EDGES_PATH = PROCESSED_DIR / "edges.csv"

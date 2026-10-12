import argparse
from pathlib import Path

from project1.paths import CASES_PATH, EDGES_PATH, NODES_PATH, WEB_DIR
from project1.src.visualization.web_server import serve_graph_interface

def main() -> None:
    """Inicia servidor web para visualização do grafo de conhecimento."""
    parser = argparse.ArgumentParser(
        description="Run the knowledge graph interface on localhost."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument(
        "--nodes",
        default=NODES_PATH,
        type=Path,
    )
    parser.add_argument(
        "--edges",
        default=EDGES_PATH,
        type=Path,
    )
    parser.add_argument(
        "--cases",
        default=CASES_PATH,
        type=Path,
    )
    args = parser.parse_args()

    serve_graph_interface(
        nodes_path=args.nodes,
        edges_path=args.edges,
        static_dir=WEB_DIR,
        cases_path=args.cases,
        host=args.host,
        port=args.port,
    )

if __name__ == "__main__":
    main()

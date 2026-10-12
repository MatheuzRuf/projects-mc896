import argparse
from pathlib import Path

from project1.paths import ASSETS_DIR, EDGES_PATH, NODES_PATH
from project1.src.visualization import visualize_case

def main() -> None:
    """Gera uma visualizacao basica de um caso clinico e salva em PNG."""
    parser = argparse.ArgumentParser(
        description="Generate a basic PNG visualization for one clinical case."
    )
    parser.add_argument(
        "--case-id",
        help="Case to visualize. The first case is used when omitted.",
    )
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
        "--output",
        default=ASSETS_DIR / "images" / "graph.png",
        type=Path,
    )
    args = parser.parse_args()

    case_id = visualize_case(
        nodes_path=args.nodes,
        edges_path=args.edges,
        output_path=args.output,
        case_id=args.case_id,
    )

    print(f"Visualized case: {case_id}")
    print(f"Graph saved at: {args.output}")

if __name__ == "__main__":
    main()

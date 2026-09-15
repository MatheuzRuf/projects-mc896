import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent # define a raiz do projeto como diretório base
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from src.graph.build import export_graph_from_cases, load_entity_vocabulary
from src.preprocessing.dataset import preprocess_cases

def main() -> None:
    cases_path = base_dir / "data" / "raw" /  "cases.csv" # diretório com os casos brutos
    vocab_dir = base_dir / "vocabularies" # diretório com os vocabulários de entidades
    out_dir = base_dir / "data" / "processed" # diretório de saída com os arquivos que representam o grafo

    cases = preprocess_cases(cases_path)
    vocabulary = load_entity_vocabulary(vocab_dir)

    export_graph_from_cases(
        cases,
        vocabulary,
        nodes_path=out_dir / "nodes.csv",
        edges_path=out_dir / "edges.csv",
    )

    print(f"Exported {len(cases)} cases to {out_dir}")
    print(f"Nodes saved at: {out_dir / 'nodes.csv'}")
    print(f"Edges saved at: {out_dir / 'edges.csv'}")

if __name__ == "__main__":
    main()

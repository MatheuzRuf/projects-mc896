from project2_from_project1.src.graph.build import export_graph_from_cases, load_entity_vocabulary
from project2_from_project1.src.preprocessing.dataset import preprocess_cases
from project2_from_project1.paths import CASES_PATH, PROCESSED_DIR, VOCAB_DIR

def main() -> None:
    cases_path = CASES_PATH # casos brutos
    vocab_dir = VOCAB_DIR # vocabulários de entidades
    out_dir = PROCESSED_DIR # saída com os arquivos que representam o grafo

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

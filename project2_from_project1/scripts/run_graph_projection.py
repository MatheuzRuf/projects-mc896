"""Project the end-to-end factorial arms into the versioned two-layer Knowledge Graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project2_from_project1.src.graph_study import run_graph_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--gold-projection", required=True)
    parser.add_argument("--partitions", required=True, help="Frozen supervised-dataset manifest")
    parser.add_argument("--entity-study", required=True)
    parser.add_argument("--factorial", required=True, help="Output directory of run_relation_factorial")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, nargs="*")
    arguments = parser.parse_args()
    summary = run_graph_study(arguments.corpus, arguments.gold_projection, arguments.partitions, arguments.entity_study,
                              arguments.factorial, arguments.output_dir, folds=arguments.folds)
    print(json.dumps({arm: {"concept_node_f1": round(data["pooled"]["concept_nodes"]["f1"], 4),
                            "concept_edge_f1": round(data["pooled"]["concept_edges"]["f1"], 4)} if "pooled" in data else None
                      for arm, data in summary["arms"].items()}, indent=2))


if __name__ == "__main__":
    main()

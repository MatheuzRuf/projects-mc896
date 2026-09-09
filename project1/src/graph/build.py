from pathlib import Path

from src.extraction import extract_entities
from src.extraction.relations import extract_case_relations
from src.graph.export import save_edges, save_nodes
from src.graph.schema import Node, NodeType
from src.vocab import load_vocabulary


ENTITY_FILES = [
    "diseases.csv",
    "symptoms.csv",
    "exams.csv",
    "treatments.csv",
    "medications.csv",
    "anatomical_sites.csv",
]


def _node_id(case_id: str, label: str) -> str:
    clean = " ".join(label.strip().lower().split())
    return f"{case_id}:{clean}"


def _entity_to_node(case_id: str, entity) -> Node:
    return Node(
        node_id=_node_id(case_id, entity.label),
        type=NodeType(entity.type),
        label=entity.label,
        attributes=f"source={entity.source};code={entity.code}",
    )


def load_entity_vocabulary(vocabulary_dir: str | Path = "vocabularies") -> list:
    vocabulary_dir = Path(vocabulary_dir)
    vocabulary = []

    for filename in ENTITY_FILES:
        vocabulary.extend(load_vocabulary(vocabulary_dir / filename))

    return vocabulary


def build_case_graph(case_id: str, case_text: str, vocabulary: list) -> tuple[list[Node], list]:
    entities = extract_entities(case_text, vocabulary)

    node_map: dict[str, Node] = {}
    nodes: list[Node] = []

    for entity in entities:
        node = _entity_to_node(case_id, entity)
        if node.node_id not in node_map:
            node_map[node.node_id] = node
            nodes.append(node)

    edges = extract_case_relations(case_id, entities)

    valid_edges = []
    for edge in edges:
        if edge.source_id in node_map and edge.target_id in node_map:
            valid_edges.append(edge)

    return nodes, valid_edges


def build_graph_from_cases(cases: list[dict], vocabulary: list) -> tuple[list[Node], list]:
    all_nodes: list[Node] = []
    all_edges: list = []
    seen_nodes: set[str] = set()

    for case in cases:
        case_id = case.get("case_id")
        case_text = case.get("case_text", "")
        if not case_id or not case_text:
            continue

        nodes, edges = build_case_graph(case_id, case_text, vocabulary)

        for node in nodes:
            if node.node_id not in seen_nodes:
                seen_nodes.add(node.node_id)
                all_nodes.append(node)

        all_edges.extend(edges)

    return all_nodes, all_edges


def export_graph_from_cases(cases: list[dict], vocabulary: list, nodes_path: str | Path, edges_path: str | Path) -> None:
    nodes, edges = build_graph_from_cases(cases, vocabulary)
    save_nodes(nodes, nodes_path)
    save_edges(edges, edges_path)


__all__ = [
    "ENTITY_FILES",
    "load_entity_vocabulary",
    "build_case_graph",
    "build_graph_from_cases",
    "export_graph_from_cases",
]

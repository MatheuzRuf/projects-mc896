from src.graph.schema import Edge, Relation
from src.extraction.entities import Entity

def _node_id(case_id: str, label: str) -> str:
    """Cria um identificador estável do nó mantendo a label legível."""
    normalized = " ".join(label.strip().lower().split())
    return f"{case_id}:{normalized}"

def _dedupe_edges(edges: list[Edge]) -> list[Edge]:
    """Remove arestas duplicadas preservando ordem."""
    seen: set[tuple[str, str, str]] = set()
    unique: list[Edge] = []

    for edge in edges:
        key = (edge.source_id, edge.target_id, edge.relation.value)
        if key in seen:
            continue
        seen.add(key)
        unique.append(edge)

    return unique

def extract_case_relations(case_id: str, entities: list[Entity]) -> list[Edge]:
    """
    Gera arestas de relacionamento para um único caso clínico.

    A lógica é intencionalmente simples e baseada em regras:
    - Patient -> Symptom : PRESENTS_WITH
    - Patient -> Exam : UNDERWENT_EXAM
    - Diagnosis -> Symptom : HAS_FINDING
    - Patient -> Medication : TREATED_BY

    Os nós do grafo são identificados por "case_id:label_normalizada".
    Isso permite que cada caso tenha uma sub-árvore de entidades em um grafo
    do tipo knowledge graph, sem depender de um repositório global de nós.
    """

    entities_by_type: dict[str, list[Entity]] = {}
    for entity in entities:
        entities_by_type.setdefault(entity.type, []).append(entity)

    edges = []

    patient_entities = entities_by_type.get("Patient", [])
    symptom_entities = entities_by_type.get("Symptom", [])
    exam_entities = entities_by_type.get("Exam", [])
    diagnosis_entities = entities_by_type.get("Diagnosis", [])
    medication_entities = entities_by_type.get("Medication", [])

    for patient in patient_entities:
        patient_node = _node_id(case_id, patient.label)
        for symptom in symptom_entities:
            symptom_node = _node_id(case_id, symptom.label)
            edges.append(
                Edge(
                    edge_id=f"{patient_node}->{symptom_node}:{Relation.PRESENTS_WITH.value}",
                    source_id=patient_node,
                    target_id=symptom_node,
                    relation=Relation.PRESENTS_WITH,
                    attributes="",
                )
            )

        for exam in exam_entities:
            exam_node = _node_id(case_id, exam.label)
            edges.append(
                Edge(
                    edge_id=f"{patient_node}->{exam_node}:{Relation.UNDERWENT_EXAM.value}",
                    source_id=patient_node,
                    target_id=exam_node,
                    relation=Relation.UNDERWENT_EXAM,
                    attributes="",
                )
            )

        for medication in medication_entities:
            medication_node = _node_id(case_id, medication.label)
            edges.append(
                Edge(
                    edge_id=f"{patient_node}->{medication_node}:{Relation.TREATED_BY.value}",
                    source_id=patient_node,
                    target_id=medication_node,
                    relation=Relation.TREATED_BY,
                    attributes="",
                )
            )

    for diagnosis in diagnosis_entities:
        diagnosis_node = _node_id(case_id, diagnosis.label)
        for symptom in symptom_entities:
            symptom_node = _node_id(case_id, symptom.label)
            edges.append(
                Edge(
                    edge_id=f"{diagnosis_node}->{symptom_node}:{Relation.HAS_FINDING.value}",
                    source_id=diagnosis_node,
                    target_id=symptom_node,
                    relation=Relation.HAS_FINDING,
                    attributes="",
                )
            )

    return _dedupe_edges(edges)


__all__ = [
    "_node_id",
    "_dedupe_edges",
    "extract_case_relations",
]

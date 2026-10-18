import csv
from pathlib import Path

from project2_from_project1.src.clinical_documents import ClinicalCase, ClinicalDocument, Patient
from project2_from_project1.src.leakage_audit import build_protected_collection, write_protected_collection


def generate_fingerprints(csv_path: str, output_path: str) -> None:
    documents = []
    with open(csv_path, mode="r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            article_id = row["article_id"].strip()
            case_id = row["case_id"].strip()
            documents.append(
                ClinicalDocument(
                    document_id=case_id,
                    text=row["case_text"],
                    cases=[ClinicalCase(case_id=f"{case_id}:case", source_id=article_id)],
                    patients=[Patient(f"{case_id}:patient")],
                    mentions=[],
                    relations=[],
                    metadata={"article_id": article_id, "pmcid": article_id},
                )
            )
    if not documents:
        raise ValueError("O CSV não retornou casos válidos. A coleção não pode ser vazia.")
    collection = build_protected_collection(documents)
    write_protected_collection(collection, output_path)
    print(f"Sucesso! {collection.article_count} artigos ({len(documents)} casos) salvos em {output_path}")


if __name__ == "__main__":
    generate_fingerprints(
        csv_path=Path(__file__).resolve().parents[2] / "project1" / "data" / "raw" / "cases.csv",
        output_path=Path(__file__).resolve().parents[1] / "data" / "secure" / "multicare.fingerprints.json",
    )
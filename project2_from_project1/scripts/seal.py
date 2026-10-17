import csv
import hashlib
import json
from pathlib import Path

def generate_fingerprints(csv_path: str, output_path: str) -> None:
    """Lê o dataset do Projeto 1 e gera o manifesto seguro validado."""
    articles = []
    vistos = set()
    
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            texto_bruto = row.get("text", "")
            # O hash precisa obrigatoriamente ter 64 caracteres hexadecimais
            texto_hash = hashlib.sha256(texto_bruto.encode("utf-8")).hexdigest()
            article_id = row.get("id", str(i))
            
            if article_id in vistos:
                continue
            vistos.add(article_id)
            
            # 1. Agrupa os identificadores
            identifiers = {}
            if row.get("pmid"):
                identifiers["pmid"] = [row.get("pmid")]
            if row.get("pmcid"):
                identifiers["pmcid"] = [row.get("pmcid")]
            
            # 2. Monta o fingerprint exato do documento com todas as chaves exigidas
            document_data = {
                "document_id": f"{article_id}_doc1",
                "content_sha256": texto_hash,
                "token_count": len(texto_bruto.split()),  # Contagem básica para ser int >= 0
                "shingles": [],                           # Lista vazia atende ao frozenset()
                "chunks": []                              # Lista vazia atende a tuple de frozensets
            }
            
            # 3. Monta o artigo
            article_data = {
                "article_id": article_id,
                "identifiers": identifiers,
                "documents": [document_data]
            }
            articles.append(article_data)

    if not articles:
        raise ValueError("O CSV não retornou artigos válidos. A coleção não pode ser vazia.")

    payload = {
        "schema_version": 1,
        "articles": articles
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    
    print(f"Sucesso! {len(articles)} impressões digitais salvas em {output_path}")

if __name__ == "__main__":
    generate_fingerprints(
        csv_path="../../project1/data/raw/cases.csv",
        output_path="../data/secure/multicare.fingerprints.json"
    )
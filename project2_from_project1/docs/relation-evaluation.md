# Avaliação de relações pelo contrato comum

Cada extrator produz uma linha JSON para cada par candidato ouro da partição a
ser avaliada. A interface usa as entidades ouro neste marco; portanto mede a
classificação de relações, e não detecção de entidades. Não omita negativos:
cada par precisa ter exatamente uma previsão, positiva ou `NO_RELATION`.

```json
{
  "document_id": "mcrc:001",
  "source_id": "mcrc:001:patient",
  "target_id": "mcrc:001:finding",
  "native_label": "case_has_finding",
  "canonical_label": "HAS_FINDING",
  "method": "rules-v1",
  "evidence": {"rule": "patient-finding"},
  "sentences": [0],
  "model_version": "rules-v1.0",
  "neighbor_vote_fraction": null,
  "confidence": null
}
```

`native_label` preserva o rótulo produzido pelo extrator e `canonical_label`
é um dos cinco rótulos supervisionados ou `NO_RELATION`. `evidence` pode ser
qualquer objeto JSON e `sentences` contém índices de sentenças sem repetição.
Quando um método kNN fornecer `neighbor_vote_fraction`, o valor deve estar em
`[0, 1]`; ele é registrado como fração de votos dos vizinhos, nunca como
confiança calibrada. Por enquanto `confidence` deve ser nulo ou ausente.

```powershell
python project2_from_project1/scripts/evaluate_relation_predictions.py `
  --gold-candidates data/secure/mcrc-supervised/test.jsonl `
  --predictions artifacts/rules-test.predictions.jsonl `
  --output artifacts/rules-test.evaluation.json `
  --seed 42 --minimum-positive-support 5
```

O relatório contém macro-F1 positiva como métrica primária (sem
`NO_RELATION`), métricas por classe, matriz de confusão, baseline que sempre
prediz a classe majoritária e tempos. A acurácia aparece explicitamente apenas
como métrica auxiliar. Classes positivas cujo suporte ouro esteja abaixo de
`minimum_positive_support` constam como
`inconclusive_insufficient_support`; seus resultados não devem sustentar uma
conclusão de desempenho. O artefato também registra configuração, semente,
versão de Python, métodos e versões de modelo, hashes SHA-256 dos candidatos e
das previsões, sem incluir texto clínico.

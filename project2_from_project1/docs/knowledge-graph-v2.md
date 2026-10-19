# Knowledge Graph supervisionado (JSONL v2)

O grafo é projetado **depois** das relações, a partir de previsões
end-to-end congeladas. O contrato canônico é o JSONL v2 de
[`supervised-contracts.md`](supervised-contracts.md). CSV e NetworkX são
apenas visões derivadas.

## Camadas

- **Menção:** um nó por entidade prevista, com documento, offsets, tipo,
  sentença, método e versão do reconhecedor, mais um nó lógico `Patient` por
  documento. Há uma aresta por relação positiva prevista, cuja proveniência
  guarda método, versão e o `candidate_id` que originou a decisão.
  `NO_RELATION` não gera aresta.
- **Conceito:** a identidade é documento, tipo canônico e superfície
  normalizada (Unicode NFKC, `casefold`, espaços consolidados). Não há
  stemming, sinônimos nem ontologias. Superfícies iguais com tipos diferentes
  continuam como conceitos distintos. Cada nó conceitual lista os IDs dos nós
  de menção que o sustentam, e cada aresta conceitual lista as arestas de
  menção que agrega.

O grafo gold de referência usa as menções de `flat-gold-v1`. Relações gold
cujo endpoint foi descartado na adjudicação não ganham aresta inventada e são
contadas em `relations_lost_by_projection`.

## Execução

```bash
python -m project2_from_project1.scripts.run_graph_projection \
  --corpus project2_from_project1/data/secure/mcrc.jsonl \
  --gold-projection project2_from_project1/artifacts/entity-gold-v1/gold-projection.jsonl \
  --partitions project2_from_project1/artifacts/first-milestone/frozen-dataset/manifest.json \
  --entity-study project2_from_project1/artifacts/entity-nested-cv-v1 \
  --factorial project2_from_project1/artifacts/relation-factorial-v1 \
  --output-dir project2_from_project1/artifacts/graph-v2
```

Para cada braço e fold, a execução grava:

| Arquivo | Conteúdo |
| :--- | :--- |
| `graph.jsonl` | contrato canônico, validado por recarga |
| `concept-nodes.csv`, `concept-edges.csv` | camada conceitual no formato CSV legado |
| `mention-nodes.csv`, `mention-edges.csv` | camada de menção no formato CSV legado |
| `evaluation.json` | métricas conceituais secundárias e diagnósticos de agregação |
| `manifest.json` | configuração, contagens, hashes de entradas e saídas, sem texto clínico |

Nos CSVs, os IDs recebem o prefixo `document_id:`, para que
`visualize_case(..., case_id=<document_id>)` selecione um documento. A coluna
`attributes` guarda o ID canônico, a camada e a proveniência. `label` contém a
superfície clínica, então os CSVs e o JSONL devem permanecer privados.
`to_networkx` gera um `MultiDiGraph` cujos atributos são os registros do
contrato.

## Resultados (5 folds; exploratório)

Métricas conceituais exatas, secundárias às métricas estritas de menções e
relações:

| Braço | F1 de nós conceituais | F1 de arestas conceituais | Arestas corretas só após agregação | Macro-F1 de relações (primária) |
| :--- | ---: | ---: | ---: | ---: |
| dicionário + regras | 0,232 ± 0,027 | 0,080 ± 0,010 | 0 | 0,040 |
| dicionário + k-NN | 0,232 ± 0,027 | 0,125 ± 0,027 | 20 | 0,063 |
| CRF + regras | 0,268 ± 0,040 | 0,152 ± 0,033 | 4 | 0,139 |
| CRF + k-NN | 0,268 ± 0,040 | 0,208 ± 0,030 | 23 | 0,157 |

- O F1 de nós é igual entre classificadores com o mesmo reconhecedor, porque
  os nós dependem só das entidades.
- "Corretas só após agregação" conta arestas conceituais gold cujas arestas de
  menção de apoio estão todas erradas no nível estrito. Elas ficam corretas
  porque a mesma superfície aparece em outra ocorrência, e por isso o grafo
  conceitual parece melhor que as relações estritas.
- Entre 8 e 28 relações gold por fold não chegam ao grafo gold por perda de
  endpoint na projeção.

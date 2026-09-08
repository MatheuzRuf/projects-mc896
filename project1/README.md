# MC896 - Projeto 1

Extração de informações clínicas do dataset **MultiCaRe** e representação como um **Knowledge Graph**.

## Estrutura

```text
.
├── sample/
│   ├── cases.csv
│   ├── metadata.csv
│   └── data_dictionary.csv
├── src/
│   ├── preprocessing/
│   ├── retrieval/
│   ├── extraction/
│   ├── graph/
│   ├── vocab/
│   └── visualization/
├── output/
│   ├── nodes.csv
│   └── edges.csv
├── vocabularies/
├── notebooks/
├── requirements.txt
└── README.md
```

## Dados

O arquivo principal é `sample/cases.csv`. Cada linha representa um caso clínico e o campo `case_text` contém o texto a ser processado.

`metadata.csv` fornece informações adicionais sobre os artigos, incluindo termos MeSH.

## Knowledge Graph

### Nodes

| Campo        | Descrição              |
| ------------ | ---------------------- |
| `node_id`    | Identificador único    |
| `type`       | Tipo da entidade       |
| `label`      | Texto da entidade      |
| `attributes` | Informações adicionais |

Tipos principais: `Patient`, `Disease`, `Symptom`, `Exam`, `Finding`, `Treatment`, `Medication`, `Measurement`.

### Edges

| Campo        | Descrição              |
| ------------ | ---------------------- |
| `edge_id`    | Identificador único    |
| `source_id`  | Nó de origem           |
| `target_id`  | Nó de destino          |
| `relation`   | Relação entre os nós   |
| `attributes` | Informações adicionais |

Exemplos de relações: `PRESENTS_WITH`, `UNDERWENT_EXAM`, `DIAGNOSED_WITH`, `TREATED_BY`, `HAS_RESULT`, `HAS_VALUE`, `HAS_UNIT`.

## Pipeline

```text
cases.csv
   ↓
Tokenização + Normalização
   ↓
Dicionários / vocabulários
   ↓
Extração de entidades + Regex
   ↓
Extração de relações por regras
   ↓
Knowledge Graph
   ↓
nodes.csv + edges.csv
```

Serão implementados **Boolean Retrieval**, **TF-IDF** e **Vector Space Model** para recuperação e ranqueamento e a extração de informações clínicas será feita utilizando métodos clássicos.
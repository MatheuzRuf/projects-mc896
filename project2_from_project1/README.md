# MC896 - Projeto 2

Extração de informações clínicas do dataset **X** e representação como um **Knowledge Graph**.

## Estrutura

## Objetivo

O objetivo do Projeto 2 foi evoluir a técnica de extração de informação dos textos de casos clínicos utilizada no Projeto 1 (que utiliza regex para extrair entidades e relações). Implementamos um pipeline de preparação de dados, treinamento, validação e testes, utilizando o KNN (k-nearest neighbors) como classificador de aprendizado supervisionado.

## Dados

Inicialmente, pretendemos utilizar os dados do Medical Case Report Corpus (MCRC) do PubMed Central. Adicionamos ao repositório, no diretório `data/secure`, o arquivo do corpus fornecido pela Ada Health no repositório https://github.com/adahealth/medical_case_report_corpus, no formato `.pickle`. O `.gitignore` define regra para não versionar arquivos de dados brutos e metadados. Foram construídos scripts que geram arquivos `manifest.json` para armazenar os metadados obtidos no pós-processamento dos dados.

O corpus do MCRC possui 53 documentos, 4294 anotações / menções (todas manuais) e 2975 relações.

Os rótulos originais utilizados para descrição de entidades e relações no MCRC são:

| Tipo | Campo | Rótulos |
| :--- | :--- | :--- |
| Entidades | `entityType` | `finding`, `factor`, `condition`, `modifier` |
| Relações | `type` | `has`, `modifies`, `causes` |

**Nó de paciente implícito:** nenhuma entidade é do tipo `case`. O nó de origem de todas as relações `has` (o paciente) não existe entre as anotações.

## Preparação dos dados

A primeira parte do projeto consistiu na implementação de um pipeline de sanitização e preparação dos dados. Como a intenção é treinar o modelo para receber e classificar o dataset do MultiCaRe, expô-lo a esses dados durante o treinamento e a validação causaria data leakage e enviesaria o modelo. Por isso, definimos um processo de auditoria de segurança pelo qual os dados passam antes de serem liberados para uso no aprendizado de máquina. Os principais passos dessa fase são:

### Conversão do `.pickle` em `.jsonl`

O arquivo `.pickle` do MCRC é convertido em um `.jsonl`, que armazena um objeto JSON por linha, com chaves para dados e metadados provenientes do dataset original. Cada linha do arquivo pós-processado tem o seguinte formato:

```json
{
  "document_id": "",
  "text": "",
  "metadata": {},
  "cases": [
    {
      "case_id": "",
      "source_id": "",
      "provenance": {}
    }
  ],
  "patients": [
    {
      "patient_id": "",
      "attributes": {}
    }
  ],
  "mentions": [
    {
      "mention_id": "",
      "canonical_type": "",
      "native_types": [],
      "spans": [
        []
      ]
    }
  ],
  "relations": [
    {
      "relation_id": "",
      "source_id": "",
      "target_id": "",
      "canonical_label": "",
      "native_label": ""
    }
  ]
}
```

Os rótulos de entidades e relações nativos do MCRC foram convertidos para entidades e relações canônicas. A conversão segue as tabelas:

| MCRC | Canônico P2 |
| :--- | :--- |
| nó implícito | Patient |
| `condition` (ex.: Crohn's disease) | Condition |
| `finding` (ex.: pain, fever, liver, liver function tests) | Finding |
| `factor` (ex.: woman, 49-year-old, hypertension) | Factor |
| `modifier` (ex.: no, normal, negative) | Modifier |

| MCRC | Canônico P2 |
| :--- | :--- |
| `has` → condition | `HAS_CONDITION` (Patient → Condition) |
| `has` → finding | `HAS_FINDING` (Patient → Finding) |
| `has` → factor | `HAS_FACTOR` (Patient → Factor) |
| `modifies` | `NEGATES` (Modifier → Finding) |
| `causes` | `CAUSES` (Condition → Finding) |
| sintética | `DESCRIBES` (Case → Patient) |

Para garantir a integridade da fonte de dados, o conversor exige o hash SHA-256 do dataset original como credencial de entrada.

#### Comando utilizado na conversão 
converter .pickle em .jsonl
```python
python -m project2_from_project1.scripts.convert_mcrc \
  --artifact project2_from_project1/data/secure/CaseReportCorpus.pickle \
  --output-jsonl project2_from_project1/data/secure/mcrc.jsonl \
  --manifest project2_from_project1/data/secure/mcrc.manifest.json \
  --repository "https://github.com/adahealth/medical_case_report_corpus" \
  --revision "b12fbc3b42d7572f26563bd50ee48ee4a6036c3d" \
  --sha256 "9afe5fc78b97860615365b41736701ea03d894d3440bb167e9b6cf9e6fc1f962" \
  --allow-unsafe-deserialization
```

### Geração do dataset de treino, validação e teste

Para gerar o dataset fixo de treino, validação e teste, utilizamos um pipeline focado em controle de contexto e bloqueio de data leakage. Com um script Python, geramos um manifesto de fingerprints, que armazena hashes SHA-256 dos textos do MultiCaRe utilizados no Projeto 1. 

O pipeline compara cada documento do MCRC com os fingerprints do MultiCaRe e atribui um de três estados, aplicando as regras nesta ordem:

| Ordem | Regra | Critério | Decisão |
| :--- | :--- | :--- | :--- |
| 1 | `identifier_match` | PMCID, PMID ou DOI iguais aos de um artigo protegido | **Excluído** |
| 2 | `content_hash_match` | SHA-256 do texto normalizado igual ao de um documento protegido | **Excluído** |
| 3 | `text_similarity_threshold` | Similaridade ≥ **0,85** e pelo menos **3 shingles** em comum | **Quarentena** (revisão manual) |
| 4 | `similarity_below_threshold` | Nenhuma das condições acima | **Liberado** |

Detalhes da comparação por similaridade:

- O texto é dividido em *shingles*: sequências de 5 tokens consecutivos, representadas por um hash.
- A similaridade é a proporção de shingles compartilhados em relação ao menor dos dois conjuntos (`interseção / min(|A|, |B|)`), e não o índice de Jaccard. Assim, um caso clínico copiado dentro de um artigo maior continua sendo detectado, sem ser diluído pelo restante do texto.
- A comparação é feita tanto sobre o documento inteiro quanto sobre trechos de 120 tokens com passo de 60, o que detecta cópias parciais. Vale o maior dos dois scores.
- Exigir ao menos 3 shingles em comum (equivalente a pelo menos 7 tokens consecutivos iguais) evita que frases genéricas curtas, comuns em texto clínico, mandem um documento para a quarentena.

Documentos **excluídos** ou em **quarentena** não podem ser enviados ao treinamento. Um documento em quarentena só é liberado após checagem manual.

A divisão do dataset reserva 20% dos dados para teste e o restante para validação e treino. A técnica utilizada foi a validação cruzada k-fold, que consiste em dividir o conjunto de dados em *k* partições de tamanho aproximadamente igual (chamadas de folds). São realizadas *k* iterações de treinamento. Em cada rodada, uma das partições é utilizada para validação e as restantes para treino, de forma que todas as partições sejam validação exatamente uma vez. Essa técnica diminui a incerteza associada ao desempenho do modelo para conjuntos de dados específicos, tornando o modelo mais generalista. Ao fim da auditoria, nenhum documento foi bloqueado, garantindo a segurança do ambiente para inicio do trabalho.

Os arquivos que armazenam os conjuntos de validação, treino e teste guardam apenas chaves estrangeiras que referenciam o documento equivalente no arquivo JSONL que armazena os dados brutos e metadados. As chaves presentes em cada objeto JSONL do dataset são:

| Chave | Descrição |
| :--- | :--- |
| `candidate_relation` | Tipo de relação avaliada como hipótese entre as duas entidades (ex.: `CAUSES`, `NEGATES`, `HAS_CONDITION`, `HAS_FACTOR`, `HAS_FINDING`). |
| `document_id` | Identificador único do documento de origem, geralmente referenciando o diretório, a URL ou o ID de publicação do caso clínico (ex.: `PMC3017697`). |
| `gold_relation_id` | Identificador da relação no conjunto anotado (padrão-ouro) caso ela seja verdadeira, ou `null` caso seja um pareamento sem relação válida. |
| `label` | Classe definitiva da instância, usada como valor real (*ground truth*) que o modelo deve aprender a prever. Assume um valor de relação (ex.: `NEGATES`, `CAUSES`) ou `NO_RELATION` para relações falsas. |
| `sentence_distance` | Distância, em número de sentenças, entre a menção de origem e a menção alvo no texto original (inteiro como 0, 1, 65, ou `null`). |
| `source_id` | Identificador único e posicional da menção de origem (*source*) dentro do documento. |
| `source_type` | Tipo canônico atribuído à entidade de origem (ex.: `Condition`, `Modifier`, `Patient`). |
| `target_id` | Identificador único e posicional da menção alvo (*target*) dentro do documento. |
| `target_type` | Tipo canônico atribuído à entidade alvo (ex.: `Finding`, `Factor`, `Condition`). |

#### Comandos utilizado para geração do arquivo de fingerprinting e do dataset de treino validação e teste
gerar as impressões digitais da coleção protegida (MultiCaRe, Projeto 1), usadas na auditoria de vazamento:
```python
python -m project2_from_project1.scripts.seal
```

gerar dataset de validação e treino (janela de 1 sentença; `-1` remove o limite):
```python
python -m project2_from_project1.scripts.reproduce_first_milestone prepare \
  --safe-jsonl project2_from_project1/data/secure/mcrc.jsonl \
  --protected-collection project2_from_project1/data/secure/multicare.fingerprints.json \
  --output-directory project2_from_project1/artifacts/first-milestone \
  --seed 42 --test-fraction 0.2 --folds 5 \
  --maximum-sentence-distance 1 --maximum-train-negatives-per-positive 2
```
### Exemplo prático de dataset gerado para um texto genérico

Dado um texto de caso clínico genérico, rotulado com as bases do MCRC, o dataset de treino gerado seguiria o seguinte padrão:

> *clinical text:* "A patient presents to the emergency department reporting a sudden onset of intense chest discomfort that began about an hour ago. The pain is described as a heavy pressure in the center of the chest. The clinical presentation raises immediate concern and warrants urgent investigation."

Exemplo de linha gerada:

```json
{
  "candidate_relation": "HAS_FINDING",
  "document_id": "mcrc:emergency_case_001",
  "gold_relation_id": "mcrc:emergency_case_001:relation:854",
  "label": "HAS_FINDING",
  "sentence_distance": 0,
  "source_id": "mcrc:emergency_case_001:patient:101",
  "source_type": "Patient",
  "target_id": "mcrc:emergency_case_001:mention:102",
  "target_type": "Finding"
}
```

- **A relação (`candidate_relation` e `label`):** estamos testando a relação `HAS_FINDING` (tem sintoma / achado). Como o texto diz explicitamente que o paciente relata o desconforto, a relação é verdadeira, então o `label` também é `HAS_FINDING` e há um `gold_relation_id` preenchido.
- **A origem (*source*):** quem tem o sintoma.
  - `source_type`: `Patient`.
  - `source_id`: identificador interno do nó do paciente.
- **O alvo (*target*):** o que o paciente tem.
  - `target_type`: `Finding` (achado clínico).
  - `target_id`: identificador interno de onde "chest discomfort" foi encontrado no texto.
- **Distância (`sentence_distance`):** "patient" e "chest discomfort" aparecem na primeira frase do trecho, portanto a distância entre elas é 0.
- **Documento (`document_id`):** identificador fictício do texto de origem (`mcrc:emergency_case_001`).

## N-grams

N-gramas são estruturas que armazenam as frequências com que um termo (unigramas), dois termos (bigramas) ou três termos (trigramas) aparecem em conjunto dentro de uma região do texto clínico. Utilizando probabilidade condicional e as frequências registradas pelos n-gramas, podemos calcular a probabilidade de ocorrência de um termo no texto, dado que um padrão de termos já ocorreu.

No projeto, utilizamos a estrutura `Counter` da biblioteca `collections` do Python para varrer os textos clínicos, armazenando contagens de unigramas, bigramas e trigramas. Como o dataset de treino armazena um id posicional dos termos de origem e alvo dentro do documento, podemos delimitar a janela de contexto usada para gerar os n-gramas: a região entre os dois termos, excluindo os próprios termos da contagem.

Como o nó `Patient` é implícito e não é anotado no corpus do MCRC, as entidades que se relacionam com ele não têm um segundo ponto de referência para delimitar o contexto. Nesses casos utilizamos uma janela de contexto pré-definida de 4 palavras antes e 4 palavras depois da menção em questão.

## KNN (k-nearest neighbors)

O classificador recebe uma linha esparsa por par candidato. O vocabulário de
unigramas, bigramas e trigramas é aprendido exclusivamente no arquivo de
treino; validação e teste são transformados sem acrescentar características,
evitando *feature leakage*. A implementação usa uma classificação binária por
tipo de relação, votação uniforme entre vizinhos e distância cosseno. Assim, um
candidato `CAUSES`, por exemplo, só pode resultar em `CAUSES` ou
`NO_RELATION`.

O comando abaixo treina no primeiro fold, gera exatamente uma previsão para
cada candidato de validação e executa o avaliador comum:

```powershell
python -m project2_from_project1.scripts.run_ngram_knn `
  --corpus project2_from_project1/data/secure/mcrc.jsonl `
  --train-candidates project2_from_project1/artifacts/first-milestone/frozen-dataset/fold-01-train.jsonl `
  --predict-candidates project2_from_project1/artifacts/first-milestone/frozen-dataset/fold-01-validation.jsonl `
  --predictions project2_from_project1/artifacts/knn/fold-01.predictions.jsonl `
  --evaluation project2_from_project1/artifacts/knn/fold-01.evaluation.json `
  --n-neighbors 3 --model-version ngram-knn-v1
```

Além das previsões, o comando grava um manifesto com hashes das entradas e da
saída, tamanho do vocabulário, número de exemplos e configuração do k-NN. O
campo `neighbor_vote_fraction` registra apenas a fração dos vizinhos que votou
na classe vencedora; ele não representa confiança calibrada.

Consulte [`docs/ngram-knn.md`](docs/ngram-knn.md) para as fronteiras de
treino/validação e os critérios necessários antes de avaliar o teste fixo.

## Estudo end-to-end (texto bruto → entidades → relações → grafo)

O estudo comparativo completo, com dicionário e semi-Markov CRF combinados com
regras e k-NN, é executado por uma única entrada:
`python -m project2_from_project1.scripts.run_full_study`. Use `--synthetic DIR`
para um teste rápido sem dados clínicos. O passo a passo para outra máquina
está em [`docs/running-the-study.md`](docs/running-the-study.md).

## Knowledge Graph

### Nodes

### Edges

## Pipeline

# Projeto X

## Visão Geral

### Objetivos

### Escopo da primeira entrega

## Organização do Projeto

### Casos clínicos

## Metodologia

## Modelo Lógico


### Tipos de nós implementados


### Famílias de relações implementadas


## Trabalhos Estudados


## Análises que Podem ser Realizadas

## Ferramentas


## Resultados


## Como Modelos de Linguagem Foram Usados


## Limitações e Próximos Passos


## Referências Bibliográficas

1. NIEVAS OFFIDANI, Mauro Andrés; DELRIEUX, Claudio Augusto. *Dataset of clinical cases, images, image labels and captions from open access case reports from PubMed Central (1990–2023).* Data in Brief, v. 52, art. 110008, 2024. DOI: [10.1016/j.dib.2023.110008](https://doi.org/10.1016/j.dib.2023.110008).
2. NIEVAS OFFIDANI, Mauro Andrés; DELRIEUX, Claudio Augusto. *The MultiCaRe Dataset: A Multimodal Case Report Dataset with Clinical Cases, Labeled Images and Captions from Open Access PMC Articles.* Zenodo, 2023. DOI: [10.5281/zenodo.10079370](https://doi.org/10.5281/zenodo.10079370).
3. DRIVENDATA. *Cookiecutter Data Science.* Disponível em: [https://cookiecutter-data-science.drivendata.org/](https://cookiecutter-data-science.drivendata.org/). Acesso em: 14 set. 2026.
4. JI, Shaoxiong et al. A Survey on Knowledge Graphs: Representation, Acquisition, and Applications. IEEE Transactions on Neural Networks and Learning Systems, v. 33, n. 2, p. 494–514, 1 fev. 2022. DOI:(https://doi.org/10.1109/TNNLS.2021.3070843)

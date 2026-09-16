# MC896 - Projeto 1

Extração de informações clínicas do dataset **MultiCaRe** e representação como um **Knowledge Graph**.

## Estrutura

```text
.
├── data/
│   ├── raw/
│   │   ├── cases.csv
│   │   ├── metadata.csv
│   │   └── data_dictionary.csv
│   └── processed/
│       ├── nodes.csv
│       └── edges.csv
├── src/
│   ├── preprocessing/
│   ├── retrieval/
│   ├── extraction/
│   ├── graph/
│   ├── vocab/
│   └── visualization/
├── vocabularies/
├── notebooks/
├── requirements.txt
└── README.md
```

## Dados

O arquivo principal é `data/raw/cases.csv`. Cada linha representa um caso clínico e o campo `case_text` contém o texto a ser processado.

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

Exemplos de relações: `PRESENTS_WITH`, `UNDERWENT_EXAM`, `DIAGNOSED_WITH`, `TREATED_BY`, `HAS_RESULT`, `HAS_DOSE`, `HAS_VALUE`, `HAS_UNIT`.

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

---

# Projeto Extração de Informações Clínicas e Construção de Grafo de Conhecimento
# Project Clinical Information Extraction and Knowledge Graph Construction

## Visão Geral

Este projeto foi desenvolvido para a disciplina **MC896 — Processamento de Línguas Naturais** e tem como objetivo transformar narrativas clínicas em inglês, provenientes de uma amostra do dataset MultiCaRe, em uma representação estruturada na forma de grafo de conhecimento.

A proposta combina técnicas clássicas de Processamento de Línguas Naturais (PLN), Recuperação de Informação (RI) e reconhecimento de padrões. O texto de cada caso é preservado, normalizado e tokenizado; em seguida, entidades clínicas são reconhecidas por correspondência com vocabulários controlados, medições são identificadas por expressões regulares e relações são inferidas por regras. Ao final, os nós e as arestas são exportados para arquivos CSV que podem ser analisados programaticamente ou explorados por meio das visualizações incluídas no projeto.

### Objetivos

- organizar os textos clínicos e seus metadados em uma estrutura reproduzível;
- implementar busca booleana com índice invertido e listas de postings;
- ranquear casos relevantes por TF-IDF e similaridade vetorial;
- identificar entidades, valores, unidades, intervalos de referência e interpretações clínicas;
- representar entidades e relações em um grafo de propriedades;
- exportar o grafo em um formato tabular simples e interoperável;
- oferecer visualização estática e exploração local interativa do resultado;
- manter testes automatizados para os componentes centrais do pipeline.

### Escopo da primeira entrega

O repositório trabalha com a amostra disponível em `data/raw`, não com a totalidade do MultiCaRe. A entrega concentra-se em métodos clássicos e interpretáveis: normalização, tokenização, remoção de stopwords para recuperação, índice invertido, operadores booleanos, TF-IDF, similaridade vetorial, casamento por dicionário, expressões regulares e regras determinísticas de relação. O projeto não se propõe, nesta etapa, a produzir diagnóstico médico ou substituir avaliação clínica.


## Organização do Projeto

A estrutura atualmente implementada é a seguinte:

```text
project1/
├── README.md                         # apresentação e documentação principal
├── LOCAL_INTERFACE.md                # instruções da interface web local
├── VISUALIZATION.md                  # instruções da visualização estática
├── pyproject.toml                    # metadados e requisito de versão do Python
├── requirements.txt                  # dependências Python
├── data/
│   ├── raw/
│   │   ├── cases.csv                 # textos e identificadores dos casos clínicos
│   │   ├── metadata.csv              # metadados dos artigos de origem
│   │   └── data_dictionary.csv       # descrição dos campos da amostra
│   └── processed/                    # saída do pipeline (nodes.csv, edges.csv)
│       ├── nodes.csv                     # nós extraídos
│       └── edges.csv                     # relações extraídas
├── vocabularies/                     # termos e aliases clínicos controlados
├── src/
│   ├── preprocessing/                # leitura, normalização e tokenização
│   ├── retrieval/                    # índice invertido e busca booleana
│   ├── RankedSearch/                 # TF-IDF, modelo vetorial e ranking
│   ├── Regex/                        # extração e estruturação de medições
│   ├── extraction/                   # entidades e relações clínicas
│   ├── graph/                        # esquema, construção e exportação do grafo
│   ├── vocab/                        # carregamento dos vocabulários
│   └── visualization/                # PNG, servidor local e aplicação web
├── scripts/
│   ├── export_knowledge_graph.py     # gera nodes.csv e edges.csv
│   ├── visualize_knowledge_graph.py  # gera uma imagem PNG de um caso
│   └── serve_graph_interface.py      # inicia a interface web local
└── tests/                             # testes automatizados do pipeline
```

Essa organização separa dados de entrada, vocabulários, código-fonte, scripts executáveis, resultados e testes. Ela segue a intenção da estrutura sugerida para a disciplina.
## Dados e Vocabulários

### Casos clínicos

O arquivo `data/raw/cases.csv` é a entrada principal do pipeline. Os campos essenciais consumidos pelo código são:

| Campo | Uso no projeto |
| --- | --- |
| `case_id` | identifica unicamente o caso e compõe os identificadores dos nós |
| `case_text` | contém a narrativa clínica usada em recuperação e extração |

O módulo de pré-processamento mantém o texto original e acrescenta três representações derivadas: texto normalizado, tokens completos e tokens de recuperação sem stopwords. A preservação do texto original é importante porque posições de caracteres e formas de superfície são usadas na extração de entidades e medições.

### Metadados

`data/raw/metadata.csv` reúne informações complementares dos artigos, incluindo termos MeSH. Na implementação atual, a construção do grafo parte diretamente de `cases.csv`; os metadados permanecem disponíveis para análises, enriquecimento semântico e rastreabilidade futura.

### Dicionário de dados

`data/raw/data_dictionary.csv` documenta os campos fornecidos pela amostra. Ele deve ser consultado antes de incluir novas colunas ou alterar qualquer etapa de leitura dos dados.

### Vocabulários controlados

Os arquivos CSV de `vocabularies/` organizam conceitos clínicos por categoria: doenças, sintomas, exames, tratamentos, medicamentos, sítios anatômicos, achados, estados, cursos clínicos, desfechos, unidades e interpretações. Cada entrada pode fornecer um termo canônico, aliases e, quando disponível, um código externo.

Durante a extração, o casamento é feito de forma insensível a maiúsculas e minúsculas e respeita limites de palavras. Quando um alias aparece no texto, o grafo utiliza o rótulo canônico e registra a origem e o código como atributos. Essa estratégia reduz variações lexicais sem perder a forma observada na narrativa.

## Metodologia

### 1. Leitura e preservação dos casos

`src/preprocessing/dataset.py` lê o CSV com `csv.DictReader`. Registros sem `case_id` ou sem `case_text` não seguem para a construção do grafo. Para os registros válidos, o texto original permanece disponível durante todo o processamento.

### 2. Normalização e tokenização

`src/preprocessing/text.py` aplica normalização Unicode NFKC, uniformiza aspas e hífens, reduz sequências de espaços e converte a cópia destinada à recuperação para letras minúsculas. A expressão de tokenização foi projetada para preservar elementos frequentes no domínio clínico, como porcentagens, números decimais, unidades compostas e palavras hifenizadas.

Exemplos de unidades e formas mantidas como tokens incluem `mg/L`, `U/L`, `92%` e `5-day`.

```python
result = preprocess_text(case["case_text"])
tokens = result.tokens
```

### 3. Recuperação booleana

O projeto constrói um índice invertido em que cada termo proveniente dos tokens gerados pelo texto, aponta para uma lista ligada de ocorrências por documento. Cada posting armazena o `case_id` e a frequência do termo no documento. Os documentos são ordenados por uma chave numérica derivada de seu identificador (construído utilizando a parte numérica do `case_id` + seu sufixo), permitindo a inserção nas listas de maneira ordenada de forma crescente (por ordenação de tuplas em python), o que garante a possibilidade de aplicação dos algoritmos de boolean retrieval. O índice mantém todos os tokens, porém, utiliza uma flag para marcar termos como stopwords, usando critérios baseados em IDF ou fração de documentos (DF), permitindo calibrar a filtragem em rotinas como do módulo TF-IDF, sem necessidade de reconstrução da lista de índices.

As operações disponíveis são:

- `AND`: interseção dos documentos que contêm os dois termos;
- `OR`: união dos documentos que contêm pelo menos um dos termos;
- `NOT`: complemento em relação ao universo conhecido de casos.


### 4. Recuperação ranqueada

O módulo `src/RankedSearch/` transforma os documentos e a consulta em vetores ponderados. A frequência do termo e a frequência inversa de documentos seguem as expressões:

```text
TF(t, d)  = log(1 + frequência(t, d))
IDF(t)    = log(N / df(t))
TF-IDF    = TF(t, d) × IDF(t)
```

Depois da vetorização, os documentos são normalizados e ordenados pelo produto escalar com o vetor da consulta, que corresponde à base do modelo de espaço vetorial. A implementação utiliza a mesma fórmula de IDF no módulo `RankedSearch` e, durante a vetorização, ignora termos marcados como stopwords no índice (ou seja, a marcação de stopwords é considerada na construção dos vetores). Consultas sem termos presentes no índice retornam uma lista vazia; por padrão, documentos com pontuação zero são omitidos.

### 5. Extração de entidades clínicas

`src/extraction/entities.py` realiza *dictionary matching* sobre os vocabulários controlados. Para cada entidade reconhecida são armazenados:

- rótulo canônico;
- tipo clínico;
- posições inicial e final no texto;
- formas de superfície encontradas;
- fonte do conceito;
- código externo, quando informado no vocabulário.

Entidades repetidas são consolidadas, mantendo todas as ocorrências encontradas. Além das entidades extraídas, cada caso recebe um nó `Patient`, que funciona como ponto de entrada para as relações clínicas daquele caso.

### 6. Extração de medições por expressões regulares

Os módulos em `src/Regex/` reconhecem valores numéricos, unidades, intervalos de referência e interpretações. O adaptador de medições transforma cada ocorrência em uma estrutura nomeada contendo texto, valor, unidade, intervalo, interpretação, posição e entidade clínica associada.

A associação considera a entidade mensurável mais próxima na mesma sentença e limita a distância entre a entidade e a medição. O tipo da unidade ajuda a restringir candidatos: unidades de tempo podem indicar duração; `mm`, `cm` e `m` podem indicar tamanho; e unidades farmacológicas podem indicar dose.

### 7. Extração de relações

`src/extraction/relations.py` aplica regras determinísticas que conectam entidades do mesmo caso. As relações cobrem histórico, sintomas, achados, exames, resultados, diagnósticos, tratamentos, medicamentos, desfechos, atributos estruturados e alinhamento com vocabulários.

As relações são deduplicadas e somente são exportadas quando os nós de origem e destino realmente existem. Essa validação evita arestas órfãs no resultado final.

### 8. Construção e exportação do grafo

`src/graph/build.py` coordena a extração de entidades, relações e medições. Os identificadores são determinísticos e combinam o `case_id` com uma versão normalizada do rótulo. O resultado é gravado por `src/graph/export.py` em dois arquivos:

- `data/processed/nodes.csv`, com os nós e seus atributos;
- `data/processed/edges.csv`, com origem, destino, relação e atributos.

O fluxo completo pode ser resumido assim:

```mermaid
flowchart TD
    A["data/raw/cases.csv"] --> B["Leitura e preservação do texto"]
    B --> C["Normalização e tokenização"]
    C --> D["Índice invertido"]
    D --> E["Busca booleana"]
    D --> F["TF-IDF e ranking vetorial"]
    B --> G["Casamento com vocabulários"]
    B --> H["Regex de medições"]
    G --> I["Entidades clínicas"]
    H --> J["Valores, unidades e interpretações"]
    I --> K["Regras de relações"]
    J --> K
    K --> L["Validação e deduplicação"]
    L --> M["data/processed/nodes.csv"]
    L --> N["data/processed/edges.csv"]
    M --> O["Visualização PNG ou interface web"]
    N --> O
```

## Modelo Lógico

O grafo segue um modelo de propriedades: nós possuem `node_id`, `type`, `label` e `attributes`; arestas possuem `edge_id`, `source_id`, `target_id`, `relation` e `attributes`.

### Tipos de nós implementados

| Grupo | Tipos |
| --- | --- |
| núcleo do caso | `Patient`, `History` |
| manifestações clínicas | `Symptom`, `Finding` |
| investigação | `Exam`, `ExamResult` |
| diagnóstico e intervenção | `Diagnosis`, `Medication`, `Treatment` |
| evolução | `Outcome`, `Status`, `Course` |
| atributos estruturados | `Measurement`, `Value`, `Unit`, `ReferenceRange`, `Interpretation`, `AnatomicalSite` |
| interoperabilidade | `VocabConcept` |

### Famílias de relações implementadas

| Família | Relações |
| --- | --- |
| paciente e história | `HAS_HISTORY`, `PRESENTS_WITH`, `HAS_FINDING` |
| exames e evidências | `UNDERWENT_EXAM`, `HAS_RESULT`, `REVEALS`, `CONFIRMS`, `EXCLUDES`, `SUPPORTS` |
| diagnóstico e causalidade | `PREDISPOSES_TO`, `DIAGNOSED_WITH` |
| intervenção | `TREATED_BY`, `TARGETS` |
| evolução | `HAS_OUTCOME`, `LEADS_TO` |
| medidas e atributos | `HAS_DURATION`, `HAS_SIZE`, `HAS_DOSE`, `LOCATED_IN`, `HAS_VALUE`, `HAS_UNIT`, `HAS_REFERENCE_RANGE`, `HAS_LOW`, `HAS_HIGH`, `HAS_INTERPRETATION`, `HAS_STATUS`, `HAS_COURSE` |
| vocabulário | `SAME_AS` |

Uma versão conceitual simplificada do modelo é apresentada abaixo:

```mermaid
graph LR
    P[Patient] -->|PRESENTS_WITH| S[Symptom]
    P -->|HAS_FINDING| F[Finding]
    P -->|UNDERWENT_EXAM| E[Exam]
    P -->|DIAGNOSED_WITH| D[Diagnosis]
    P -->|TREATED_BY| T[Treatment ou Medication]
    P -->|HAS_OUTCOME| O[Outcome]
    E -->|HAS_RESULT| R[ExamResult]
    R -->|HAS_VALUE| V[Value]
    R -->|HAS_UNIT| U[Unit]
    R -->|HAS_REFERENCE_RANGE| RR[ReferenceRange]
    R -->|HAS_INTERPRETATION| I[Interpretation]
    F -->|LOCATED_IN| A[AnatomicalSite]
    D -->|SAME_AS| C[VocabConcept]
```

Para a versão final, o diagrama também pode ser exportado como PNG para `assets/images/`, conforme o modelo sugerido pela disciplina.

## Trabalhos Estudados

O trabalho central é o artigo de apresentação do MultiCaRe, que descreve a aquisição, a organização e o pré-processamento de um conjunto multimodal de relatos de caso de acesso aberto publicados no PubMed Central entre 1990 e 2023. Neste projeto, utiliza-se apenas uma amostra textual desse universo para investigar a passagem de narrativas clínicas não estruturadas para um grafo consultável.

## Análises que Podem ser Realizadas

O grafo produzido permite formular perguntas que seriam trabalhosas diretamente sobre texto livre. Entre as possibilidades estão:

- identificar sintomas, achados e exames mais associados a cada diagnóstico;
- comparar tratamentos utilizados em casos com desfechos distintos;
- localizar valores laboratoriais, unidades e interpretações ligados a exames específicos;
- investigar quais achados apoiam, confirmam ou excluem diagnósticos;
- observar sequências entre predisposição, diagnóstico, tratamento e desfecho;
- contabilizar entidades e relações por caso, tipo clínico ou artigo de origem;
- detectar componentes desconectados, nós isolados ou relações incompletas como parte do controle de qualidade;
- usar busca booleana para formar subconjuntos de casos e o ranking TF-IDF para priorizar os mais relevantes;
- comparar consultas textuais com a vizinhança das entidades correspondentes no grafo;
- avaliar cobertura dos vocabulários pela proporção de menções reconhecidas e não reconhecidas.

Essas análises são exploratórias. As relações são inferidas por proximidade e regras linguísticas, portanto uma conexão no grafo representa a saída do método, e não uma afirmação médica validada.

## Ferramentas

| Ferramenta ou tecnologia | Papel no projeto |
| --- | --- |
| Python 3.10+ | implementação do pipeline e dos scripts |
| biblioteca padrão (`csv`, `re`, `unicodedata`, `dataclasses`) | leitura, normalização, expressões regulares e estruturas de dados |
| NumPy | operações vetoriais do ranking |
| Pandas | suporte à manipulação tabular |
| scikit-learn | dependência disponível para métodos de RI e PLN |
| NetworkX | montagem e análise da representação em rede para visualização |
| Matplotlib | geração da visualização estática em PNG |
| HTML, CSS, JavaScript e SVG | interface web local e renderização interativa |
| pytest | testes automatizados |
| CSV | intercâmbio simples dos dados de entrada e do grafo exportado |

A escolha por métodos clássicos torna as decisões mais auditáveis: termos reconhecidos podem ser rastreados aos vocabulários, medições às expressões regulares e relações às regras que as produziram. Como contrapartida, o desempenho depende da cobertura lexical e da variedade de construções previstas manualmente.

## Resultados

O estado atual do repositório inclui os dois artefatos de grafo esperados, `data/processed/nodes.csv` e `data/processed/edges.csv`, além dos mecanismos para regenerá-los a partir da amostra. O resultado materializa:

- entidades clínicas normalizadas como nós;
- atributos de proveniência e código quando fornecidos pelo vocabulário;
- valores, unidades, intervalos e interpretações como componentes explícitos;
- relações clínicas e relações de atributos entre os nós;
- identificadores determinísticos por caso;
- filtragem de arestas inválidas e deduplicação das relações;
- visualização estática de um caso e navegação interativa local.

Os resultados devem ser interpretados considerando as limitações do método. Casamento por dicionário pode perder sinônimos ausentes ou gerar ambiguidades; regras de proximidade não resolvem toda a sintaxe clínica; negação, temporalidade e correferência podem exigir tratamento adicional; e uma amostra não representa necessariamente toda a diversidade do MultiCaRe. Uma avaliação quantitativa futura deve comparar uma amostra anotada manualmente com a extração e reportar precisão, revocação e F1 por tipo de entidade e relação.

## Como Modelos de Linguagem Foram Usados

O pipeline base documentado neste repositório não depende de modelos de linguagem generativos, a recuperação e a extração são implementadas com métodos clássicos, vocabulários, expressões regulares e regras. Foram usados modelos de linguagem para auxilio na elaboração de regex complexos. A extração dos termos em csv também foram feitas usando auxiliarmente modelos de linguagem. 

Este README também foi feito em parte usando LLM, com posterior revisão, detalhamento e alterações.

## Limitações e Próximos Passos

- incorporar explicitamente os metadados e termos MeSH ao grafo;
- aprofundar o tratamento de negação, incerteza, temporalidade e correferência;
- separar dados brutos, intermediários e processados conforme a estrutura completa sugerida pela disciplina, caso a equipe aprove essa migração;
- registrar métricas quantitativas e exemplos comentados na versão final do relatório;
- adaptação das etapas que estão hardcoded.
- otimização do módulo boolean retrieval para operar com operações que exigem precedência.

## Referências Bibliográficas

1. NIEVAS OFFIDANI, Mauro Andrés; DELRIEUX, Claudio Augusto. *Dataset of clinical cases, images, image labels and captions from open access case reports from PubMed Central (1990–2023).* Data in Brief, v. 52, art. 110008, 2024. DOI: [10.1016/j.dib.2023.110008](https://doi.org/10.1016/j.dib.2023.110008).
2. NIEVAS OFFIDANI, Mauro Andrés; DELRIEUX, Claudio Augusto. *The MultiCaRe Dataset: A Multimodal Case Report Dataset with Clinical Cases, Labeled Images and Captions from Open Access PMC Articles.* Zenodo, 2023. DOI: [10.5281/zenodo.10079370](https://doi.org/10.5281/zenodo.10079370).
3. DRIVENDATA. *Cookiecutter Data Science.* Disponível em: [https://cookiecutter-data-science.drivendata.org/](https://cookiecutter-data-science.drivendata.org/). Acesso em: 14 set. 2026.
4. JI, Shaoxiong et al. A Survey on Knowledge Graphs: Representation, Acquisition, and Applications. IEEE Transactions on Neural Networks and Learning Systems, v. 33, n. 2, p. 494–514, 1 fev. 2022. DOI:(https://doi.org/10.1109/TNNLS.2021.3070843)

# MC896 - Projeto 2

Extração de informações clínicas do dataset **X** e representação como um **Knowledge Graph**.

## Estrutura

## Dados

Os dados utilizados para validação e treino do modelo são provenientes do Medical Case Report Corpus. Disponível em: https://github.com/adahealth/medical_case_report_corpus

## Preparação dos dados de validação e treino

Descrever pipeline de preparação dos dados (conversão .pickle em jsonl, controle de data leakage, etc)

Comandos executados:

1 - converter .pickle em .jsonl
```python
python -m project2_from_project1.scripts.convert_mcrc \
  --artifact project2_from_project1/data/secure/CaseReportCorpus.pickle \
  --output-jsonl project2_from_project1/data/secure/mcrc.jsonl \
  --manifest project2_from_project1/data/secure/mcrc.manifest.json \
  --repository "[https://github.com/adahealth/medical_case_report_corpus](https://github.com/adahealth/medical_case_report_corpus)" \
  --revision "b12fbc3b42d7572f26563bd50ee48ee4a6036c3d" \
  --sha256 "9afe5fc78b97860615365b41736701ea03d894d3440bb167e9b6cf9e6fc1f962" \
  --allow-unsafe-deserialization
```

2 - gerar dataset de validação e treino:
```python
python -m project2_from_project1.scripts.reproduce_first_milestone prepare \
  --safe-jsonl project2_from_project1/data/secure/mcrc.jsonl \
  --protected-collection project2_from_project1/data/secure/multicare.fingerprints.json \
  --output-directory project2_from_project1/artifacts/first-milestone \
  --seed 42 --test-fraction 0.2 --folds 5 \
  --maximum-sentence-distance -1 --maximum-train-negatives-per-positive 2
```

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

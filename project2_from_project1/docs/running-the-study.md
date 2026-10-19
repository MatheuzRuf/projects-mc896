# Executar o estudo completo em outra máquina (CPU)

Este guia reproduz o estudo end-to-end: texto clínico bruto → entidades →
candidatos → relações → Knowledge Graph, para todos os braços. Todos os
comandos rodam a partir da raiz do monorepo.

## 1. Ambiente

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
pip install -r requirements.txt
pip install -e .
```

A variante CPU do PyTorch evita baixar as bibliotecas CUDA. Nenhuma etapa usa
GPU.

## 2. Teste rápido sem dados clínicos

```bash
python -m project2_from_project1.scripts.run_full_study \
  --synthetic /tmp/mc896-synthetic --output-dir /tmp/mc896-synthetic-out --max-iterations 30
```

Esse comando gera uma fixture sintética de 12 relatos e percorre todos os
braços em poucos segundos. Ele verifica que as etapas se encaixam, não a
qualidade dos modelos. A mesma costura é testada em `tests/test_full_study.py`.

## 3. Dados

1. Baixe `CaseReportCorpus.pickle` do repositório da Ada Health para
   `project2_from_project1/data/secure/` e converta-o com o comando
   `convert_mcrc` do [README](../README.md), que confere o SHA-256 do pickle.
2. O `mcrc.jsonl` resultante deve ter SHA-256
   `28844ee5c2770024ace12c95026c03e59bd0708e34a16ed05b9bc418799b8050`. A
   adjudicação versionada (`annotations/entity-gold-v1/adjudication.jsonl`) está
   vinculada a esse hash. Um arquivo com outros bytes, por exemplo com quebras
   de linha CRLF, é recusado.
3. Gere as impressões digitais do MultiCaRe (`scripts.seal`) e o dataset
   congelado (`reproduce_first_milestone prepare`), também descritos no
   README. Os folds devem sair em
   `project2_from_project1/artifacts/first-milestone/frozen-dataset/`, e seu
   manifesto deve registrar o mesmo hash do corpus.

`data/` e `artifacts/` não são versionados porque contêm texto clínico.

## 4. Estudo completo

```bash
python -m project2_from_project1.scripts.run_full_study \
  --corpus project2_from_project1/data/secure/mcrc.jsonl \
  --adjudication project2_from_project1/annotations/entity-gold-v1/adjudication.jsonl \
  --frozen-dataset project2_from_project1/artifacts/first-milestone/frozen-dataset \
  --output-dir project2_from_project1/artifacts/study-v1 \
  --workers 5 --threads-per-worker 1
```

| Etapa | Saída | Tempo de referência (12 núcleos, 5 workers) |
| :--- | :--- | ---: |
| Projeção gold | `gold/` | segundos |
| Entidades (65 treinos do CRF + dicionário) | `entities/` | ~4 h |
| Relações (matriz fatorial + teto) | `relations/` | ~1 min |
| Grafo | `graph/` | segundos |

Ajuste `--workers` à memória disponível: cada worker usa entre 0,5 e 1 GB.
Treinos do CRF com 1 thread em paralelo rendem mais que um treino com várias
threads.

**Retomada:** os treinos internos do CRF e os modelos finais ficam em cache,
indexados pelos hashes das entradas e pela configuração. Ao repetir o mesmo
comando, só o que falta é treinado; qualquer mudança de dado ou configuração
força um novo treino.

## 5. Saídas

`manifest.json`, na raiz da saída, é gravado por último e registra:

- commit, versões de Python, PyTorch e demais dependências;
- configuração do CRF (grade L2, folds internos, 250 iterações) e dos
  classificadores;
- partições documentais usadas;
- tempo de cada etapa;
- SHA-256 de **todos** os artefatos filhos;
- `test_partition_read: false`. O teste MCRC consumido nunca é aberto.

Os resumos de cada etapa ficam em `entities/summary.json`,
`relations/summary.json` e `graph/summary.json`. Os resultados estão
discutidos em [`entity-recognizer-comparison.md`](entity-recognizer-comparison.md),
[`relation-factorial-results.md`](relation-factorial-results.md) e
[`knowledge-graph-v2.md`](knowledge-graph-v2.md).

## 6. Falhas

- Não há limiar de desempenho: todos os braços aparecem nos resumos.
- Um CRF que termina as 250 iterações sem convergir é avaliado normalmente,
  com `converged=false`.
- Um CRF com objetivo ou gradiente `NaN`/infinito grava manifesto
  `failed_non_finite`, sem previsões. Apenas os braços `crf+*` daquele fold
  ficam `skipped_failed_recognizer`; dicionário, teto e os demais folds
  continuam. Nada substitui o CRF.

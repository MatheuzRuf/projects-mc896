# Pipeline de n-gramas e k-NN

O pipeline classifica relações entre as entidades ouro do dataset congelado.
Ele não realiza reconhecimento de entidades e não projeta diretamente o
resultado no Knowledge Graph.

## Entradas

- o corpus clínico convertido para JSONL seguro;
- um `fold-XX-train.jsonl` produzido pelo congelamento do dataset;
- o `fold-XX-validation.jsonl` correspondente, ou outra partição congelada.

Cada candidato produz exatamente um exemplo. Para duas menções textuais, o
contexto é o trecho entre elas. Quando um endpoint é um nó lógico sem span,
como `Patient`, o contexto usa quatro palavras antes e quatro depois da menção
textual. O exemplo é representado por contagens de unigramas, bigramas e
trigramas.

## Fronteira contra leakage

O vocabulário é ajustado somente nos candidatos de treino. N-gramas exclusivos
da validação ou do teste são ignorados durante a transformação. A matriz CSR é
compartilhada, mas cada tipo de relação possui um classificador binário próprio:
por exemplo, candidatos `CAUSES` só podem resultar em `CAUSES` ou
`NO_RELATION`. Isso preserva a compatibilidade semântica definida pelo dataset.
Cada classificador usa distância cosseno, votação uniforme e o valor de `k`
fornecido na linha de comando.

## Execução de um fold

```powershell
python -m project2_from_project1.scripts.run_ngram_knn `
  --corpus project2_from_project1/data/secure/mcrc.jsonl `
  --train-candidates project2_from_project1/artifacts/first-milestone/frozen-dataset/fold-01-train.jsonl `
  --predict-candidates project2_from_project1/artifacts/first-milestone/frozen-dataset/fold-01-validation.jsonl `
  --predictions project2_from_project1/artifacts/knn/fold-01.predictions.jsonl `
  --evaluation project2_from_project1/artifacts/knn/fold-01.evaluation.json `
  --n-neighbors 3
```

O comando produz:

- previsões JSONL no contrato comum;
- `fold-01.predictions.manifest.json`, com hashes e configuração;
- relatório de avaliação, quando `--evaluation` for informado.

Uma previsão contém o rótulo canônico, método, versão do modelo, evidência da
configuração e `neighbor_vote_fraction`. Essa fração é a proporção dos `k`
vizinhos que votou na classe escolhida, não uma probabilidade calibrada.

## Validação experimental

Escolha `k` e demais decisões de representação usando somente os folds de
treino e validação. Registre macro-F1 positiva e métricas por classe em todos
os folds. O `test.jsonl` deve permanecer fechado durante essa seleção e ser
avaliado uma única vez após a configuração ser fixada.

O pipeline estar executável não demonstra que o MCRC fornece suporte suficiente
para todas as classes. A conclusão depende do relatório de adequação e das
métricas reais, especialmente para classes minoritárias como `CAUSES`.

O primeiro resultado reproduzido com `k=3` está registrado em
[`ngram-knn-results.md`](ngram-knn-results.md). O teste fixo não foi aberto.

# Comparação exploratória dos reconhecedores de entidades

**Status:** exploratório. Os resultados usam somente os cinco folds dos 42
documentos de desenvolvimento. O teste MCRC consumido não foi lido, e a
conclusão final depende do novo teste externo (tickets 12 e 13). A projeção
gold `flat-gold-v1` foi adjudicada por modelos; ver
[`annotations/entity-gold-v1/README.md`](../annotations/entity-gold-v1/README.md).

## Protocolo

- Folds externos: as partições documentais congeladas
  (`artifacts/first-milestone/frozen-dataset/manifest.json`, corpus
  `28844ee5…`).
- CRF: em cada fold externo, 3 folds internos por documento (embaralhamento
  semeado, semente 42) comparam λ L2 ∈ {0,01; 0,1; 1; 10} por macro-F1
  estrita. Empates seriam resolvidos por menor variância e, depois, λ maior. O
  modelo final é retreinado no treino externo inteiro com o λ escolhido.
- Dicionário: reconstruído em cada fold externo com o gold de treino.
- Mesmo avaliador estrito, offsets e tipo exatos, para os dois braços.
- 65 treinos do CRF: 5 processos paralelos com 1 thread cada, 3,96 h no total.

Reprodução:

```bash
python -m project2_from_project1.scripts.run_entity_nested_cv \
  --corpus project2_from_project1/data/secure/mcrc.jsonl \
  --gold-projection project2_from_project1/artifacts/entity-gold-v1/gold-projection.jsonl \
  --gold-projection-manifest project2_from_project1/artifacts/entity-gold-v1/manifest.json \
  --partitions project2_from_project1/artifacts/first-milestone/frozen-dataset/manifest.json \
  --output-dir project2_from_project1/artifacts/entity-nested-cv-v1 \
  --workers 5 --threads-per-worker 1
```

`summary.json` agrega métricas, variância, convergência, tempos, erros e o
SHA-256 de cada manifesto de fold. Cada `crf/fold-XX/selection.json` registra
as partições internas e a tabela de seleção. Uma execução interrompida é
retomada sem refazer os treinos concluídos.

## Seleção de λ (macro-F1 média nos folds internos)

| Fold | 0,01 | 0,1 | 1 | 10 | Escolhido |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0,264 | **0,274** | 0,262 | 0,159 | 0,1 |
| 2 | 0,245 | **0,252** | 0,232 | 0,112 | 0,1 |
| 3 | 0,251 | 0,262 | **0,266** | 0,127 | 1 |
| 4 | 0,242 | **0,267** | 0,229 | 0,114 | 0,1 |
| 5 | 0,239 | **0,261** | 0,236 | 0,092 | 0,1 |

## Resultados nos folds externos (média ± desvio-padrão, n = 5)

| Métrica | Dicionário | Semi-Markov CRF |
| :--- | ---: | ---: |
| **Macro-F1 estrita** | 0,121 ± 0,013 | **0,285 ± 0,044** |
| Micro-F1 | 0,149 ± 0,016 | **0,243 ± 0,032** |
| Precisão (micro) | 0,108 ± 0,015 | **0,343 ± 0,033** |
| Revocação (micro) | **0,241 ± 0,029** | 0,193 ± 0,043 |
| F1 Condition | 0,013 ± 0,019 | 0,021 ± 0,033 |
| F1 Factor | 0,179 ± 0,042 | **0,518 ± 0,117** |
| F1 Finding | 0,166 ± 0,031 | **0,207 ± 0,034** |
| F1 Modifier | 0,126 ± 0,034 | **0,392 ± 0,056** |
| Revocação de formas não vistas no treino | 0,000 | **0,098 ± 0,022** |

Macro-F1 por fold: dicionário 0,113 / 0,105 / 0,118 / 0,136 / 0,132; CRF
0,319 / 0,320 / 0,219 / 0,307 / 0,258. O CRF supera o dicionário em todos os
folds.

### Decomposição dos erros (soma dos 5 folds)

| | Dicionário | CRF |
| :--- | ---: | ---: |
| Espúrios | 4.131 | 688 |
| Limite incorreto | 410 | 181 |
| Limite e tipo incorretos | 83 | 14 |
| Tipo incorreto | 0 | 5 |
| Ausentes (nada previsto sobre a menção) | 1.320 | 1.613 |
| Sobrepostos por previsão incorreta | 415 | 234 |

## Convergência

Nenhum dos 5 modelos finais convergiu: todos atingiram o limite
pré-registrado de 250 iterações do L-BFGS com valores finitos, e o gradiente
máximo final ficou entre 13,6 e 79,7. Dos 60 treinos internos, 1 convergiu.
Nenhum treino teve falha não finita. Pelo protocolo, todos foram avaliados e
mantidos com `converged=false`.

## Leitura

- O CRF tem macro-F1 cerca de 2,4 vezes maior que o dicionário. O ganho vem
  sobretudo da precisão: o contexto elimina a maioria dos espúrios, que caem
  de 4.131 para 688.
- O CRF é conservador. Ele deixa mais menções sem previsão (1.613 contra
  1.320), e por isso sua revocação micro fica abaixo da do dicionário. A
  ausência de convergência é uma explicação plausível, não verificada, para
  parte desse conservadorismo.
- Somente o CRF recupera formas não vistas no treino, com cerca de 10% de
  revocação nesse estrato.
- `Condition` fica perto de zero nos dois braços. Pela decomposição, o erro
  predominante é menção ausente, não confusão de tipo. Este é o principal
  ponto fraco a investigar.
- O desvio-padrão do CRF é maior: o fold 3, o único com λ = 1, tem o pior
  resultado.

Nenhum braço é removido ou promovido por esses números (ER-D15). Ambos
seguem para a geração de candidatos e a matriz fatorial de relações.

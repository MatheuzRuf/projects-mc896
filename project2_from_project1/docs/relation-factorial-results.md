# Matriz fatorial de relações end-to-end

**Status:** exploratório. Os resultados usam os cinco folds de desenvolvimento,
sem ler o teste MCRC consumido. As linhas `gold+*` são **teto diagnóstico** com
entidades gold e não são pipelines end-to-end.

## Protocolo

Para cada fold externo:

1. As entidades previstas pelo estudo aninhado
   ([`entity-recognizer-comparison.md`](entity-recognizer-comparison.md))
   viram candidatos não rotulados (`relation_candidates`).
2. As regras canônicas, ajustadas com os candidatos gold de treino do fold,
   classificam os candidatos do dicionário e os do CRF.
3. Um único k-NN de n-gramas por fold (unigramas, `k=1`, a configuração
   selecionada em [`ngram-knn-results.md`](ngram-knn-results.md)), ajustado
   com os candidatos gold de treino, classifica os candidatos dos dois
   reconhecedores.
4. Os mesmos classificadores, aplicados aos candidatos gold de validação,
   formam o teto.
5. Uma relação é correta somente com documento, span e tipo da origem, span e
   tipo do alvo e rótulo exatos. Todas as relações gold supervisionadas dos
   documentos de validação entram no denominador, inclusive as inalcançáveis.

```bash
python -m project2_from_project1.scripts.run_relation_factorial \
  --corpus project2_from_project1/data/secure/mcrc.jsonl \
  --gold-projection project2_from_project1/artifacts/entity-gold-v1/gold-projection.jsonl \
  --frozen-dataset project2_from_project1/artifacts/first-milestone/frozen-dataset \
  --entity-study project2_from_project1/artifacts/entity-nested-cv-v1 \
  --output-dir project2_from_project1/artifacts/relation-factorial-v1
```

`summary.json` traz, para cada braço e fold, o SHA-256 da avaliação e os
caminhos e hashes dos manifestos do reconhecedor, dos candidatos e do
classificador. A execução completa leva cerca de 50 s.

## Resultados (5 folds, contagens somadas; macro-F1 média ± desvio entre folds)

| Braço | Papel | Macro-F1 (folds) | Macro-F1 (somado) | Micro-F1 | Precisão | Revocação |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| dicionário + regras | end-to-end | 0,040 ± 0,007 | 0,040 | 0,052 | 0,040 | 0,075 |
| dicionário + k-NN | end-to-end | 0,063 ± 0,009 | 0,063 | 0,075 | 0,054 | 0,121 |
| CRF + regras | end-to-end | 0,139 ± 0,031 | 0,137 | 0,134 | 0,433 | 0,079 |
| CRF + k-NN | end-to-end | **0,157 ± 0,039** | **0,158** | **0,172** | 0,256 | 0,130 |
| gold + regras | teto | 0,345 ± 0,076 | 0,361 | 0,419 | 0,481 | 0,371 |
| gold + k-NN | teto | 0,517 ± 0,039 | 0,534 | 0,582 | 0,519 | 0,662 |

### F1 por relação (contagens somadas)

| Braço | CAUSES (24) | HAS_CONDITION (101) | HAS_FACTOR (181) | HAS_FINDING (914) | NEGATES (387) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| dicionário + regras | 0,000 | 0,000 | 0,164 | 0,000 | 0,037 |
| dicionário + k-NN | 0,000 | 0,012 | 0,182 | 0,086 | 0,036 |
| CRF + regras | 0,000 | 0,000 | 0,522 | 0,000 | 0,165 |
| CRF + k-NN | 0,000 | 0,015 | 0,502 | 0,134 | 0,137 |
| gold + regras | 0,000 | 0,316 | 0,903 | 0,000 | 0,589 |
| gold + k-NN | 0,245 | 0,496 | 0,832 | 0,607 | 0,489 |

`CAUSES` é inconclusiva nos folds 1, 4 e 5, com suporte abaixo de 5; as
demais relações são conclusivas em todos os folds.

## Atribuição dos erros (soma dos 5 folds)

Relações gold não encontradas (1.607 relações gold no total):

| Causa | dic.+regras | dic.+k-NN | CRF+regras | CRF+k-NN | gold+regras | gold+k-NN |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| endpoint descartado na projeção | 80 | 79 | 80 | 78 | 50 | 25 |
| fora da janela de candidatos | 2 | 2 | 2 | 2 | 2 | 2 |
| **herdado: endpoint não reconhecido** | 930 | 930 | 1.104 | 1.104 | 0 | 0 |
| **herdado: borda ou tipo do endpoint** | 312 | 312 | 148 | 148 | 0 | 0 |
| candidato inexistente | 0 | 0 | 0 | 0 | 0 | 0 |
| **próprio da relação: classificado `NO_RELATION`** | 162 | 90 | 146 | 66 | 958 | 516 |

Relações previstas incorretas:

| Causa | dic.+regras | dic.+k-NN | CRF+regras | CRF+k-NN | gold+regras | gold+k-NN |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| endpoint descartado na projeção | 24 | 15 | 0 | 2 | 90 | 102 |
| **herdado: endpoint não é entidade gold** | 2.819 | 3.212 | 146 | 537 | 0 | 0 |
| **próprio da relação** | 50 | 144 | 20 | 70 | 555 | 885 |

## Leitura

- **As entidades dominam o erro end-to-end.** Nos quatro braços, entre 84% e
  90% das relações gold perdidas vêm de endpoints não reconhecidos ou com borda
  ou tipo errados; o classificador responde por 5% a 11%. Com entidades
  gold, o mesmo k-NN chega a 0,53 de macro-F1, mais de três vezes o melhor
  braço end-to-end.
- **O CRF é o melhor reconhecedor para relações.** Com o mesmo classificador,
  o CRF supera o dicionário em todos os pares: com k-NN, 0,157 contra 0,063.
  Com o dicionário, 90% das relações previstas apoiam-se em uma entidade
  inexistente (3.212 de 3.565); com o CRF, 66%.
- **O k-NN supera as regras** com qualquer fonte de entidades. As regras usam
  apenas hipótese e distância e nunca preveem `HAS_FINDING` (a maioria dos
  candidatos de treino é negativa); também não preveem `CAUSES`.
- **Candidato inexistente é zero** em todos os braços: quando os dois endpoints
  estão corretos e dentro da janela, o gerador sempre cria o par.
- **Perdas da projeção** (25 a 80 relações) são pequenas, mas não nulas: elas
  vêm de menções que a adjudicação descartou.
- `HAS_CONDITION` e `CAUSES` herdam a fraqueza de `Condition` no NER e ficam
  perto de zero em todos os braços end-to-end.

## Limitações

- Os candidatos gold do teto vêm do dataset congelado, que usa o segmentador
  anterior e amostra negativos apenas no treino. A janela das previsões usa o
  segmentador versionado.
- No teto, menções lossless duplicadas são resolvidas para a mesma chave da
  projeção; uma previsão positiva em qualquer duplicata conta uma vez.
- Nenhum modelo final do CRF convergiu em 250 iterações (ticket 05).

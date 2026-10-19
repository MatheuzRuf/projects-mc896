# Braço de entidades por dicionário canônico

O baseline lexical reconhece `Condition`, `Finding`, `Factor` e `Modifier` em
texto bruto. Ele é reconstruído em cada fold usando somente as menções da
projeção gold plana (`flat-gold-v1`) dos documentos de treino. Validação,
teste, vocabulários legados e fontes externas não contribuem para o
dicionário.

## Regras

- **Forma:** a chave é a sequência de tokens do tokenizador versionado, cada um
  normalizado por Unicode NFKC e `casefold`. Espaços entre tokens não fazem
  parte da chave.
- **Ambiguidade:** uma forma vista com vários tipos recebe o tipo estritamente
  mais frequente no treino. Em empate, a forma é registrada como abstida: não
  gera previsão e não bloqueia correspondências menores sobrepostas.
- **Correspondência:** somente tokens completos dentro de uma única sentença.
  Todas as correspondências da sentença são coletadas e escolhidas da mais
  longa para a mais curta; em comprimentos iguais vence a mais à esquerda. A
  saída é plana, não sobreposta e preserva offsets do texto original.
- **Marginal:** `marginal_probability` é sempre nula; o dicionário não produz
  scores comparáveis aos do semi-Markov CRF.

## Fronteira de rótulos

A execução segue fases ordenadas: ajuste com o gold de treino, inferência com
`RawDocument` (somente ID e texto), gravação e recarga das previsões e, por
fim, avaliação. As previsões não mudam quando as anotações da partição prevista
mudam; um teste automatizado verifica essa propriedade.

## Execução de um fold

Exige a projeção gold adjudicada (ticket 02) e um manifesto de partições
congelado a partir do mesmo `mcrc.jsonl`; hashes divergentes interrompem a
execução.

```powershell
python -m project2_from_project1.scripts.run_entity_dictionary `
  --corpus project2_from_project1/data/secure/mcrc.jsonl `
  --gold-projection project2_from_project1/artifacts/entity-gold-v1/gold-projection.jsonl `
  --gold-projection-manifest project2_from_project1/artifacts/entity-gold-v1/manifest.json `
  --partitions project2_from_project1/artifacts/first-milestone/frozen-dataset/manifest.json `
  --fold 1 `
  --output-dir project2_from_project1/artifacts/entity-dictionary/fold-01
```

Saídas:

| Arquivo | Conteúdo |
| :--- | :--- |
| `predictions.jsonl` | `EntityPrediction` v1, sem texto clínico |
| `evaluation.json` | métricas estritas, decomposição de erros e diagnósticos |
| `dictionary.jsonl` | formas e contagens por tipo; **contém texto clínico**, manter privado |
| `manifest.json` | commit, hashes, partições, configuração, dependências, duração |

## Avaliação

`src/entity_recognition/evaluation.py` é compartilhado por todos os
reconhecedores. Uma entidade é correta somente com offsets e tipo idênticos. O
relatório publica micro-F1, macro-F1 sobre os quatro tipos (métrica primária),
F1 por tipo e:

- falsos positivos decompostos em `spurious`, `type_mismatch`,
  `boundary_mismatch` e `boundary_and_type_mismatch`;
- falsos negativos decompostos em `missing` e
  `overlapped_by_incorrect_prediction`;
- estratos por forma vista ou não vista no treino (`oov_surface`), por presença
  de token não visto no treino (`oov_token`) e por comprimento em tokens;
- correspondência por sobreposição com o mesmo tipo, apenas como diagnóstico.

Os estratos OOV são definidos exclusivamente pelo treino do fold.

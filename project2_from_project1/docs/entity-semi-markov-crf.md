# Braço de entidades por semi-Markov CRF

Reconhecedor aprendido que escolhe globalmente, por sentença, uma segmentação
plana em segmentos `Condition`, `Finding`, `Factor`, `Modifier` e `O` (fora de
entidade). Segmentos de entidade têm de 1 a 40 tokens; segmentos `O` têm
exatamente um token, de modo que cada saída corresponde a uma única
segmentação.

## Modelo

O score de uma segmentação `(i_k, d_k, y_k)` é

```
start[y_1] + Σ_k w_{y_k} · φ(i_k, d_k) + Σ_k trans[y_{k-1}, y_k] + end[y_K]
```

em que `φ` é o vetor esparso de features do segmento. A probabilidade é
`exp(score) / Z`, e `Z` soma todas as segmentações válidas.

O núcleo em `src/entity_recognition/semicrf.py` é implementado no projeto:

- **forward**: `log Z` por programação dinâmica com `logsumexp`;
- **Viterbi**: a mesma recorrência com `max` e ponteiros de retorno;
- **marginais**: `∂ log Z / ∂ S`, obtidas por autograd (equivalente ao
  forward-backward). Cada `EntityPrediction` registra a marginal do segmento
  escolhido em `marginal_probability`. O valor é um posterior do modelo, não
  uma confiança calibrada, e não altera a decodificação Viterbi.

Os testes comparam partição, Viterbi e marginais com enumeração exaustiva de
sentenças sintéticas pequenas.

## Features (`semicrf-sparse-v1`)

| Família | Papel | Lexical |
| :--- | :--- | :--- |
| `first.w`, `last.w`, `in.w` | identidade do primeiro, do último e dos tokens internos | sim |
| `prev.w`, `next.w` | token anterior/seguinte, ou `<BOS>`/`<EOS>` | sim |
| `first.p3`, `last.s3` | prefixo do primeiro e sufixo do último token | sim |
| `*.shape`, `*.case` | forma ortográfica e caixa de first/last/prev/next | não |
| `len`, `has_digit`, `all_punct`, `bias` | comprimento, números, pontuação, viés | não |

- Tokens são normalizados por NFKC + `casefold`.
- O vocabulário é ajustado somente nas sentenças de treino. Identidades e
  afixos exigem frequência mínima 2 no treino.
- Features desconhecidas são ignoradas, sem hashing. Um token OOV mantém forma,
  caixa, contexto e comprimento.
- `in.w` soma o one-hot de cada token interno, então um token repetido conta
  por ocorrência. Essa forma permite montar o score dos segmentos a partir de
  scores por token e somas prefixadas.
- Nenhuma feature usa o braço de dicionário.

## Treino

- PyTorch em CPU, `float64`, `torch.use_deterministic_algorithms(True)`.
- Inicialização em zero; L-BFGS full-batch (`strong_wolfe`, histórico 10).
- Objetivo `Σ [log Z − score(gold)] + (λ/2)‖θ‖²` sobre pesos, transições,
  início e fim; sem pesos de classe nem reamostragem.
- Limite de 250 iterações; tolerâncias de gradiente `1e-5` e de mudança `1e-9`.
  Ao esgotar o limite com valores finitos, o resultado é avaliado e marcado
  `converged=false`.
- Objetivo ou gradiente `NaN`/infinito interrompe a execução com
  `NonFiniteTrainingError`. O manifesto fica com `status=failed_non_finite`
  e diagnóstico, e não há troca de otimizador nem fallback.
- Uma menção gold com mais de 40 tokens gera erro explícito e nunca é
  truncada.

λ é escolhido somente por validação cruzada interna no ticket 05.

## Instalação

`requirements.txt` fixa `torch==2.14.1`. Para evitar o download das
bibliotecas CUDA, instale a variante CPU:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
```

## Execução de um fold

```powershell
python -m project2_from_project1.scripts.run_entity_crf `
  --corpus project2_from_project1/data/secure/mcrc.jsonl `
  --gold-projection project2_from_project1/artifacts/entity-gold-v1/gold-projection.jsonl `
  --gold-projection-manifest project2_from_project1/artifacts/entity-gold-v1/manifest.json `
  --partitions project2_from_project1/artifacts/first-milestone/frozen-dataset/manifest.json `
  --fold 1 --l2 1 `
  --output-dir project2_from_project1/artifacts/entity-crf/fold-01
```

A execução segue as mesmas fases e o mesmo avaliador do dicionário
([`entity-dictionary.md`](entity-dictionary.md)). Ela grava
`predictions.jsonl`, `evaluation.json`, `model.json` e `manifest.json`.
`model.json` contém tokens do treino e deve permanecer privado. O manifesto
registra configuração, threads, convergência, iterações, avaliações da função,
objetivo final, duração e hashes. Além das métricas comuns, `evaluation.json`
inclui a taxa de acerto por faixa de marginal, rotulada
`uncalibrated_model_posterior`.

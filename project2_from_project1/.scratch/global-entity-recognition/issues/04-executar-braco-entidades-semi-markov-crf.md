# 04 — Executar o braço de entidades por semi-Markov CRF

**What to build:** treinar e executar em CPU um semi-Markov CRF linear que recebe texto tokenizado com offsets, escolhe globalmente entidades dos quatro tipos canônicos e produz previsões, marginais opcionais, avaliação estrita e manifesto em um fold completo.

**Blocked by:** 02 — Produzir a projeção gold plana adjudicada.

**Status:** done

**Implementation note (2026-10-18):** núcleo segmental próprio
(`src/entity_recognition/semicrf.py`), features esparsas
(`crf_features.py`), treino L-BFGS (`crf_model.py`), executor compartilhado com
o dicionário (`run.py`) e CLI `scripts/run_entity_crf.py`. Testes em
`tests/test_semicrf_dynamic_programming.py` (enumeração exaustiva) e
`tests/test_entity_crf.py`. A dependência `torch==2.14.1` (CPU) está fixada. A
projeção gold oficial do ticket 02 já existe e o executor roda nos dados
reais. Ver
`docs/entity-semi-markov-crf.md`.

- [x] A programação dinâmica própria calcula função de partição, Viterbi e marginais de maneira numericamente estável.
- [x] Partição, melhor caminho e marginais coincidem com enumeração exaustiva em sequências sintéticas pequenas.
- [x] Features são esparsas, ajustadas apenas no treino, têm frequência lexical mínima dois e não incluem sinais do dicionário.
- [x] O treino usa PyTorch em CPU, inicialização em zero, L-BFGS full-batch, L2 e limite de 250 iterações.
- [x] Uma execução finita não convergida é avaliada com `converged=false`; valores não finitos geram falha estruturada sem fallback.
- [x] O fold produz `EntityPrediction`, métricas estritas, análise de erros e manifesto reproduzível.

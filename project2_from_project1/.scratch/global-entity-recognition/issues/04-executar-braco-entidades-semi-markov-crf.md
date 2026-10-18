# 04 — Executar o braço de entidades por semi-Markov CRF

**What to build:** treinar e executar em CPU um semi-Markov CRF linear que recebe texto tokenizado com offsets, escolhe globalmente entidades dos quatro tipos canônicos e produz previsões, marginais opcionais, avaliação estrita e manifesto em um fold completo.

**Blocked by:** 02 — Produzir a projeção gold plana adjudicada.

**Status:** ready-for-agent

- [ ] A programação dinâmica própria calcula função de partição, Viterbi e marginais de maneira numericamente estável.
- [ ] Partição, melhor caminho e marginais coincidem com enumeração exaustiva em sequências sintéticas pequenas.
- [ ] Features são esparsas, ajustadas apenas no treino, têm frequência lexical mínima dois e não incluem sinais do dicionário.
- [ ] O treino usa PyTorch em CPU, inicialização em zero, L-BFGS full-batch, L2 e limite de 250 iterações.
- [ ] Uma execução finita não convergida é avaliada com `converged=false`; valores não finitos geram falha estruturada sem fallback.
- [ ] O fold produz `EntityPrediction`, métricas estritas, análise de erros e manifesto reproduzível.

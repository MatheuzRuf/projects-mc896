# 09 — Avaliar a matriz fatorial de relações end-to-end

**What to build:** comparar dicionário e semi-Markov CRF combinados com regras e k-NN, ao lado do teto gold-entity, usando correspondência estrita e uma decomposição que atribua falhas às entidades ou ao classificador de relações.

**Blocked by:** 05 — Selecionar e comparar reconhecedores por validação aninhada; 07 — Executar relações por regras no contrato canônico; 08 — Executar k-NN sobre candidatos previstos.

**Status:** ready-for-agent

- [ ] Os quatro braços end-to-end são executados e mantidos no relatório independentemente do desempenho.
- [ ] O teto gold-entity é identificado como diagnóstico e nunca apresentado como pipeline completo.
- [ ] Relações corretas exigem endpoints, tipos e rótulo exatos; candidatos ausentes e espúrios afetam recall e precisão.
- [ ] A análise distingue erro herdado de entidade, erro próprio da relação e candidato inexistente.
- [ ] Métricas incluem micro-F1, macro-F1, F1 por relação, suporte e classes inconclusivas.
- [ ] Cada resultado referencia os manifestos exatos dos reconhecedores, candidatos e classificadores usados.

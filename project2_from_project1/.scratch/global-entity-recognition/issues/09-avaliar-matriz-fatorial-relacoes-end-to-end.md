# 09 — Avaliar a matriz fatorial de relações end-to-end

**What to build:** comparar dicionário e semi-Markov CRF combinados com regras e k-NN, ao lado do teto gold-entity, usando correspondência estrita e uma decomposição que atribua falhas às entidades ou ao classificador de relações.

**Blocked by:** 05 — Selecionar e comparar reconhecedores por validação aninhada; 07 — Executar relações por regras no contrato canônico; 08 — Executar k-NN sobre candidatos previstos.

**Status:** done

**Implementation note (2026-10-19):** avaliador estrito com atribuição de erros
(`src/relation_end_to_end.py`), orquestrador fatorial
(`src/relation_factorial.py`), CLI `scripts/run_relation_factorial.py` e
`tests/test_relation_end_to_end.py`. Macro-F1 end-to-end (média dos 5 folds):
dicionário+regras 0,040, dicionário+k-NN 0,063, CRF+regras 0,139,
CRF+k-NN 0,157. Teto gold-entity: regras 0,345, k-NN 0,517. Resultados em
`docs/relation-factorial-results.md`.

- [x] Os quatro braços end-to-end são executados e mantidos no relatório independentemente do desempenho.
- [x] O teto gold-entity é identificado como diagnóstico e nunca apresentado como pipeline completo.
- [x] Relações corretas exigem endpoints, tipos e rótulo exatos; candidatos ausentes e espúrios afetam recall e precisão.
- [x] A análise distingue erro herdado de entidade, erro próprio da relação e candidato inexistente.
- [x] Métricas incluem micro-F1, macro-F1, F1 por relação, suporte e classes inconclusivas.
- [x] Cada resultado referencia os manifestos exatos dos reconhecedores, candidatos e classificadores usados.

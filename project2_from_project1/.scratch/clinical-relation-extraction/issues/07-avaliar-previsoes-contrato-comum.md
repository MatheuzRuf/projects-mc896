# 07 — Avaliar previsões pelo contrato comum

**What to build:** aceitar previsões produzidas por qualquer extrator compatível e gerar uma avaliação comparável, auditável e resistente ao domínio de `NO_RELATION` ou das classes mais frequentes.

**Blocked by:** 02 — Preparar e recarregar um documento clínico completo.

**Status:** complete

- [x] Cada previsão identifica documento, menções de origem e destino, relações nativa e canônica, método, evidências, sentenças e versão do modelo.
- [x] `neighbor_vote_fraction` é aceito como fração de votos e não é apresentado como confiança calibrada.
- [x] A avaliação inicial usa entidades ouro e compara pares e relações pelo contrato comum.
- [x] A saída inclui macro-F1 das classes positivas, métricas por classe, matriz de confusão, baseline majoritário e tempos.
- [x] A acurácia, quando exibida, é identificada como métrica auxiliar.
- [x] Classes com suporte insuficiente tornam a conclusão correspondente inconclusiva em vez de silenciosamente positiva.
- [x] Configuração, semente, versões, hashes dos dados e métricas são registrados em artefato reproduzível.
- [x] Fixtures cobrem domínio de `NO_RELATION`, ausência de classe positiva e erro em uma classe minoritária.


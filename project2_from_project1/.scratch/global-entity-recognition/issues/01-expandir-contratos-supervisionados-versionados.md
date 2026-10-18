# 01 — Expandir os contratos supervisionados versionados

**What to build:** introduzir, ao lado dos contratos legados, contratos versionados e auditáveis para entidades previstas, candidatos de relação não rotulados e as camadas de menção e conceito do grafo. Uma fixture sintética deve atravessar serialização e recarga sem perda de identidade, offsets, tipos ou proveniência.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] `EntityPrediction` faz round-trip determinístico sem campos que finjam anotação nativa e sem duplicar texto clínico.
- [ ] `RelationCandidate` faz round-trip sem campo `label` e distingue rótulo desconhecido de `NO_RELATION`.
- [ ] O grafo JSONL v2 representa nós e arestas das camadas de menção e conceito com esquema e proveniência explícitos.
- [ ] Contratos e exportações legados continuam funcionando sem mudança funcional.
- [ ] Entradas inválidas falham com mensagens que identificam o campo e o registro problemáticos.

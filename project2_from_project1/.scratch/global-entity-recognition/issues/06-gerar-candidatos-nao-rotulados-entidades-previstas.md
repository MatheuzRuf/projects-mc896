# 06 — Gerar candidatos não rotulados das entidades previstas

**What to build:** transformar previsões de qualquer reconhecedor compatível em pares candidatos auditáveis, usando somente tipos previstos, nó lógico `Patient` e distância entre sentenças, sem consultar endpoints ou relações gold.

**Blocked by:** 03 — Executar o braço de entidades por dicionário canônico; 04 — Executar o braço de entidades por semi-Markov CRF.

**Status:** ready-for-agent

- [ ] Entradas do dicionário e do CRF produzem o mesmo contrato de candidatos.
- [ ] `HAS_CONDITION`, `HAS_FINDING`, `HAS_FACTOR`, `NEGATES` e `CAUSES` respeitam compatibilidade e direção de tipos.
- [ ] O nó lógico `Patient` é criado deterministicamente sem copiar uma entidade gold.
- [ ] A janela de até uma sentença usa o segmentador versionado e registra a distância calculada.
- [ ] O artefato não contém `label`, `gold_relation_id` ou qualquer correção baseada no gold.
- [ ] IDs, hashes, contagens e configuração são determinísticos e registrados no manifesto.

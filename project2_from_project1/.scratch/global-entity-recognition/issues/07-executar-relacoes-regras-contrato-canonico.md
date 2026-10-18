# 07 — Executar relações por regras no contrato canônico

**What to build:** criar um baseline experimental de regras que consome candidatos não rotulados canônicos e produz exatamente uma previsão compatível por candidato, independentemente do extrator legado e sem usar respostas gold na inferência.

**Blocked by:** 06 — Gerar candidatos não rotulados das entidades previstas.

**Status:** ready-for-agent

- [ ] As regras são definidas e ajustadas apenas com documentos de treino e usam os cinco rótulos supervisionados ou `NO_RELATION`.
- [ ] Cada candidato recebe exatamente uma previsão compatível com sua hipótese e direção.
- [ ] Evidências identificam a regra disparada e nunca incluem texto clínico no manifesto.
- [ ] O baseline executa sobre candidatos vindos tanto do dicionário quanto do CRF.
- [ ] Predições, tempos, versão do método e hashes são serializados pelo contrato comum.

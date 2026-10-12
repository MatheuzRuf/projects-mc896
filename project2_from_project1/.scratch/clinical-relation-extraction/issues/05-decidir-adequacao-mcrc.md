# 05 — Decidir a adequação do MCRC

**What to build:** executar conversão e auditoria no MCRC candidato e emitir uma decisão explícita e reproduzível sobre sua adequação ao experimento supervisionado.

**Blocked by:** 03 — Converter uma distribuição MCRC verificada; 04 — Auditar um corpus contra a coleção MultiCaRe protegida.

**Status:** ready-for-agent

- [ ] O corpus convertido passa pelos validadores do contrato comum.
- [ ] O relatório apresenta documentos e relações antes e depois de exclusões e quarentenas.
- [ ] O relatório apresenta suporte por classe nativa e canônica, incluindo relações minoritárias.
- [ ] A decisão considera cobertura, qualidade estrutural, impacto do leakage e viabilidade da avaliação sem rejeitar o corpus apenas por desbalanceamento.
- [ ] O resultado é registrado como aprovado ou rejeitado, acompanhado das razões e evidências.
- [ ] Se rejeitado, nenhum split de treino é produzido e a necessidade de obter i2b2/n2c2 é indicada claramente.
- [ ] Se aprovado, somente documentos liberados são encaminhados ao congelamento das partições.


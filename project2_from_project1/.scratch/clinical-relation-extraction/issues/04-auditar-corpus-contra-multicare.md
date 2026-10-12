# 04 — Auditar um corpus contra a coleção MultiCaRe protegida

**What to build:** comparar um corpus externo normalizado com a coleção MultiCaRe protegida e produzir decisões auditáveis de manutenção, exclusão ou quarentena antes de qualquer treino.

**Blocked by:** 02 — Preparar e recarregar um documento clínico completo.

**Status:** ready-for-agent

- [ ] A coleção protegida representa os 50 artigos MultiCaRe atuais sem expor seus textos no controle de versão.
- [ ] Coincidências por PMCID, PMID, DOI ou hash de conteúdo resultam em exclusão do artigo externo.
- [ ] A auditoria compara documentos completos e trechos para detectar sobreposição textual sem identificador comum.
- [ ] Similaridade textual alta sem evidência decisiva resulta em quarentena, não em inclusão automática.
- [ ] Documentos excluídos ou em quarentena não podem ser liberados para treino pelo fluxo normal.
- [ ] O relatório apresenta evidência, regra aplicada e decisão para cada caso sinalizado.
- [ ] Fixtures cobrem coincidência exata, similaridade parcial, quarentena e ausência de sobreposição.


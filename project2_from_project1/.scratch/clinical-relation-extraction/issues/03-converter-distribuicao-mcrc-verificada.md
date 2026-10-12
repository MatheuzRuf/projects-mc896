# 03 — Converter uma distribuição MCRC verificada

**What to build:** converter uma distribuição MCRC explicitamente autorizada para o contrato seguro do projeto, exigindo verificação de origem e integridade e produzindo um manifesto reproduzível.

**Blocked by:** 02 — Preparar e recarregar um documento clínico completo.

**Status:** ready-for-agent

- [ ] A conversão exige versão ou commit de origem e SHA-256 esperado do artefato bruto.
- [ ] Um hash divergente interrompe a conversão antes da desserialização.
- [ ] A abertura do artefato serializado exige uma ação explícita e não ocorre no fluxo operacional comum.
- [ ] Documentos, menções descontínuas, aninhadas ou multilabel e relações intersentenciais são preservados no JSONL.
- [ ] O mapeamento nativo-canônico segue o vocabulário e as decisões arquiteturais aprovadas.
- [ ] Execuções equivalentes produzem JSONL e manifesto determinísticos.
- [ ] O manifesto registra origem, versão, hashes, contagens e resultado da conversão sem incorporar o corpus bruto.


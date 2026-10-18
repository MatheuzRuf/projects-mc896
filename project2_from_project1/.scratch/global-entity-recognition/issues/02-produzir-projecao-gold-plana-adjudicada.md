# 02 — Produzir a projeção gold plana adjudicada

> [!IMPORTANT]
> Antes de iniciar este ticket, troque para um modelo com alto nível de reasoning. A adjudicação exige decisões semânticas cuidadosas e auditáveis.

**What to build:** transformar o corpus lossless em uma projeção gold plana, separada e reproduzível, usando tokenização com offsets e segmentação de sentenças protegida contra abreviações. Duplicatas exatas devem ser colapsadas; sobreposições, conflitos de tipo e desalinhamentos devem receber duas revisões independentes, consenso ou terceiro voto e justificativa registrada.

**Blocked by:** 01 — Expandir os contratos supervisionados versionados.

**Status:** blocked-human-review

**Implementation note (2026-10-18):** o fluxo versionado, os validadores e o
pacote local de revisão estão implementados. O pacote atual contém 233 casos e
está vinculado ao corpus `5104d36f…a9b` e ao plano `fecd718e…3e57`. O ticket
permanece aberto até dois revisores humanos independentes concluírem os votos,
resolverem divergências e produzirem `annotations/entity-gold-v1/adjudication.jsonl`.

- [ ] O corpus lossless permanece byte a byte inalterado e a projeção é produzida em artefato separado.
- [ ] Tokens e sentenças preservam offsets e não criam falsos limites nas abreviações e unidades auditadas.
- [ ] Duplicatas exatas são colapsadas deterministicamente sem esconder suas origens.
- [ ] Cada conflito registra decisões independentes, consenso ou terceiro voto, justificativa e proveniência.
- [ ] Toda menção resultante é contígua, não sobreposta, possui um tipo canônico e no máximo 40 tokens.
- [ ] O manifesto registra hashes de entrada e saída, contagens, transformações e casos rejeitados sem conter texto clínico.

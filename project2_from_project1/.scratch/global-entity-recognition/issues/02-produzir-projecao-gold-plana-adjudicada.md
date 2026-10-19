# 02 — Produzir a projeção gold plana adjudicada

> [!IMPORTANT]
> Antes de iniciar este ticket, troque para um modelo com alto nível de reasoning. A adjudicação exige decisões semânticas cuidadosas e auditáveis.

**What to build:** transformar o corpus lossless em uma projeção gold plana, separada e reproduzível, usando tokenização com offsets e segmentação de sentenças protegida contra abreviações. Duplicatas exatas devem ser colapsadas; sobreposições, conflitos de tipo e desalinhamentos devem receber duas revisões independentes, consenso ou terceiro voto e justificativa registrada.

**Blocked by:** 01 — Expandir os contratos supervisionados versionados.

**Status:** done

**Implementation note (2026-10-18):** a revisão humana foi considerada
inviável e a adjudicação foi feita por modelos (`reviewer_kind = model`), sob
`annotations/entity-gold-v1/GUIDELINES.md`. Dois revisores independentes (A e
B) concordaram em 230 dos 233 casos. Os 3 casos divergentes receberam um
terceiro voto independente (C). `annotations/entity-gold-v1/adjudication.jsonl`
está vinculado ao corpus `28844ee5…` (a mesma fonte dos folds congelados) e ao
plano `fecd718e…`. A projeção resultante tem 3.063 menções em 53 documentos.
Como os revisores são execuções do mesmo modelo, a concordância não equivale a
concordância entre anotadores humanos.

- [x] O corpus lossless permanece byte a byte inalterado e a projeção é produzida em artefato separado.
- [x] Tokens e sentenças preservam offsets e não criam falsos limites nas abreviações e unidades auditadas.
- [x] Duplicatas exatas são colapsadas deterministicamente sem esconder suas origens.
- [x] Cada conflito registra decisões independentes, consenso ou terceiro voto, justificativa e proveniência.
- [x] Toda menção resultante é contígua, não sobreposta, possui um tipo canônico e no máximo 40 tokens.
- [x] O manifesto registra hashes de entrada e saída, contagens, transformações e casos rejeitados sem conter texto clínico.

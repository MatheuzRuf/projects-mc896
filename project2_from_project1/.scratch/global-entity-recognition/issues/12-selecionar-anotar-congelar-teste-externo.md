# 12 — Selecionar, anotar e congelar o teste externo

**What to build:** criar um conjunto externo intocado de relatos clínicos abertos, auditado contra os dados existentes e anotado sob os mesmos tipos e relações, congelando-o somente quando a regra pré-registrada de documentos e suporte tiver sido aplicada.

**Blocked by:** 02 — Produzir a projeção gold plana adjudicada.

**Status:** in-progress — pausado pelo usuário após A110/B70/C75; prefixo comum de 70 candidatos ainda não congelável

- [x] A seleção parte de uma regra documental definida antes de observar resultados dos modelos e exclui sobreposições ou quarentenas.
- [x] Pelo menos 50 novos documentos e 100 entidades de cada tipo são anotados com revisão independente e proveniência.
- [ ] A amostragem continua sequencialmente até 20 relações positivas por classe ou o teto de 100 documentos.
- [ ] Classes abaixo do suporte no teto são marcadas previamente como inconclusivas.
- [ ] O conjunto é congelado com IDs, hashes, esquema, relatório de leakage e manifesto antes da primeira execução dos modelos.
- [x] Nenhuma decisão de modelo, feature ou hiperparâmetro usa as anotações desse conjunto.

Checkpoint: `docs/external-review-checkpoint.md`. O prefixo comum adjudicado tem
63 documentos elegíveis e 9 `CAUSES`; a regra exige continuar até 20 `CAUSES`
ou 100 documentos elegíveis. A issue 13 permanece bloqueada.

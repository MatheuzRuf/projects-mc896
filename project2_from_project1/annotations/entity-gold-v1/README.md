# Adjudicação da projeção gold de entidades

O arquivo final `adjudication.jsonl` será criado depois de duas revisões humanas
independentes dos casos produzidos por `build_projection_plan`. Ele não está
presente enquanto essa etapa de dados estiver pendente; resultados heurísticos
ou gerados por modelo não constituem o gold.

Cada linha deve seguir `AdjudicationRecord` e conter:

- dois votos humanos iniciais, feitos sem acesso ao voto do outro revisor;
- uma justificativa específica em cada voto;
- consenso confirmado pelos dois revisores ou exatamente um terceiro voto
  humano independente;
- o vínculo retornado por `build_adjudication_binding`, que fixa fonte, plano,
  versões de segmentação e limite de tokens.

O procedimento e a reprodução estão descritos em
[`docs/gold-entity-projection.md`](../../docs/gold-entity-projection.md).

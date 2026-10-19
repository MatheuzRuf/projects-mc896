# Adjudicação da projeção gold de entidades

## Decisão: revisão por modelo (2026-10-18)

A revisão humana independente dos 233 casos foi considerada inviável pelo
time. A adjudicação desta versão foi feita por **modelos de linguagem**
(`reviewer_kind = model`) sob o protocolo abaixo. Os resultados que dependem de
`flat-gold-v1` devem declarar que a projeção plana foi adjudicada por modelos.
O corpus lossless original, anotado por humanos, permanece inalterado.

## Protocolo

1. O pacote de revisão é gerado do `mcrc.jsonl` com SHA-256 `28844ee5…`, a
   mesma fonte dos folds congelados.
2. Todos os revisores seguem o mesmo [`GUIDELINES.md`](GUIDELINES.md), fixado
   antes das revisões.
3. Os revisores A e B são execuções separadas de modelo com alto nível de
   reasoning, sem acesso aos votos um do outro. Os casos são divididos em
   partes por conveniência, mas cada papel mantém o mesmo identificador.
4. Os casos em que A e B divergem recebem um terceiro voto independente (C),
   também sem acesso aos votos anteriores.
5. `scripts/model_adjudication.py collect` valida cada revisão: cobertura,
   aliases, seleção plana e ausência de texto clínico nas justificativas.
   `scripts/assemble_gold_adjudication.py` monta e valida o arquivo final.

Cada linha de `adjudication.jsonl` segue `AdjudicationRecord`: dois votos,
consenso ou terceiro voto, justificativas e o vínculo com fonte, plano,
versões de segmentação e limite de tokens. As justificativas não reproduzem
texto clínico.

## Resultado

- 233 casos: A e B concordaram em 230; os 3 restantes foram decididos pelo
  terceiro voto.
- Os três papéis são execuções separadas do mesmo modelo sob o mesmo guia. A
  concordância de 98,7% mede consistência do protocolo, não concordância entre
  anotadores humanos, e erros sistemáticos do modelo podem ser compartilhados.

O procedimento e a reprodução estão descritos em
[`docs/gold-entity-projection.md`](../../docs/gold-entity-projection.md).

# Avaliação externa entregue

Em 2026-10-19 foram treinados e congelados os componentes exclusivamente no
conjunto de desenvolvimento reconciliado e executados uma vez sobre o snapshot
externo fornecido. Por decisão explícita do usuário, este snapshot de 63
documentos elegíveis é considerado o escopo concluído desta entrega, embora a
revisão registrada em `external-review-checkpoint.md` ainda não satisfaça a
regra original de congelamento de 100 documentos elegíveis ou 20 relações por
classe. Portanto, os resultados não devem ser apresentados como uma validação
externa definitiva pré-registrada.

## Proveniência

- Modelos: `artifacts/external-final-models-v1/manifest.json`, status
  `models_frozen_before_external_annotation_access`; 42 documentos de
  desenvolvimento, dicionário, regras, k-NN unigrama com `k=1` e CRF com
  `l2=0.1`.
- Snapshot: `artifacts/external-final-snapshot-v1/manifest.json`; 63 dos 70
  candidatos adjudicados. Ele registra tanto o hash do checkpoint quanto o da
  triagem de elegibilidade posteriormente ampliada.
- Resultado: `artifacts/external-final-evaluation-v1/report.json` e
  `manifest.json`. Todos os hashes de artefatos listados no manifesto foram
  verificados após a execução.

O CRF fez 250 iterações e 284 avaliações de função em 1.643,679 segundos. A
execução foi finita, mas não convergiu pelo critério configurado; esse braço
permanece visível sem substituição por fallback. As predições de dicionário e
CRF foram gravadas antes da abertura do gold. A pontuação posterior reutilizou
essas predições congeladas após a primeira passagem revelar uma incompatibilidade
do conversor com o formato de gold já adjudicado.

## Resultados estritos

| Braço | Papel | F1 macro de entidades | F1 macro de relações | F1 micro de arestas conceituais |
| --- | --- | ---: | ---: | ---: |
| dicionário + regras | fim a fim | 0,1373 | 0,0458 | 0,0578 |
| dicionário + k-NN | fim a fim | 0,1373 | 0,0601 | 0,0724 |
| CRF + regras | fim a fim; CRF não convergido finitamente | 0,2415 | 0,1078 | 0,0941 |
| CRF + k-NN | fim a fim; CRF não convergido finitamente | 0,2415 | 0,1123 | 0,1153 |
| gold-entity + regras | teto diagnóstico | — | 0,3265 | — |
| gold-entity + k-NN | teto diagnóstico | — | 0,6053 | — |

O teto gold-entity é somente diagnóstico e não é comparável a um sistema fim a
fim. O relatório JSON contém as métricas por classe e a decomposição de falsos
positivos e negativos para cada braço.

`CAUSES` tem somente 9 relações no snapshot e é marcado como inconclusivo em
todos os braços. O conversor flat-gold também registrou 8 menções que cruzam
fronteiras de sentença; elas não entram na métrica estrita de entidades e suas
8 relações `HAS_FINDING` aparecem explicitamente como perdas de projeção do
grafo, sem modificar o gold de origem.

## Reprodução

Os comandos públicos são `prepare_external_models.py`,
`prepare_external_snapshot.py` e `run_external_evaluation.py`. O último aceita
`--resume-frozen-predictions` apenas para concluir uma pontuação interrompida:
ele verifica os hashes de `predictions-manifest.json`, do snapshot e do
manifesto de modelos, e não executa novamente a inferência fim a fim.

# Resultado inicial do pipeline n-gramas + k-NN

Execução realizada com a distribuição oficial do MCRC na revisão
`23883361353ecf123f180ac2f4ef45ef17277ca2`, após verificação SHA-256,
auditoria contra os 50 artigos MultiCaRe protegidos e congelamento documental
com semente 42.

## Dados liberados

- 53 documentos, sem exclusões ou quarentenas;
- 4.294 menções;
- 2.906 relações supervisionadas;
- 47 `CAUSES`;
- 155 `HAS_CONDITION`;
- 288 `HAS_FACTOR`;
- 1.412 `HAS_FINDING`;
- 1.004 `NEGATES`.

O dataset usa teste fixo de 20%, cinco folds nos demais documentos, janela
máxima de uma sentença e até dois negativos de treino por positivo. Três
relações `CAUSES` ficaram fora dessa janela e foram registradas no manifesto.

## Validação cruzada inicial

Configuração: contagens de unigramas, bigramas e trigramas, matriz CSR,
distância cosseno, votação uniforme, classificador binário por relação e
`k=3`.

| Fold | Macro-F1 positiva | `CAUSES` | `HAS_CONDITION` | `HAS_FACTOR` | `HAS_FINDING` | `NEGATES` |
|---|---:|---:|---:|---:|---:|---:|
| 01 | 0,4621 | 0,0000 | 0,6122 | 0,7711 | 0,4519 | 0,4753 |
| 02 | 0,5616 | 0,2727 | 0,5091 | 0,8889 | 0,5659 | 0,5714 |
| 03 | 0,3756 | 0,2857 | 0,4545 | 0,4844 | 0,2937 | 0,3599 |
| 04 | 0,4089 | 0,0000 | 0,5333 | 0,6612 | 0,4505 | 0,3996 |
| 05 | 0,4764 | 0,4706 | 0,3415 | 0,6306 | 0,3647 | 0,5747 |

A média da macro-F1 positiva foi **0,4569**, com desvio-padrão populacional de
**0,0637**. O baseline majoritário, que sempre responde `NO_RELATION`, tem
macro-F1 positiva igual a zero.

Este resultado demonstra que o pipeline completo produz sinal útil, mas não é
uma seleção final de modelo. `CAUSES` permanece inconclusiva em folds com
suporte inferior a cinco e deve motivar busca de hiperparâmetros estratificada
e, se necessário, expansão do corpus com mapeamento semântico explícito.

O teste fixo não foi usado nesta análise e deve permanecer fechado até a
configuração ser escolhida somente pela validação cruzada.

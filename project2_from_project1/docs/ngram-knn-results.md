# Resultados do pipeline n-gramas + k-NN

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

## Seleção de hiperparâmetros

Foram avaliadas nove configurações nos mesmos cinco folds, sem leitura do
teste fixo. A métrica de seleção foi a média da macro-F1 das cinco relações
positivas.

| Ordem máxima | k | Macro-F1 média | Desvio-padrão |
|---:|---:|---:|---:|
| 1 | 1 | **0,4592** | 0,0877 |
| 1 | 3 | 0,4562 | 0,0559 |
| 1 | 5 | 0,4543 | 0,0551 |
| 2 | 1 | 0,4507 | 0,0947 |
| 2 | 3 | 0,4590 | 0,0603 |
| 2 | 5 | 0,4383 | 0,0532 |
| 3 | 1 | 0,4558 | 0,0947 |
| 3 | 3 | 0,4569 | 0,0637 |
| 3 | 5 | 0,4374 | 0,0556 |

A configuração fixada foi somente unigramas e `k=1`. A vantagem sobre
bigramas com `k=3` foi pequena (cerca de 0,0002 de macro-F1), portanto não há
evidência forte de que uma delas seja intrinsecamente superior; a regra
determinística selecionou a maior média observada.

## Avaliação final no teste fixo

Depois da seleção, os cinco folds de validação disjuntos foram unidos para
reconstruir os 42 documentos de desenvolvimento. O treino final reteve 2.242
positivos e amostrou 4.484 negativos com semente 42. A configuração escolhida
foi então avaliada uma única vez nos 3.514 candidatos do teste.

| Classe | Suporte | Precisão | Recall | F1 |
|---|---:|---:|---:|---:|
| `CAUSES` | 20 | 0,0625 | 0,2000 | 0,0952 |
| `HAS_CONDITION` | 46 | 0,0816 | 0,6087 | 0,1440 |
| `HAS_FACTOR` | 98 | 0,2500 | 0,8571 | 0,3871 |
| `HAS_FINDING` | 335 | 0,2855 | 0,6060 | 0,3881 |
| `NEGATES` | 162 | 0,4079 | 0,7654 | 0,5322 |
| `NO_RELATION` | 2.853 | 0,8759 | 0,5391 | 0,6674 |

A macro-F1 positiva final foi **0,3093**, queda de aproximadamente 0,1498 em
relação à média de validação. O modelo recupera muitos positivos, mas gera
muitos falsos positivos; `CAUSES` e `HAS_CONDITION` são os casos mais fracos.
O teste agora está consumido e não deve ser reutilizado para escolher novos
hiperparâmetros. Melhorias posteriores devem ser decididas nos folds de
desenvolvimento ou em uma base ampliada e avaliadas em um novo teste externo.

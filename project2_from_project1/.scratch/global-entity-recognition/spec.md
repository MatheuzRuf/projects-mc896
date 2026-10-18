# Reconhecimento global de entidades e pipeline clínico end-to-end

Status: ready-for-agent

## Problem Statement

O pipeline supervisionado atual classifica relações com n-gramas e k-NN, mas
recebe entidades gold. Portanto, seus resultados não demonstram que o sistema
consiga partir de texto clínico bruto, reconhecer as menções, decidir quais
palavras devem ser tratadas conjuntamente, classificar as relações e produzir
um Knowledge Graph sem usar a resposta correta como entrada.

O extrator legado reconhece entidades por dicionário e agrega ocorrências antes
da classificação de relações. Esse comportamento é útil como baseline, mas não
aprende contexto, não generaliza para formas não registradas e não preserva a
identidade de cada menção durante a avaliação end-to-end. Seus tipos também não
correspondem integralmente aos quatro tipos supervisionados pelo MCRC.

O projeto é um estudo comparativo. O objetivo não é promover antecipadamente o
melhor modelo para produção, mas executar e comparar pipelines completos sob os
mesmos contratos. O semi-Markov CRF deve permanecer no estudo mesmo que seu
resultado seja inferior ao dicionário. Resultados fracos, falhas de
convergência finitas e propagação de erros entre entidades, relações e grafo são
resultados científicos que precisam continuar observáveis.

O MCRC preserva duplicidades, sobreposições e conflitos de tipo que não podem
ser consumidos diretamente por uma segmentação plana. Além disso, o teste MCRC
fixo já foi usado anteriormente e não pode sustentar uma nova conclusão final
imparcial. O trabalho precisa criar uma projeção gold plana auditável, evitar
leakage em todas as decisões, executar validação cruzada aninhada e preparar um
novo teste externo congelado.

## Solution

Construir um pipeline supervisionado end-to-end que recebe texto clínico bruto
e produz previsões estruturadas e um Knowledge Graph versionado. O primeiro
reconhecedor aprendido será um semi-Markov CRF linear, capaz de escolher
globalmente uma segmentação plana e não sobreposta. Cada menção prevista terá
exatamente um entre `Condition`, `Finding`, `Factor` e `Modifier`; palavras que
não pertencem a entidades serão representadas por segmentos externos
unitários.

O gold original permanecerá lossless e imutável. Uma projeção gold plana será
derivada em artefato separado: duplicatas exatas serão colapsadas e conflitos
de sobreposição, tipo ou alinhamento serão decididos por dois revisores
independentes, com consenso ou terceiro voto. O ticket dessa adjudicação deverá
começar com um aviso explícito para usar um modelo de alto nível de reasoning.

O texto será segmentado por um tokenizador determinístico de palavras, números
e pontuação que preserve offsets. O separador de sentenças protegerá iniciais,
abreviações clínicas e unidades. Menções terão no máximo 40 tokens; violações
serão diagnosticadas, nunca truncadas silenciosamente.

O semi-Markov CRF usará features categóricas esparsas equivalentes a one-hot,
sem embeddings no braço principal. O vocabulário será aprendido somente no
treino de cada rodada, com frequência lexical mínima dois. O score combinará
features de segmentos, transições entre tipos e parâmetros de início e fim. A
implementação usará PyTorch em CPU, programação dinâmica pertencente ao
projeto, `logsumexp`, autograd, L-BFGS full-batch, inicialização em zero e
regularização L2.

O desenvolvimento usará cinco folds externos agrupados por documento. Em cada
fold externo, três folds internos selecionarão `lambda` L2 entre `0.01`, `0.1`,
`1` e `10`, usando macro-F1 estrita de entidades. Empates favorecerão menor
variância e depois regularização mais forte. O baseline de dicionário canônico
será reconstruído exclusivamente com formas do treino de cada fold, sem
consultar validação, teste, dicionários legados ou fontes externas.

Cada reconhecedor produzirá `EntityPrediction` em JSONL separado do gold. Os
pares compatíveis gerados exclusivamente dessas previsões serão serializados
como `RelationCandidate`, sem campo de rótulo. Regras e classificadores de
relações serão configurados ou treinados sobre entidades gold dos documentos
de treino e receberão entidades previstas durante a inferência end-to-end. O
mesmo classificador de relações será mantido entre reconhecedores para permitir
atribuição fatorial dos resultados.

O estudo comparará entidades de dicionário e semi-Markov CRF, relações por
regras e k-NN, e todas as combinações compatíveis no pipeline end-to-end. Um
método híbrido poderá entrar na mesma matriz quando possuir desenho e
implementação próprios sob o contrato comum. Entidades gold serão usadas como
teto experimental para relações, nunca como um braço end-to-end.

As relações serão previstas entre menções antes de qualquer consolidação. O
grafo canônico terá uma camada de menção e uma camada conceitual posterior. Um
conceito será identificado por documento, tipo canônico e superfície
normalizada por Unicode, `casefold` e espaços consolidados, sem stemming,
sinônimos ou ontologia externa. O contrato principal do grafo será JSONL
versionado; CSV e NetworkX serão adaptações de exportação e visualização.

A avaliação principal exigirá offsets e tipos exatos para entidades, e
endpoints, tipos e rótulos exatos para relações. Serão publicados micro-F1,
macro-F1, métricas por classe, diagnósticos de sobreposição, resultados por OOV
e comprimento, métricas secundárias de nós e arestas conceituais e uma análise
que separe erros de reconhecimento, relações e projeção. Nenhum limiar de
desempenho removerá um braço experimental.

Toda execução produzirá manifesto com commit, hashes, partições documentais,
configuração, dependências, determinismo, convergência, duração e hashes das
saídas. Uma execução finita que alcance 250 iterações será avaliada com
`converged=false`; `NaN` ou infinito interromperá apenas a execução afetada e
gerará diagnóstico, sem troca silenciosa de otimizador.

A conclusão final dependerá de um novo teste externo: no mínimo 50 novos
relatos clínicos abertos, 100 entidades de cada tipo e amostragem sequencial até
20 relações positivas de cada classe ou o teto pré-registrado de 100
documentos. Classes ainda abaixo do suporte serão declaradas inconclusivas.

## User Stories

1. Como pesquisador, quero reconhecer entidades a partir de texto bruto, para que a avaliação deixe de depender de entidades gold durante a inferência.

2. Como pesquisador, quero que o modelo escolha globalmente quais tokens formam cada menção, para que fronteiras e tipos não sejam decididos como classificações isoladas.

3. Como pesquisador, quero executar o semi-Markov CRF até o final mesmo quando seu resultado for ruim, para preservar comparações honestas entre métodos.

4. Como pesquisador, quero manter o dicionário como baseline independente, para comparar aprendizado contextual com correspondência lexical.

5. Como pesquisador, quero que o CRF não use correspondências do dicionário como features, para evitar uma dependência oculta entre os braços.

6. Como pesquisador, quero usar exatamente os tipos `Condition`, `Finding`, `Factor` e `Modifier` no primeiro braço, para aprender somente distinções sustentadas pelo MCRC.

7. Como consumidor do grafo, quero que cada menção prevista possua um único tipo canônico, para obter nós interpretáveis e sem tipos concorrentes.

8. Como consumidor do grafo, quero permitir várias relações para o mesmo nó, para representar multiplicidade clínica nas arestas em vez de nos tipos de entidade.

9. Como curador, quero preservar integralmente as anotações MCRC originais, para que a preparação do modelo não apague duplicidades, sobreposições ou conflitos.

10. Como curador, quero uma projeção gold plana separada, para treinar o reconhecedor sem apresentar essa projeção como anotação original.

11. Como revisor, quero que duplicatas exatas sejam colapsadas deterministicamente, para não revisar casos semanticamente idênticos.

12. Como revisor, quero adjudicar conflitos por duas revisões independentes, para reduzir decisões individuais não auditadas.

13. Como revisor, quero registrar consenso, terceiro voto, justificativa e proveniência, para reconstruir cada decisão da projeção gold.

14. Como responsável pelos tickets, quero um aviso de modelo com alto reasoning no ticket de adjudicação, para que a tarefa semântica receba capacidade adequada.

15. Como desenvolvedor, quero tokens com offsets no texto original, para converter previsões em spans exatos e auditáveis.

16. Como desenvolvedor, quero separar palavras, números e pontuação deterministicamente, para alinhar a grande maioria das anotações sem destruir o texto.

17. Como desenvolvedor, quero diagnosticar os 31 alvos não alinhados, para que nenhum intervalo seja ajustado silenciosamente.

18. Como pesquisador, quero proteger abreviações clínicas e unidades durante a segmentação de sentenças, para não criar falsos limites em expressões como nomes de organismos e doses.

19. Como operador, quero receber erro explícito quando uma menção exceder 40 tokens, para não truncar entidades sem registro.

20. Como pesquisador, quero features esparsas equivalentes a one-hot, para manter o primeiro modelo interpretável e metodologicamente separado de embeddings.

21. Como pesquisador, quero aprender o vocabulário de features somente no treino, para impedir leakage lexical.

22. Como pesquisador, quero ignorar features lexicais desconhecidas e preservar sinais ortográficos, para tratar OOV sem hashing ou consulta ao teste.

23. Como desenvolvedor, quero programação dinâmica segmental numericamente estável, para calcular partição, Viterbi e marginais sem overflow.

24. Como pesquisador, quero comparar marginais segmentais sem chamá-las de confiança calibrada, para analisar incerteza sem exagerar sua interpretação.

25. Como pesquisador, quero que o dicionário deixe a marginal nula, para não inventar scores incomparáveis.

26. Como pesquisador, quero treinamento determinístico com L-BFGS e inicialização em zero, para reduzir variação irrelevante no modelo linear.

27. Como pesquisador, quero selecionar L2 por validação cruzada aninhada e documental, para não usar o fold externo na escolha de hiperparâmetros.

28. Como pesquisador, quero uma grade pequena e pré-registrada, para limitar ajuste excessivo aos 42 documentos de desenvolvimento.

29. Como pesquisador, quero preservar a distribuição natural das sequências no braço principal, para que reamostragem ou pesos não redefinam silenciosamente a likelihood.

30. Como pesquisador, quero declarar qualquer variante ponderada como outro braço, para comparar a alteração sem substituir o modelo principal.

31. Como pesquisador, quero construir o dicionário canônico somente com menções de treino, para obter um baseline lexical sem leakage.

32. Como pesquisador, quero que ambiguidades lexicais sejam resolvidas pela frequência do treino e que empates causem abstenção, para evitar prioridade arbitrária por classe.

33. Como desenvolvedor, quero serializar entidades previstas separadamente do gold, para não inventar tipos nativos ou misturar respostas e previsões.

34. Como desenvolvedor, quero IDs determinísticos para previsões e candidatos, para ligar artefatos entre estágios e reproduzir comparações.

35. Como desenvolvedor, quero candidatos de relação sem campo `label`, para não confundir rótulo desconhecido com `NO_RELATION`.

36. Como pesquisador, quero gerar candidatos somente das entidades previstas no end-to-end, para medir integralmente a propagação de erros.

37. Como pesquisador, quero criar o nó lógico `Patient` deterministicamente, para representar as relações `HAS_*` sem usar uma menção gold inexistente no texto.

38. Como pesquisador, quero aplicar o mesmo classificador de relações a cada reconhecedor, para atribuir diferenças à etapa de entidades.

39. Como pesquisador, quero avaliar relações sobre entidades gold como teto experimental, para separar capacidade relacional de erros do reconhecedor.

40. Como pesquisador, quero comparar dicionário e CRF com regras e k-NN sob o mesmo contrato, para obter uma matriz fatorial interpretável.

41. Como implementador futuro, quero que o contrato aceite um método híbrido, para adicioná-lo posteriormente sem alterar o avaliador.

42. Como pesquisador, quero correspondência estrita de offsets e tipo como métrica principal de entidades, para não esconder fronteiras erradas.

43. Como pesquisador, quero relações corretas somente com endpoints, tipos e rótulo exatos, para medir o pipeline realmente entregue ao grafo.

44. Como pesquisador, quero métricas de sobreposição apenas como diagnóstico, para analisar erros parciais sem substituir o critério principal.

45. Como pesquisador, quero micro-F1, macro-F1 e F1 por classe, para tornar visível o desempenho dos tipos e relações minoritários.

46. Como pesquisador, quero decompor erros em ausentes, espúrios, tipo incorreto e limite incorreto, para explicar diferenças entre reconhecedores.

47. Como pesquisador, quero estratificar resultados por OOV e comprimento, para entender quando a representação esparsa falha.

48. Como pesquisador, quero distinguir erros herdados das entidades de erros próprios das relações, para não atribuir toda falha ao classificador final.

49. Como pesquisador, quero manter uma camada de menção no grafo, para preservar os endpoints exatos usados pelas relações.

50. Como consumidor do grafo, quero uma camada conceitual posterior, para reunir ocorrências compatíveis sem perder sua proveniência textual.

51. Como consumidor do grafo, quero que tipo faça parte da identidade conceitual, para não fundir superfícies iguais que representam categorias diferentes.

52. Como consumidor do grafo, quero normalização conservadora sem sinônimos automáticos, para não apresentar equivalência clínica não supervisionada.

53. Como pesquisador, quero comparar nós e arestas conceituais projetados do gold e das previsões, para medir também o resultado estrutural final.

54. Como desenvolvedor, quero JSONL versionado como fonte de verdade do grafo, para preservar estruturas e proveniência de forma validável.

55. Como usuário legado, quero exportações adaptadas para CSV e NetworkX, para manter visualização e compatibilidade sem limitar o contrato canônico.

56. Como operador, quero um manifesto completo por execução, para reproduzir dados, configuração, dependências e saídas.

57. Como pesquisador, quero avaliar uma execução finita que não convergiu, para não esconder um resultado negativo relevante.

58. Como operador, quero interromper e diagnosticar `NaN` ou infinito, para impedir artefatos numericamente inválidos.

59. Como pesquisador, quero que nenhum fallback de otimizador ou dicionário substitua silenciosamente o CRF, para manter a identidade do braço experimental.

60. Como pesquisador, quero que a conclusão operacional dependa de artefatos completos e contratos válidos, para não confundir F1 alto com pipeline funcional.

61. Como pesquisador, quero um novo teste externo congelado antes da execução, para obter uma conclusão não orientada pelo teste MCRC já consumido.

62. Como curador, quero auditar os novos documentos contra duplicidade e leakage, para manter independência em relação aos corpora anteriores.

63. Como pesquisador, quero suporte mínimo pré-registrado por tipo e relação, para declarar classes raras inconclusivas em vez de exagerar resultados.

64. Como responsável pelo estudo, quero expandir posteriormente todos os modelos para classificações legadas mais específicas, para melhorar a compreensão humana sem alegar equivalências não demonstradas.

65. Como mantenedor, quero que qualquer expansão semântica tenha nova supervisão ou mapeamento explícito, nova versão de esquema e nova avaliação, para preservar validade científica.

66. Como desenvolvedor, quero uma fixture sintética que atravesse o pipeline completo, para verificar a principal costura sem publicar texto clínico real.

67. Como desenvolvedor, quero comparar Viterbi, partição e marginais com enumeração exaustiva em sequências pequenas, para validar a matemática sem testar detalhes acidentais da implementação.

68. Como desenvolvedor, quero que os testes confirmem disjunção documental e vocabulários ajustados apenas no treino, para detectar leakage automaticamente.

69. Como revisor, quero artefatos determinísticos e validáveis por contrato, para auditar o estudo sem executar novamente todos os modelos.

70. Como usuário do projeto, quero um fluxo documentado do texto bruto ao grafo, para repetir a experiência em outra máquina e comparar todos os braços.

## Implementation Decisions

- O corpus lossless e a projeção gold plana serão artefatos diferentes. A projeção nunca sobrescreverá a fonte.
- A projeção plana aceitará somente menções contíguas, não sobrepostas e com um tipo canônico.
- Duplicatas exatas serão colapsadas automaticamente; todos os demais conflitos serão adjudicados e registrados.
- O ticket de adjudicação exigirá explicitamente modelo com alto nível de reasoning e duas revisões independentes.
- O inventário supervisionado inicial será `Condition`, `Finding`, `Factor`, `Modifier` e o estado externo.
- O reconhecedor aprendido será um semi-Markov CRF linear com inferência global de segmentos.
- Tokens externos formarão segmentos unitários para que cada saída possua uma única segmentação canônica.
- O limite de entidade será 40 tokens; violações gerarão diagnóstico.
- Tokenização e segmentação de sentenças serão determinísticas, offset-preserving e protegidas contra abreviações e unidades.
- As 31 anotações atualmente desalinhadas serão encaminhadas à adjudicação em vez de corrigidas automaticamente.
- Features incluirão identidade e forma de tokens, prefixos, sufixos, caixa, números, pontuação, contexto local e comprimento do segmento.
- O braço principal não usará embeddings, hashing, features de dicionário, pesos de classe ou reamostragem.
- O vocabulário de features será criado por fold de treino e exigirá frequência lexical mínima dois.
- O núcleo segmental será implementado no projeto sobre PyTorch em CPU, com autograd e operações `logsumexp` estáveis.
- O treino usará L-BFGS full-batch, inicialização em zero, no máximo 250 iterações e regularização L2.
- Cinco folds externos por documento estimarão desempenho; três folds internos selecionarão somente L2 na grade pré-registrada.
- Seleção usará macro-F1 estrita de entidades, com desempate por menor variância e regularização mais forte.
- Não convergência finita permanecerá avaliável e sinalizada; valores não finitos causarão falha estruturada sem fallback.
- O dicionário canônico será reconstruído por fold exclusivamente com superfícies gold de treino.
- No dicionário, o tipo mais frequente do treino resolverá uma superfície ambígua; empates causarão abstenção; correspondências mais longas terão prioridade.
- `EntityPrediction` será um contrato JSONL separado com documento, ID determinístico, span, tipo, sentença, método, versão e marginal opcional.
- A marginal segmental será nomeada `marginal_probability`, não será apresentada como calibrada e não controlará a decodificação Viterbi.
- `RelationCandidate` será um contrato JSONL não rotulado com endpoints previstos, tipos, hipótese relacional e distância entre sentenças.
- Candidatos end-to-end serão produzidos somente de previsões, compatibilidade de tipos, nó lógico `Patient` e janela máxima de uma sentença.
- Classificadores de relações serão ajustados sobre entidades gold dos documentos de treino e aplicados sem correção às entidades previstas.
- O treino sobre previsões out-of-fold será permitido apenas como variante futura explicitamente separada.
- A matriz principal combinará dicionário e semi-Markov CRF com cada método relacional compatível disponível.
- Regras e k-NN participarão sob o contrato comum. O método híbrido exige desenho próprio antes de entrar na matriz.
- O pipeline legado incompatível será apresentado somente como referência histórica.
- Predições de relações continuarão usando o contrato comum e preservarão método, versão e evidências.
- O grafo será construído depois das relações e manterá camadas de menção e conceito.
- A identidade conceitual será documento, tipo e superfície normalizada por Unicode, `casefold` e espaços consolidados.
- O primeiro pipeline não fará stemming, resolução de sinônimos nem vinculação a ontologias.
- O grafo canônico será JSONL versionado; exportações CSV e objetos NetworkX serão adaptadores posteriores.
- A avaliação principal de entidades exigirá span e tipo exatos.
- A avaliação principal de relações exigirá endpoints, tipos e rótulo exatos.
- Métricas de token, sobreposição e grafo conceitual serão diagnósticas ou secundárias.
- O relatório publicará micro-F1, macro-F1, F1 por classe, contagens e decomposição reproduzível dos erros.
- Todo braço permanecerá na comparação independentemente do desempenho.
- Cada execução produzirá manifesto com proveniência integral e hashes de todas as entradas e saídas.
- O teste MCRC consumido não orientará modelo, features ou hiperparâmetros do reconhecedor.
- O teste externo será congelado antes da execução e seguirá a regra de 50 a 100 documentos e suporte mínimo pré-registrado.
- Resultados anteriores ao teste externo serão identificados como exploratórios.
- A expansão para classificações legadas mais específicas ocorrerá somente depois do pipeline inicial e exigirá supervisão ou mapeamento semântico, nova versão e nova avaliação.

## Testing Decisions

- Um bom teste verificará comportamento observável nas interfaces públicas, nos artefatos serializados e nos manifestos, sem acoplar-se à organização interna das funções.
- A principal costura será um fluxo público completo que recebe uma fixture clínica sintética e configuração congelada, executa reconhecimento, candidatos, relações e projeção do grafo, e produz previsões, métricas e manifestos válidos.
- A costura completa exercitará tanto o dicionário quanto o semi-Markov CRF e pelo menos um método de relações compatível, sem substituir previsões por gold.
- O fluxo completo será considerado aprovado pela integridade dos contratos e artefatos, não por alcançar um F1 mínimo.
- Fixtures sintéticas pequenas evitarão versionar texto clínico real e permitirão resultados rápidos em CPU.
- Testes focados validarão tokenização e sentenças por offsets, abreviações, unidades, pontuação e spans no limite de 40 tokens.
- Testes focados validarão a projeção gold, incluindo colapso de duplicatas, preservação da fonte, trim auditável e recusa de conflitos sem adjudicação.
- Partição, Viterbi e marginais serão comparados com enumeração exaustiva em sequências artificiais pequenas.
- Testes numéricos cobrirão estabilidade por `logsumexp`, gradientes finitos, não convergência sinalizada e falha em `NaN` ou infinito.
- Testes de features verificarão frequência mínima, OOV, vocabulário ajustado apenas no treino e ausência de features de dicionário no CRF.
- Testes do dicionário verificarão construção somente pelo treino, tipo majoritário, abstenção em empate e preferência por correspondência mais longa.
- Testes de validação cruzada verificarão separação documental, três folds internos, grade L2 fixa e ausência de leitura do teste.
- Testes de contratos farão round-trip determinístico de `EntityPrediction`, `RelationCandidate`, predições de relações e grafo JSONL.
- Testes end-to-end confirmarão que candidatos são derivados apenas das previsões e que `NO_RELATION` não representa um rótulo desconhecido.
- Testes fatoriais confirmarão que o mesmo classificador de relações é usado entre reconhecedores e que cada combinação produz manifestos distintos.
- Testes de avaliação cobrirão entidades ausentes, espúrias, tipo incorreto, limite incorreto, OOV e comprimento.
- Testes de relações separarão gold-entity de end-to-end e exigirão correspondência exata dos endpoints.
- Testes do grafo verificarão camada de menção, agregação conceitual tipada, preservação de proveniência e não colisão entre superfícies iguais com tipos diferentes.
- Testes de exportação verificarão que CSV e NetworkX derivam do JSONL canônico sem se tornarem a fonte de verdade.
- Testes de manifesto verificarão commit, hashes, partições, configuração, versões, determinismo, duração e convergência sem incluir texto clínico.
- Como prior art, serão seguidos os testes pytest públicos que já atravessam preparação, avaliação, pipeline k-NN e exportação do grafo.

## Out of Scope

- Substituir ou modificar funcionalmente o `project1`.
- Usar entidades gold durante inferência end-to-end.
- Sobrepor previsões do CRF com fallback de dicionário.
- Remover um braço experimental por desempenho inferior.
- Treinar embeddings do zero ou incorporar embeddings biomédicos pré-treinados no primeiro braço.
- Ajustar transformers, LLMs ou outros modelos neurais de linguagem.
- Implementar o semi-Markov CRF por biblioteca estruturada externa que oculte a recorrência principal.
- Reamostrar sequências ou ponderar classes no braço principal.
- Inferir tipos legados mais específicos sem supervisão ou mapeamento semanticamente justificável.
- Vincular conceitos automaticamente a terminologias ou ontologias clínicas.
- Implementar o método híbrido de relações sem uma decisão de design própria.
- Usar i2b2/n2c2 como substituição automática para o novo teste externo.
- Apresentar o teste MCRC já consumido como confirmação final imparcial.
- Publicar textos clínicos, modelos grandes ou corpora protegidos no controle de versão.
- Calibrar empiricamente a marginal segmental como probabilidade de confiança.
- Transformar a expansão semântica posterior em requisito para concluir o primeiro pipeline.

## Further Notes

- O MCRC convertido possui 53 documentos e 4.294 registros de menção. A projeção preliminar contém 3.296 alvos únicos antes da adjudicação.
- Existem 269 pares de spans distintos sobrepostos, 68 grupos com o mesmo span e tipos diferentes e 31 alvos que não se alinham ao tokenizador proposto.
- O texto de todas as anotações coincide com os offsets do pickle verificado; os desalinhamentos pertencem à fonte, não foram introduzidos pelo conversor.
- As nove aparentes menções intersentenciais são falsos limites produzidos por abreviações e unidades.
- Sob a tokenização proposta, o maior span alinhado observado possui 35 tokens; o limite operacional de 40 oferece margem explícita.
- A projeção preliminar reúne os alvos em 2.632 conceitos tipados e encontra 63 superfícies documentais usadas por mais de um tipo.
- `CAUSES` possui somente 47 relações no corpus completo, justificando suporte mínimo por classe no novo teste externo.
- O ambiente local usa Windows e Python 3.14. A dependência PyTorch deverá ser fixada e registrada na etapa de implementação.
- A adjudicação humana e a construção do novo teste externo são atividades de dados, não atalhos para adaptar o gold ao resultado do modelo.
- A especificação estará implementada quando todos os braços previstos percorrerem texto bruto, entidades, candidatos, relações e grafo, produzindo contratos, métricas, análises e manifestos válidos, independentemente dos valores de F1.

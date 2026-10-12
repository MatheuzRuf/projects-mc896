# Infraestrutura segura para extração supervisionada de relações clínicas

Status: ready-for-agent

## Problem Statement

O projeto precisa evoluir a partir do comportamento existente no `project1` sem alterar esse projeto e sem depender acidentalmente de seus módulos internos. A nova linha de trabalho deve preparar uma base reproduzível para comparar, em etapas posteriores, extração por regras, kNN e uma abordagem híbrida na identificação de relações clínicas.

Antes de treinar ou comparar extratores, é necessário resolver dois riscos centrais. O primeiro é a compatibilidade entre os dados externos usados para supervisão e o esquema clínico da aplicação. O segundo é data leakage entre o corpus externo e os 50 artigos da amostra MultiCaRe já usada pelo projeto. Como ambos os conjuntos podem conter artigos do PubMed Central, uma divisão aleatória ingênua não é suficiente.

O primeiro corpus candidato é o Medical Case Report Corpus (MCRC). Ele precisa ser adquirido e convertido de forma controlada, pois sua distribuição oficial utiliza um artefato serializado inseguro para consumo automático. O corpus também possui relações majoritariamente intersentenciais, menções descontínuas, aninhadas e com múltiplos rótulos, exigindo contratos de dados que não descartem essa informação.

O marco inicial deve terminar antes do treinamento definitivo de kNN. Seu resultado será uma infraestrutura independente, validada e auditável que consiga decidir se o MCRC é adequado, gerar uma divisão congelada sem vazamento e avaliar futuras previsões por uma interface comum.

## Solution

Criar uma infraestrutura independente no novo pacote, preservando o `project1` como referência de regressão. A preparação de dados será executada como um fluxo reproduzível que recebe um corpus externo previamente inspecionado e a coleção MultiCaRe protegida, normaliza documentos, menções e relações, audita sobreposição, decide a adequação do corpus e, quando aprovado, produz uma divisão documental congelada.

O MCRC será convertido uma única vez de seu artefato original para JSONL seguro. A origem será fixada por versão e hash, e a conversão registrará um manifesto. O pipeline normal consumirá apenas o JSONL convertido, nunca desserializando silenciosamente o artefato bruto. Dados brutos, textos clínicos e artefatos grandes permanecerão fora do controle de versão; manifestos e relatórios pequenos poderão ser versionados.

Os contratos internos distinguirão documento, caso clínico, paciente, menção, relação anotada e previsão. Uma menção poderá conter múltiplos intervalos no padrão fechado-aberto, tipos nativos múltiplos e um tipo canônico separado. Relações conservarão o rótulo nativo e sua projeção canônica. A projeção inicial compreenderá `HAS_CONDITION`, `HAS_FINDING`, `HAS_FACTOR`, `NEGATES`, `CAUSES` e `NO_RELATION`, sem inventar subclasses supervisionadas não presentes no corpus.

O caso clínico descreverá o paciente, em vez de ambos serem tratados como a mesma entidade. A negação será preservada como evidência explícita entre um modificador e um achado; um achado negado não receberá simultaneamente uma relação clínica positiva incompatível.

A auditoria de leakage comparará identificadores bibliográficos, hashes de conteúdo, documentos completos e trechos. Coincidências exatas por identificador ou hash excluirão o artigo do corpus de desenvolvimento. Similaridade textual alta sem identificador decisivo colocará o documento em quarentena. Itens em quarentena não poderão entrar em treino enquanto não houver revisão explícita.

Se o MCRC passar pelos critérios documentados de cobertura, qualidade e ausência de contaminação impeditiva, o teste será congelado em aproximadamente 20% dos documentos. Os 80% restantes serão usados em validação cruzada agrupada por documento. Pares candidatos serão construídos sem cruzar partições. No treino, negativos serão apenas pares compatíveis por tipo e limitados a dois por relação positiva; validação e teste manterão todos os candidatos elegíveis.

Uma segunda interface comum receberá o ouro e previsões de qualquer extrator futuro. Cada previsão identificará documento e menções de origem e destino, relações nativa e canônica, método, evidências, sentenças, versão do modelo e, quando aplicável, `neighbor_vote_fraction`. O campo `confidence` permanecerá ausente ou nulo até existir calibração formal.

A avaliação priorizará macro-F1 nas classes positivas, excluindo `NO_RELATION`, e publicará métricas por classe, matriz de confusão, baseline majoritário, tempos e contagens. A acurácia poderá aparecer apenas como métrica auxiliar. Resultados sem suporte suficiente nas classes minoritárias serão declarados inconclusivos em vez de apresentados como evidência de bom desempenho.

## User Stories

1. Como mantenedor, quero executar os testes do novo pacote sem importar acidentalmente módulos do `project1`, para que a evolução possa ocorrer de forma isolada.

2. Como pesquisador, quero preservar o `project1` sem alterações funcionais, para que ele continue servindo como referência de regressão.

3. Como desenvolvedor, quero validar documentos, casos, pacientes, menções, relações e previsões por contratos explícitos, para detectar dados inválidos perto da origem.

4. Como pesquisador, quero representar menções contínuas, descontínuas, aninhadas e com múltiplos rótulos, para não perder fenômenos anotados pelo MCRC.

5. Como pesquisador, quero manter tipos nativos separados de tipos canônicos, para comparar resultados sem apagar a semântica do corpus de origem.

6. Como consumidor do grafo, quero que caso clínico e paciente sejam entidades distintas conectadas por `DESCRIBES`, para evitar uma identidade conceitual incorreta.

7. Como pesquisador, quero que modificadores negativos apontem explicitamente para os achados que negam, para preservar a evidência textual de negação.

8. Como operador do pipeline, quero verificar a versão e o hash do artefato bruto antes da conversão, para tornar a aquisição rastreável.

9. Como operador do pipeline, quero que a desserialização do artefato bruto exija uma ação explícita em ambiente controlado, para não executar conteúdo não confiável durante um processamento normal.

10. Como desenvolvedor, quero consumir o corpus normalizado em JSONL depois da conversão, para que execuções rotineiras usem um formato inspecionável e seguro.

11. Como revisor, quero um manifesto com origem, versão, hashes, contagens e resultado da conversão, para auditar exatamente quais dados foram preparados.

12. Como responsável pelos dados, quero que corpora, textos e artefatos grandes sejam ignorados pelo controle de versão, para evitar publicação acidental de material sensível ou volumoso.

13. Como pesquisador, quero comparar identificadores bibliográficos do corpus externo com os artigos MultiCaRe protegidos, para excluir sobreposições inequívocas.

14. Como pesquisador, quero comparar hashes de conteúdo, documentos completos e trechos, para detectar cópias ou variantes que não compartilhem o mesmo identificador.

15. Como revisor, quero que similaridades altas não resolvidas sejam colocadas em quarentena, para impedir que casos ambíguos contaminem o treinamento.

16. Como revisor, quero um relatório de leakage com evidências e decisões por documento, para conferir exclusões e quarentenas antes do treinamento.

17. Como pesquisador, quero um relatório de adequação do MCRC com cobertura das classes, suporte por relação e impacto da auditoria, para decidir objetivamente se o corpus pode ser usado.

18. Como responsável pelo projeto, quero que uma inadequação do MCRC seja apresentada como decisão explícita, para então providenciar acesso a i2b2/n2c2 sem que o pipeline prossiga silenciosamente.

19. Como pesquisador, quero congelar o teste por documento, para impedir que menções ou relações do mesmo artigo apareçam em partições diferentes.

20. Como pesquisador, quero realizar seleção nos demais documentos por validação cruzada agrupada, para estimar desempenho sem quebrar a independência entre artigos.

21. Como pesquisador, quero registrar os identificadores de cada partição, para reproduzir a mesma avaliação em todos os métodos futuros.

22. Como pesquisador, quero gerar candidatos compatíveis por tipo, para evitar negativos semanticamente impossíveis.

23. Como pesquisador, quero limitar negativos de treino a no máximo dois por positivo, para controlar o domínio de `NO_RELATION` sem adulterar validação e teste.

24. Como pesquisador, quero manter todos os candidatos elegíveis em validação e teste, para medir o problema real sem uma amostra artificialmente balanceada.

25. Como implementador de extratores, quero produzir previsões em um contrato comum, para que regras, kNN e o método híbrido usem o mesmo avaliador e a mesma projeção para o grafo.

26. Como pesquisador, quero avaliar previsões usando entidades ouro neste marco, para medir a classificação de relações separadamente dos erros de detecção de entidades.

27. Como pesquisador, quero macro-F1 das classes positivas e métricas por classe, para que `HAS` ou `NO_RELATION` não escondam falhas em relações minoritárias.

28. Como pesquisador, quero comparar cada execução com um baseline majoritário, para saber se o método aprendeu sinal útil além da distribuição dos rótulos.

29. Como revisor, quero que resultados com suporte minoritário insuficiente sejam marcados como inconclusivos, para evitar conclusões estatísticas frágeis.

30. Como pesquisador, quero registrar configuração, semente, versões, hashes dos dados, partições, política de negativos, métricas, matriz de confusão, tempos e auditoria, para reproduzir e comparar execuções.

31. Como desenvolvedor, quero testar o fluxo completo de preparação com um corpus seguro e uma coleção protegida de exemplo, para verificar a principal costura do sistema sem depender de dados privados.

32. Como desenvolvedor, quero testar o avaliador com ouro e previsões no contrato comum, para verificar a costura usada por todos os extratores futuros.

33. Como desenvolvedor, quero testes focados em intervalos, menções complexas, mapeamentos e invariantes de negação, para localizar violações dos contratos com mensagens claras.

34. Como usuário do projeto, quero executar o marco inicial por uma interface documentada e obter relatórios determinísticos, para repetir a preparação em outra máquina.

## Implementation Decisions

- O novo pacote será independente do `project1`; dependências acidentais entre os dois serão removidas antes das demais funcionalidades.
- O `project1` será preservado como implementação legada e referência de regressão.
- O MCRC será a primeira opção de corpus supervisionado externo. i2b2/n2c2 dependerá de acesso fornecido pelo usuário e será considerado somente se os critérios rejeitarem o MCRC.
- A preparação separará corpus supervisionado externo e coleção MultiCaRe de aplicação. Os 50 artigos MultiCaRe atuais serão protegidos contra uso em treino.
- O formato operacional do corpus será JSONL. O artefato serializado original existirá apenas na etapa controlada de conversão e será identificado por origem, versão e SHA-256.
- O pipeline não abrirá automaticamente artefatos serializados não confiáveis.
- Menções usarão um ou mais intervalos de caracteres no padrão `[start, end)`, com identificadores estáveis e tipos nativos separados do tipo canônico.
- Relações serão direcionais e ligarão identificadores de menções ou entidades explicitamente declaradas.
- O mapeamento inicial será: documento para `ClinicalCase`; entidade nativa de caso para `Patient`; caso-tem-condição para `HAS_CONDITION`; caso-tem-achado para `HAS_FINDING`; caso-tem-fator para `HAS_FACTOR`; modificador-modifica-achado para `NEGATES`; condição-causa-achado para `CAUSES`.
- `Finding` continuará amplo neste marco. Não serão inferidas subclasses supervisionadas como sintoma ou resultado de exame sem apoio explícito nos dados.
- Relações intersentenciais serão preservadas. Janelas de candidatos serão configuráveis e registrarão as sentenças envolvidas.
- A unidade de classificação futura será o par de menções. Normalização e consolidação em conceitos ocorrerão depois da predição.
- Coincidências exatas por identificador ou hash serão excluídas; similaridades textuais altas sem identificação decisiva serão colocadas em quarentena.
- Nenhum item em quarentena poderá participar do treino até revisão e decisão registradas.
- A divisão será feita por documento, com teste fixo de aproximadamente 20% e validação cruzada agrupada nos aproximadamente 80% restantes.
- Negativos de treino serão pares compatíveis por tipo e limitados a dois por positivo. Validação e teste preservarão todos os candidatos elegíveis.
- Previsões manterão rótulos nativo e canônico, método, evidências, sentenças e versão. `neighbor_vote_fraction` representará somente a fração de votos dos vizinhos, não uma confiança calibrada.
- POS tagging e dependency parsing usarão inicialmente um pipeline scispaCy pré-treinado, sem NER e sem ajuste fino. Sua integração funcional, porém, pertence a um marco posterior à infraestrutura descrita nesta especificação.
- Uma implementação própria de POS tagging ou dependency parsing será apenas uma extensão opcional ao fim do projeto.

## Testing Decisions

- Um bom teste verificará comportamento observável nas fronteiras do sistema — entradas aceitas, artefatos produzidos, decisões registradas e erros devolvidos — sem depender da organização interna das funções ou de chamadas privadas.
- A principal costura de teste será o fluxo completo de preparação: receber um corpus externo seguro e uma coleção MultiCaRe protegida, validar os contratos, executar a auditoria de leakage e produzir dados preparados, relatório e divisão documental congelada quando o corpus for aprovado.
- A costura secundária será o avaliador: receber ouro e previsões no contrato comum e produzir métricas por classe, macro-F1 positiva, baseline, matriz de confusão e metadados reproduzíveis.
- Serão exercitados como módulos observáveis: contratos e serialização, conversor seguro, auditor de leakage, decisão de adequação, particionador documental, gerador de candidatos e negativos, contrato de previsões e avaliador.
- Fixtures pequenas e sintéticas cobrirão os fluxos completos sem versionar textos clínicos reais.
- Testes focados cobrirão intervalos fechados-abertos, múltiplos intervalos, aninhamento, múltiplos rótulos, referências inválidas, direção das relações, mapeamento nativo-canônico e incompatibilidades de negação.
- A conversão será testada quanto a verificação de hash, recusa de origem não aprovada, determinismo do JSONL e geração do manifesto.
- A auditoria será testada com coincidência por identificador, coincidência por hash, similaridade por documento, similaridade por trecho, quarentena e ausência de sobreposição.
- A divisão será testada para garantir disjunção documental, estabilidade sob a mesma semente e ausência de itens excluídos ou em quarentena.
- A geração de negativos será testada para garantir compatibilidade de tipos, limite apenas no treino e retenção integral em validação e teste.
- As métricas serão testadas em cenários dominados por `NO_RELATION`, classes positivas ausentes e suporte minoritário insuficiente.
- Os testes do novo pacote deverão rodar de forma independente dos testes e imports do `project1`.
- Como prior art, serão mantidos os padrões já usados no projeto: testes pytest orientados à saída pública, fixtures temporárias para artefatos CSV, testes de pipeline que atravessam transformação e exportação, e asserções sobre contratos serializados e invariantes do grafo.

## Out of Scope

- Treinamento definitivo, busca de hiperparâmetros ou comparação final do kNN.
- Implementação final do extrator por regras experimental e da estratégia híbrida.
- Integração definitiva das previsões ao knowledge graph de produção.
- Avaliação ponta a ponta que inclua detecção automática de entidades.
- Normalização de menções em conceitos clínicos consolidados.
- Calibração probabilística de confiança.
- Criação do zero de um POS tagger ou dependency parser.
- Ajuste fino de modelos linguísticos ou uso de LLM na extração.
- Obtenção autônoma de datasets restritos i2b2/n2c2.
- Publicação de corpora, textos clínicos, modelos ou outros artefatos grandes no repositório.

## Further Notes

- O MCRC contém poucos exemplos de algumas relações, especialmente `CAUSES`. A decisão de adequação deve considerar suporte real depois das exclusões e quarentenas, sem rejeitar automaticamente o corpus apenas por desbalanceamento.
- A elevada proporção de relações intersentenciais no MCRC impede assumir que candidatos estarão sempre na mesma sentença.
- Caso o MCRC seja rejeitado, os contratos, o conversor seguro, a auditoria, o mecanismo de partição e o avaliador devem continuar reutilizáveis com outro corpus.
- Relatórios e manifestos pequenos podem ser versionados; dados brutos, textos, modelos e saídas volumosas devem permanecer ignorados.
- O marco estará concluído quando o pacote for independente, os contratos e validadores estiverem cobertos por testes, o MCRC tiver uma decisão de adequação auditável, uma divisão estiver congelada se aprovado, e o avaliador comum produzir artefatos reproduzíveis sem treinamento definitivo de kNN.

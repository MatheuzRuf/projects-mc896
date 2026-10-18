# Extração de Relações Clínicas

Este contexto define a linguagem usada para comparar métodos clássicos de extração de relações clínicas e aplicar os resultados à construção de um Knowledge Graph.

## Language

**Relação clínica explícita**:
Relação semântica entre entidades sustentada pelo texto de origem, sem inferência baseada apenas em conhecimento médico externo.
_Evitar_: relação presumida, associação clínica implícita

**Benchmark supervisionado**:
Comparação de regras, kNN e método híbrido sobre o mesmo corpus acadêmico anotado e a mesma divisão fixa de avaliação.
_Evitar_: avaliação MultiCaRe, comparação de grafos

**Aplicação MultiCaRe**:
Uso dos extratores sobre relatos do MultiCaRe para produzir o Knowledge Graph, sem atribuir métricas supervisionadas na ausência de anotações de referência.
_Evitar_: benchmark MultiCaRe

**ClinicalCase**:
Unidade documental identificada por `case_id` que preserva o artigo e a proveniência do relato e descreve um paciente.
_Evitar_: Patient, artigo

**Patient**:
Pessoa descrita por um ClinicalCase e usada como âncora clínica para condições, achados e fatores atribuídos pelo texto.
_Evitar_: ClinicalCase, documento

**Condition**:
Doença ou condição médica atribuída ao paciente no esquema canônico inicial.
_Evitar_: Finding, diagnóstico presumido

**Finding**:
Sinal, sintoma ou observação clínica mensurável atribuída ao paciente, sem exigir uma subdivisão que o corpus não anota.
_Evitar_: Condition, resultado inferido

**Factor**:
Característica, histórico ou comportamento do paciente que o corpus anota como potencialmente relevante para uma condição.
_Evitar_: causa confirmada, Finding

**Modifier**:
Expressão textual que nega um Finding e permanece registrada como a evidência da relação `NEGATES`.
_Evitar_: Finding ausente sem evidência, atributo descartável

**Anotação gold**:
Referência humana validada que identifica entidades e relações corretas em um texto e permite calcular métricas supervisionadas.
_Evitar_: predição, anotação automática

**Corpus acadêmico**:
Dataset externo com anotações de referência usado no benchmark supervisionado, desde que passe pela auditoria de leakage e seja compatível com o esquema de relações.
_Evitar_: corpus de aplicação, MultiCaRe

**Medical Case Report Corpus**:
Primeiro candidato a corpus acadêmico, composto por relatos clínicos do PubMed Central com entidades e relações anotadas.
_Evitar_: MultiCaRe

**Esquema nativo de relações**:
Vocabulário de relações preservado exatamente como definido pelo corpus acadêmico de origem, sem reinterpretá-lo para adequação ao grafo.
_Evitar_: esquema do Knowledge Graph, relação convertida

**Esquema canônico de relações**:
Vocabulário próprio do projeto para representar relações clínicas explícitas de forma uniforme entre extratores e no Knowledge Graph.
_Evitar_: rótulo nativo do corpus

**Esquema canônico inicial**:
Primeira versão do vocabulário supervisionado, limitada a `HAS_CONDITION`, `HAS_FINDING`, `HAS_FACTOR`, `NEGATES`, `CAUSES` e `NO_RELATION`.
_Evitar_: inventário completo do Knowledge Graph

**Esquema canônico inicial de entidades**:
Primeira versão do vocabulário supervisionado de menções, limitada a `Condition`, `Finding`, `Factor` e `Modifier`, exatamente como sustentado pelo MCRC.
_Evitar_: inventário completo do Knowledge Graph, tipos legados inferidos sem supervisão

**Expansão semântica posterior**:
Fase posterior ao fechamento do pipeline end-to-end em que entidades e relações poderão adotar distinções mais específicas do esquema legado, desde que cada distinção possua supervisão ou mapeamento semanticamente justificado e nova avaliação.
_Evitar_: renomeação por legibilidade, expansão automática de rótulos

**Mapeamento de relações**:
Correspondência explícita e auditável entre uma relação do esquema nativo e uma relação semanticamente compatível do esquema canônico.
_Evitar_: renomeação automática, equivalência presumida

**Relação intersentencial**:
Relação anotada cujas entidades participantes não estão contidas na mesma sentença do texto de origem.
_Evitar_: relação distante

**Enriquecimento sintático**:
Características de POS, lemas, núcleos e dependências fornecidas inicialmente por um pipeline linguístico pronto e usadas pelos experimentos que incluem informação sintática.
_Evitar_: parser próprio, relação semântica

**Parser próprio opcional**:
Possível extensão ao final do projeto que substitui o provedor pronto de POS e dependências por uma implementação desenvolvida pela equipe, sem alterar os contratos dos extratores.
_Evitar_: requisito do primeiro marco, parser inicial

**Menção clínica**:
Ocorrência específica de uma expressão clínica no texto, identificada por seus offsets e preservada como unidade da classificação de relações.
_Evitar_: conceito clínico, nó agregado

**Menção complexa**:
Menção clínica que pode ocupar vários intervalos descontínuos, sobrepor-se a outras menções ou possuir mais de um tipo nativo.
_Evitar_: intervalo envolvente único, tipo nativo descartado

**Segmentação canônica plana**:
Interpretação global do texto em menções previstas não sobrepostas, cada uma com exatamente um tipo canônico, destinada à extração end-to-end e à projeção no Knowledge Graph.
_Evitar_: anotação gold original, conjunto arbitrário de spans independentes

**Projeção gold plana**:
Alvo supervisionado derivado das anotações gold originais por uma política auditável de resolução de duplicidades, sobreposições e conflitos de tipo, sem modificar o corpus de origem.
_Evitar_: correção do corpus, descarte silencioso de menções

**Menção prevista**:
Segmento textual contíguo escolhido pelo reconhecedor de entidades, com offsets no texto original e exatamente um tipo canônico.
_Evitar_: conceito normalizado, token isolado, menção gold

**Conceito clínico**:
Representação normalizada que pode reunir várias menções equivalentes e originar um único nó no Knowledge Graph.
_Evitar_: menção textual, ocorrência

**Par candidato**:
Par ordenado de menções cujos tipos e contexto permitem alguma relação prevista pelo esquema em avaliação.
_Evitar_: qualquer combinação de entidades

**Exemplo negativo**:
Par candidato coberto pela política de anotação, mas sem relação gold, rotulado como `NO_RELATION` para o treinamento ou a avaliação.
_Evitar_: qualquer relação não anotada

**Baseline legado**:
Extrator de relações original do Project 1, preservado como referência de regressão e não como comparador experimental principal.
_Evitar_: baseline experimental

**Baseline experimental de regras**:
Extrator determinístico construído sobre os contratos compartilhados de menções, pares candidatos e relações e ajustado sem consultar o conjunto de teste.
_Evitar_: baseline legado

**Avaliação gold-entity**:
Avaliação da extração de relações usando exatamente as menções anotadas pelo corpus, isolando o classificador dos erros de reconhecimento de entidades.
_Evitar_: avaliação end-to-end

**Avaliação end-to-end**:
Avaliação conjunta em que as menções são produzidas pelo extrator real antes da classificação das relações.
_Evitar_: avaliação gold-entity

**Auditoria de leakage**:
Verificação reproduzível de sobreposição por artigo, identificador e similaridade textual entre um corpus acadêmico e os documentos protegidos de avaliação ou aplicação.
_Evitar_: divisão aleatória, deduplicação apenas por sentença

**Documento protegido**:
Artigo e todos os seus casos que não podem contribuir para treinamento, ajuste de características, regras ou hiperparâmetros por pertencerem ao universo reservado para avaliação.
_Evitar_: apenas sentença de teste, apenas caso de teste

**Documento em quarentena**:
Documento externo com indício não resolvido de sobreposição por similaridade textual, impedido de entrar no treinamento até revisão.
_Evitar_: duplicata confirmada, documento limpo

**Relatório de leakage**:
Artefato reproduzível que registra identificadores coincidentes, duplicatas exatas, candidatos quase duplicados, exclusões, quarentenas e o corpus limpo resultante.
_Evitar_: log informal, inspeção não registrada

**Predição de relação**:
Registro intermediário comum produzido por regras, kNN ou método híbrido, ligado às menções e evidências originais antes da projeção no Knowledge Graph.
_Evitar_: aresta do grafo, relação gold

**Fração de votos dos vizinhos**:
Proporção dos vizinhos do kNN que votaram na classe predita, registrada como score descritivo e não como probabilidade calibrada.
_Evitar_: confiança, probabilidade

**Execução experimental**:
Rodada reproduzível identificada por sua configuração, seed, versões, hashes dos dados, divisão documental e artefatos de avaliação.
_Evitar_: execução ad hoc, resultado sem manifest

**Braço experimental end-to-end**:
Combinação fixa de um reconhecedor de entidades e um classificador de relações executada do texto bruto até as predições estruturadas, mantida na comparação independentemente de superar ou não outro método.
_Evitar_: modelo promovido, fallback automático, vencedor presumido

**Dicionário canônico**:
Baseline lexical que reconhece menções usando termos associados aos quatro tipos de entidades do esquema MCRC e que é construído ou adaptado sem consultar a partição avaliada.
_Evitar_: dicionário legado, vocabulário obtido do teste

**Teto experimental gold-entity**:
Resultado da classificação de relações quando o método recebe as menções gold, usado para separar erros de relações dos erros de reconhecimento de entidades e não apresentado como execução end-to-end.
_Evitar_: pipeline completo, resultado de produção

**Adjudicação da projeção gold**:
Revisão humana independente dos conflitos de sobreposição ou tipo para produzir uma segmentação canônica plana, preservando decisão, justificativa e proveniência sem alterar a anotação original.
_Evitar_: correção silenciosa, prioridade automática

**Validação cruzada aninhada**:
Protocolo por documento em que folds externos estimam desempenho e subdivisões internas do treino de cada fold externo selecionam hiperparâmetros, sem consultar a validação externa durante a seleção.
_Evitar_: reutilizar o mesmo fold para escolher e estimar, validação cruzada simples de seleção

**Correspondência estrita de entidade**:
Critério em que uma menção prevista só é considerada correta quando seus offsets e seu tipo canônico coincidem exatamente com a projeção gold plana.
_Evitar_: sobreposição parcial, acerto apenas por token

**Relação end-to-end estrita**:
Relação prevista cujos dois endpoints, tipos canônicos e rótulo coincidem exatamente com o gold, após reconhecimento real das entidades e sem fornecer candidatos gold ao classificador.
_Evitar_: avaliação gold-entity, relação com endpoint aproximado

**Tokenização com offsets**:
Divisão determinística do texto em palavras, números e pontuação na qual cada token conserva seus índices inicial e final no texto original.
_Evitar_: tokenização destrutiva, token sem posição original

**Segmento externo unitário**:
Representação canônica de um único token fora de qualquer entidade no semi-Markov CRF, usada para impedir várias segmentações latentes equivalentes da mesma saída.
_Evitar_: entidade `O`, segmento externo arbitrário

**Predição de entidade**:
Registro intermediário separado do gold que identifica uma ocorrência prevista por documento, offsets, tipo canônico, sentença, método e versão do modelo.
_Evitar_: anotação nativa, conceito agregado, menção gold

**Teste externo congelado**:
Conjunto de documentos ausente das fontes de desenvolvimento, auditado contra duplicidades, anotado sob o mesmo esquema e imutável antes da primeira execução dos modelos comparados.
_Evitar_: teste MCRC consumido, conjunto ajustado após resultados

**Manifesto experimental**:
Registro estruturado que liga uma execução ao commit, hashes dos dados e saídas, partições documentais, configuração, dependências, determinismo, convergência e duração.
_Evitar_: log informal, métricas sem proveniência

**Camada de menção do grafo**:
Representação em que cada ocorrência textual prevista permanece um nó identificável por documento, offsets e tipo, servindo de base para relações e avaliação estrita.
_Evitar_: conceito agregado, nó sem proveniência textual

**Camada conceitual do grafo**:
Projeção posterior que reúne menções semanticamente compatíveis sem remover os IDs, offsets e relações que sustentam cada conceito.
_Evitar_: agregação precoce, perda da camada de menção

**Conclusão operacional**:
Estado em que um braço experimental executa integralmente, valida seus contratos e produz todos os artefatos exigidos, independentemente do valor de suas métricas.
_Evitar_: promoção por F1, sucesso presumido pelo treinamento isolado

**Identidade conceitual tipada**:
Chave documental formada pelo tipo canônico e pela superfície normalizada com Unicode, `casefold` e espaços consolidados, mantendo tipos distintos mesmo quando o texto é igual.
_Evitar_: identidade apenas pelo rótulo, fusão entre tipos

**Candidato de relação não rotulado**:
Par de endpoints previstos compatível com uma relação hipotética e serializado antes da classificação, sem usar `NO_RELATION` como substituto para um rótulo ainda desconhecido.
_Evitar_: candidato gold, negativo presumido

**Marginal segmental**:
Posterior atribuída pelo semi-Markov CRF a um segmento e tipo, calculada pelo forward-backward e registrada sem alegação de calibração empírica.
_Evitar_: confiança calibrada, score bruto de sequência


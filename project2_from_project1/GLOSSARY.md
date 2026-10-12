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


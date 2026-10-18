# Avaliação de i2b2/n2c2 para expansão do treinamento

## Contexto

O resultado final do pipeline n-gramas + k-NN mostrou macro-F1 positiva de
0,3093 no teste fixo. As classes mais fracas foram `CAUSES` e
`HAS_CONDITION`, enquanto o conjunto de treino do MCRC permaneceu pequeno e
desbalanceado. Por isso, i2b2/n2c2 foram avaliados como possíveis fontes de
dados adicionais.

## Compatibilidade semântica

### i2b2 2010 Relations

O corpus anota conceitos do tipo problema, tratamento e exame, além de oito
relações. Entre elas estão tratamento causando problema (`TrCP`), tratamento
administrado para problema (`TrAP`), exame revelando problema (`TeRP`) e
problema indicando outro problema (`PIP`). A tarefa de assertions também
classifica problemas como presentes, ausentes, possíveis, condicionais,
hipotéticos ou associados a outra pessoa. A descrição completa está no
[artigo oficial do desafio i2b2 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC3168320/).

O corpus é relevante, mas não possui equivalência direta com o esquema atual:

- `PIP` reúne indicação, descrição e causalidade; mapear todo `PIP` para
  `CAUSES` introduziria falsos rótulos;
- `TrCP` representa `Treatment → Problem`, enquanto o `CAUSES` atual representa
  `Condition → Finding`;
- a assertion `absent` pode ensinar negação, mas não fornece necessariamente
  o span do modificador requerido por `Modifier → Finding` em `NEGATES`;
- a classe ampla `Problem` não separa diretamente `Condition`, `Finding` e
  `Factor`.

Assim, i2b2 2010 pode servir como tarefa auxiliar, pré-treinamento ou fonte de
subconjuntos curados, mas não deve ser concatenado diretamente ao dataset
canônico.

### n2c2 2018 Adverse Drug Events

O conjunto possui 505 notas clínicas e anota medicamentos, eventos adversos e
atributos como dose, frequência, rota, duração e motivo. Suas relações ligam
esses atributos ao medicamento, conforme a
[descrição do desafio n2c2 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC7489085/).

Ele seria valioso para uma futura extensão com `Drug`/`Treatment` e
`AdverseDrugEvent`, mas possui baixa compatibilidade com as cinco relações
atuais. Em particular, uma relação medicamento–evento adverso não deve ser
renomeada como o atual `Condition → Finding` sem mudar e versionar o esquema.

### n2c2 2022 SDOH e i2b2 2014

O n2c2 2022 usa anotações orientadas a eventos para uso de substâncias,
emprego e situação de moradia, com argumentos de estado, tipo, extensão e
tempo. Ele pode apoiar uma futura extração de fatores sociais, mas não fornece
diretamente o atual `Patient → Factor`. Consulte a
[descrição oficial do n2c2 2022](https://n2c2.dbmi.hms.harvard.edu/2022-track-2).

O i2b2 2014 contém fatores de risco cardiovascular anotados e pode ser
investigado para `HAS_FACTOR`. Entretanto, também exige um adaptador e uma
avaliação semântica antes de qualquer mapeamento.

## Restrições de acesso

Os datasets i2b2/n2c2 são disponibilizados para pesquisa sob Data Use
Agreement. Cada integrante precisa obter acesso individualmente e os arquivos
não podem ser enviados ao GitHub nem repassados a outras pessoas. O
[catálogo oficial](https://n2c2.dbmi.hms.harvard.edu/data-sets) descreve essas
condições; o [portal DBMI](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/)
também deve ser consultado para confirmar a disponibilidade no momento da
solicitação.

## Decisão

Não adicionar i2b2/n2c2 diretamente ao treinamento atual. O ganho potencial
não justifica, neste momento, o risco de incompatibilidade semântica, mudança
de domínio entre relatos de caso e prontuários e ruído adicional no k-NN por
contagem de n-gramas.

Se a expansão for retomada, a ordem recomendada é:

1. i2b2 2010 como experimento auxiliar para assertions/negação e relações
   clínicas;
2. n2c2 2022 ou i2b2 2014 para fatores sociais e fatores de risco;
3. n2c2 2018 somente junto de uma extensão versionada do esquema para
   medicamentos e eventos adversos.

Cada corpus deverá ter conversor e adaptador próprios, preservar seus rótulos
nativos e usar somente mapeamentos canônicos explicitamente justificados. A
comparação entre MCRC isolado e MCRC com dados externos deve ocorrer apenas
nos folds de desenvolvimento. O teste MCRC já consumido não poderá ser usado
para escolher essa integração; uma avaliação futura exigirá teste externo novo.

Por enquanto, a prioridade do projeto deve ser outra melhoria no pipeline. Uma
base adicional de relatos de caso com anotações semanticamente equivalentes ao
MCRC continuaria sendo preferível a uma concatenação direta de i2b2/n2c2.

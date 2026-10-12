# Preservar menções complexas sem perda

O formato interno aceitará vários spans `[start, end)` e vários tipos nativos por menção, além de aninhamento entre menções. O tipo canônico será armazenado separadamente dos tipos de origem; não substituiremos spans descontínuos por um intervalo envolvente nem descartaremos rótulos para adaptar o corpus ao modelo, pois isso alteraria as anotações gold usadas na avaliação.

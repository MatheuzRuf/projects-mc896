# Versionar o grafo para o esquema supervisionado antes de expandi-lo

A primeira integração end-to-end representará diretamente os tipos de entidade
`Condition`, `Finding`, `Factor` e `Modifier` e as relações canônicas
sustentadas pelo MCRC. O esquema do Knowledge Graph será versionado em vez de
forçar equivalências com categorias legadas como `Diagnosis`, `Symptom` ou
`History`. O dicionário legado permanecerá como baseline independente; o braço
semi-Markov CRF será executado end-to-end mesmo quando tiver resultado inferior,
sem fallback que esconda suas previsões.

O contrato canônico do novo grafo será JSONL versionado e manterá duas camadas:
menções textuais individuais, usadas pela classificação e avaliação estrita, e
conceitos derivados depois das relações. A identidade conceitual incluirá
documento, tipo e superfície normalizada; os IDs e offsets das menções de origem
serão preservados. Os CSVs legados serão somente uma exportação adaptada.

Depois de o pipeline completo estar funcional e avaliado, entidades e relações
poderão ser expandidas para classificações legadas mais específicas quando
houver supervisão ou mapeamento semanticamente justificável, acompanhado de
nova versão do esquema e nova avaliação.

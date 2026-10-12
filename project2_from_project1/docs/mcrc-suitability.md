# Decisão de adequação do MCRC

Depois de converter o artefato autorizado para JSONL seguro e executar a
auditoria contra a coleção MultiCaRe protegida, produza a decisão com:

```powershell
python project2_from_project1/scripts/decide_mcrc_suitability.py `
  --converted-jsonl data/secure/mcrc.jsonl `
  --leakage-report data/secure/mcrc.leakage.json `
  --output-report data/secure/mcrc.suitability.json
```

O comando nunca abre o pickle original. Ele relê e valida o contrato de cada
documento do JSONL, exige que o relatório de leakage cubra exatamente esses
documentos e registra, sem texto clínico:

- contagens de documentos, menções e relações antes da auditoria e nos grupos
  liberado, excluído e em quarentena;
- suporte por rótulo nativo e por classe canônica, inclusive suporte zero;
- critérios, impacto da auditoria, decisão e razões;
- somente os IDs liberados, quando a decisão for `approved`.

Por padrão, há ao menos uma relação liberada para cada classe supervisionada
canônica (`HAS_CONDITION`, `HAS_FINDING`, `HAS_FACTOR`, `NEGATES`, `CAUSES`).
Isso verifica cobertura sem exigir distribuição balanceada. Os limiares podem
ser registrados explicitamente pelas opções `--minimum-*`,
`--required-canonical-relation` e `--maximum-blocked-document-fraction`.

Uma decisão `rejected` deixa `eligible_document_ids` vazio: a etapa de
particionamento não deve criar split de treino e o relatório aponta a obtenção
de acesso autorizado a i2b2/n2c2 ou outro corpus como próximo passo.

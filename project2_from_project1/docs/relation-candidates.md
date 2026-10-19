# Candidatos de relações a partir de entidades previstas

O gerador recebe JSONL de documentos com `document_id` e `text` e previsões
`EntityPrediction` de qualquer reconhecedor. Campos de anotação presentes no
corpus são ignorados. Nenhum endpoint ou relação gold participa da geração.

```powershell
python -m project2_from_project1.scripts.generate_relation_candidates `
  --corpus project2_from_project1/data/secure/mcrc.jsonl `
  --predictions project2_from_project1/artifacts/entity-dictionary/fold-01/predictions.jsonl `
  --prediction-manifest project2_from_project1/artifacts/entity-dictionary/fold-01/manifest.json `
  --output-dir project2_from_project1/artifacts/relation-candidates/fold-01
```

O manifesto do reconhecedor é opcional; quando fornecido, seu hash é registrado,
e os hashes do corpus e das previsões e as versões de segmentação são verificados. Também é
possível usar o arquivo de previsões do CRF no mesmo comando.

`Patient` é um nó lógico por documento, com ID `patient:` seguido do SHA-256 do
array JSON compacto UTF-8 contendo o ID do documento. `Patient` aponta para
cada `Condition`, `Finding` ou `Factor` previsto, com distância nula. `Modifier`
aponta para `Finding` em `NEGATES`; `Condition` aponta para `Finding` em
`CAUSES`. Esses pares textuais são mantidos quando a distância entre sentenças
é zero ou um. Os spans e índices são verificados contra as sentenças produzidas
pelo segmentador e tokenizador versionados, incluindo regras de abreviação. As relações são
hipóteses, não decisões: `candidates.jsonl` usa `RelationCandidate` v1 sem
`label` ou IDs gold.

`manifest.json` registra hashes de entrada e saída, versões, configuração,
contagens, IDs do corpus, partições disponíveis, métodos e informações de ambiente. As
informações Git indicam alterações em arquivos rastreados; o hash do código do
gerador também é registrado, inclusive quando ele ainda não foi commitado. As
previsões precisam referir documentos presentes no corpus e, quando informada,
na partição de validação do manifesto do reconhecedor. A lista dessa partição
inclui documentos sem previsões; sem manifesto, ela é registrada como nula. IDs duplicados,
spans fora da sentença e índices incompatíveis causam erro. A entrada completa
pode conter documentos sem previsões, que não produzem candidatos.

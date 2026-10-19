# Baseline canônico de relações por regras

O baseline ajusta uma regra por hipótese e distância entre sentenças em candidatos
gold **de treino** do dataset supervisionado congelado. Para cada grupo, uma
maioria estrita positiva prevê a relação hipotética; empate, ausência de exemplos
ou maioria negativa prevê `NO_RELATION`. Para `Patient`, a distância é nula;
para os demais pares, é zero ou um. As regras são deliberadamente simples: não
consultam texto, extratores legados ou respostas gold na inferência. O dataset
congelado amostra negativos no treino e usa seu próprio segmentador; portanto
as contagens e as decisões podem refletir essa amostragem e diferenças de
segmentação em relação aos candidatos de entidades previstas.

```powershell
python -m project2_from_project1.scripts.run_canonical_relation_rules `
  --training-candidates project2_from_project1/data/secure/mcrc-supervised/fold-01-train.jsonl `
  --partition-manifest project2_from_project1/data/secure/mcrc-supervised/manifest.json `
  --fold 1 `
  --candidates project2_from_project1/artifacts/relation-candidates/fold-01/candidates.jsonl `
  --candidate-manifest project2_from_project1/artifacts/relation-candidates/fold-01/manifest.json `
  --output-dir project2_from_project1/artifacts/relation-rules/fold-01
```

O manifesto dos candidatos deve provir do gerador com manifesto do
reconhecedor, para comprovar o corpus e as partições de treino e validação.
Ambos os reconhecedores, dicionário e CRF, usam a mesma entrada e as mesmas
regras ajustadas para o fold. A saída `rules.json` registra contagens e decisões;
`predictions.jsonl` usa `RelationPrediction`, com o ID da regra e do candidato
na evidência. O manifesto registra hashes, versão, tempos e partições sem
reproduzir texto clínico. Nenhuma probabilidade ou confiança é atribuída.

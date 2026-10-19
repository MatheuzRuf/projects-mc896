# Reconciliação da fonte de desenvolvimento

O corpus MCRC local e as partições originais usam SHA-256 `5104d36f…`,
enquanto a adjudicação da projeção gold exige `28844ee5…`. A diferença
comprovada é a terminação das linhas. O comando abaixo preserva o original,
aceita apenas uma transformação CRLF → LF cujo hash corresponda ao vínculo de
todas as decisões de adjudicação e registra os hashes de entrada e saída.

Na raiz do repositório, com o ambiente instalado:

```powershell
python -m project2_from_project1.scripts.reconcile_development_source `
  --corpus project2_from_project1/data/secure/mcrc.jsonl `
  --adjudication project2_from_project1/annotations/entity-gold-v1/adjudication.jsonl `
  --partitions project2_from_project1/artifacts/first-milestone/frozen-dataset/manifest.json `
  --output-dir project2_from_project1/artifacts/development-source-reconciled-v1
```

O destino deve ser novo. A cópia não altera o manifesto de partições: elas
precisam ser geradas novamente com a fonte reconciliada. Nesta máquina, a
preparação documentada no README foi repetida com a mesma configuração em
`artifacts/first-milestone-reconciled-v1/`. O novo manifesto vincula o hash
`28844ee5…`; os IDs das partições e os hashes de todos os arquivos de candidatos
coincidem com a preparação anterior. A projeção de desenvolvimento foi gerada
em `artifacts/development-source-reconciled-v1/gold/` e passou a validação das
decisões de adjudicação.

Esses artefatos não são modelos treinados nem uma avaliação externa. Os
diretórios `data/` e `artifacts/` permanecem locais e não versionados.

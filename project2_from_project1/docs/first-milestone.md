# Primeiro marco reproduzível

Esta interface une as etapas já validadas de auditoria de leakage, decisão de
adequação, congelamento do dataset e avaliação. Ela aceita somente o JSONL
seguro convertido: não abre pickle, não recebe distribuição bruta e não grava
texto clínico nos relatórios produzidos.

Execute a preparação em um diretório limpo e ignorado pelo Git:

```powershell
python project2_from_project1/scripts/reproduce_first_milestone.py prepare `
  --safe-jsonl project2_from_project1/data/secure/mcrc.jsonl `
  --protected-collection project2_from_project1/data/secure/multicare.fingerprints.json `
  --output-directory project2_from_project1/artifacts/first-milestone `
  --seed 42 --test-fraction 0.2 --folds 5 `
  --maximum-sentence-distance 1 --maximum-train-negatives-per-positive 2
```

`prepare` cria `leakage-report.json`, `suitability-report.json`,
`preparation-manifest.json`, `preparation-run-report.json` e, apenas quando a
decisão for `approved`, o diretório `frozen-dataset/`. Este último contém o
teste protegido, os pares treino/validação por fold e seu `manifest.json`. Uma
decisão `rejected` preserva a evidência, mas não cria partições.

O manifesto de preparação registra hashes das entradas e artefatos, versões,
semente, políticas de candidatos e negativos, partições, contagens e o
relatório de leakage. Ele é determinístico. O relatório de execução registra
os tempos observados; tempos não são comparáveis byte a byte entre máquinas ou
execuções.

Para avaliar um extrator, ele deve escrever uma previsão de contrato comum para
cada candidato ouro da partição, inclusive `NO_RELATION`:

```powershell
python project2_from_project1/scripts/reproduce_first_milestone.py evaluate `
  --prepared-directory project2_from_project1/artifacts/first-milestone `
  --predictions project2_from_project1/artifacts/rules-test.predictions.jsonl `
  --output project2_from_project1/artifacts/rules-test.evaluation.json `
  --partition test --seed 42 --minimum-positive-support 5
```

Além do relatório de avaliação, a interface grava
`rules-test.evaluation.manifest.json`. O relatório conserva métricas, matriz de
confusão, baseline majoritário, métodos, versões e tempos; o manifesto conserva
as partes determinísticas para comparação entre execuções. A avaliação usa
entidades ouro e mede somente classificação de relações.

Para reproduzir, execute `prepare` duas vezes com os mesmos dois arquivos
seguros, configuração e semente, em diretórios limpos. Compare os JSONL de
`frozen-dataset/` e os manifestos determinísticos (não os campos de tempo).
Depois repita `evaluate` com o mesmo arquivo de previsões e compare o manifesto
de avaliação.

Os caminhos `data/secure/`, `artifacts/`, dados brutos, modelos e outputs
volumosos são ignorados pelo Git. Este marco deixa o contrato, o ouro e a
avaliação prontos para extratores experimentais posteriores; não afirma ter
treinado ou validado definitivamente kNN, regras experimentais, híbridos, POS
tagging ou dependency parsing.

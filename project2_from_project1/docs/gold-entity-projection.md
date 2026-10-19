# Projeção gold plana de entidades

A projeção `flat-gold-v1` deriva uma segmentação plana do JSONL lossless do
MCRC sem alterar ou duplicar seu texto. O artefato guarda somente IDs, offsets,
tipos canônicos, limites de tokens e sentenças e proveniência das menções de
origem. O texto continua sendo lido exclusivamente do `ClinicalDocument`
lossless.

## Reprodução

Na raiz do monorepo:

```powershell
python -m project2_from_project1.scripts.project_gold_entities `
  --source-jsonl project2_from_project1/data/secure/mcrc.jsonl `
  --adjudication-jsonl project2_from_project1/annotations/entity-gold-v1/adjudication.jsonl `
  --output-dir project2_from_project1/artifacts/entity-gold-v1
```

Depois de concluída a adjudicação humana, o comando produz
`gold-projection.jsonl`, `adjudication-audit.jsonl` e `manifest.json`. O
manifesto contém hashes de todas as entradas e saídas, contagens, configuração,
transformações, revisões e rejeições, sem texto clínico. A execução calcula os
hashes da fonte e das decisões antes e depois da projeção e falha se qualquer
entrada mudar.

## Tokenização e sentenças

O tokenizador separa palavras, números e pontuação em intervalos semiabertos no
texto original. Assim, `5.0`, `mg/day` e `post-op` produzem pontuação separada,
sem perder offsets. O separador de sentenças usa os mesmos offsets e protege
abreviações, cadeias de iniciais e unidades observadas, incluindo `H.`, `J. R.`,
`p.o.`, `i.v.` e `mg.`. Uma menção só entra na projeção quando ocupa tokens
completos de uma única sentença e possui no máximo 40 tokens.

Espaços periféricos são removidos na camada derivada. Fragmentos de palavra
podem ser expandidos até o token completo. Ambas as mudanças criam caso de
adjudicação, preservam o span original no audit log e registram a transformação
aplicada. Não há truncamento automático.

## Adjudicação versionada

Duplicatas exatas por documento, spans e tipo são colapsadas antes da revisão;
todos os `mention_id` de origem continuam em `source_mention_ids`. Os demais
casos são componentes de sobreposição ou ajustes de limites.

O pacote local para os revisores pode ser exportado com:

```powershell
python -m project2_from_project1.scripts.export_gold_adjudication_packet `
  --source-jsonl project2_from_project1/data/secure/mcrc.jsonl `
  --output-dir project2_from_project1/artifacts/entity-gold-review-v1
```

`review-cases.jsonl` inclui contexto e superfícies clínicas e, por isso, deve
permanecer no diretório ignorado e com acesso controlado. Seu manifesto não
contém texto clínico e fixa o mesmo vínculo que deve ser copiado para cada
decisão final.

O arquivo versionado de decisões registra duas revisões feitas
separadamente, com identificadores distintos e justificativas específicas para
o caso. O segundo revisor não consulta o voto do primeiro. Quando os votos
iniciais divergem, um terceiro voto independente decide. O terceiro voto pode
escolher qualquer conjunto plano válido; o resultado final deve reproduzi-lo
exatamente.

Na versão `flat-gold-v1`, a revisão humana foi considerada inviável e os três
papéis de revisor foram executados por modelos de linguagem
(`reviewer_kind = model`), conforme
[`annotations/entity-gold-v1/README.md`](../annotations/entity-gold-v1/README.md)
e [`GUIDELINES.md`](../annotations/entity-gold-v1/GUIDELINES.md). A versão de
leitura dos casos e a coleta das revisões são feitas com:

```powershell
python -m project2_from_project1.scripts.model_adjudication prepare `
  --source-jsonl project2_from_project1/data/secure/mcrc.jsonl `
  --output-json project2_from_project1/artifacts/entity-gold-review-v1/model-review-cases.json
python -m project2_from_project1.scripts.model_adjudication collect `
  --packet project2_from_project1/artifacts/entity-gold-review-v1/model-review-cases.json `
  --review revisao-a.alias.jsonl --reviewer-id model-review-a `
  --output-jsonl revisao-a.jsonl
python -m project2_from_project1.scripts.assemble_gold_adjudication `
  --source-jsonl project2_from_project1/data/secure/mcrc.jsonl `
  --review-a-jsonl revisao-a.jsonl --review-b-jsonl revisao-b.jsonl `
  --third-review-jsonl revisao-c.jsonl `
  --output-jsonl project2_from_project1/annotations/entity-gold-v1/adjudication.jsonl
```

Cada decisão também carrega um vínculo imutável com o SHA-256 da fonte, o hash
do plano de candidatos, o limite de tokens e as versões do tokenizador e do
segmentador. Alterar texto, IDs de origem ou configuração invalida as decisões
anteriores. Passagens heurísticas não são aceitas como votos.

O carregador rejeita campos ausentes ou extras, casos ausentes, votos sem
proveniência distinta, candidatos de outro caso, resultados não alinhados,
sobreposições residuais, votos excedentes e vínculos com outra fonte ou outro
plano. O audit JSONL reúne o caso, os candidatos, os votos e a decisão. Como as
justificativas são livres, esse arquivo deve ser tratado com o mesmo controle de
acesso da fonte clínica. O manifesto continua sem texto clínico.

## Contrato da projeção

Cada linha de `gold-projection.jsonl` representa um documento e contém:

- `document_id` e `schema_version`;
- sentenças com `sentence_index`, span e spans dos tokens;
- menções com ID determinístico, span, tipo canônico, sentença, intervalo de
  tokens, IDs de origem, transformações e caso de adjudicação quando aplicável.

Os tipos aceitos são `Condition`, `Finding`, `Factor` e `Modifier`. Os registros
são ordenados deterministicamente e toda menção resultante é contígua,
alinhada, não sobreposta e limitada a 40 tokens.

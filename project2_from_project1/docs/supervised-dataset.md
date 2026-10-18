# Dataset supervisionado congelado

O congelamento parte exclusivamente do JSONL seguro e de uma decisão de
adequação `approved`. Somente `eligible_document_ids` do relatório podem entrar
nas partições; documentos excluídos ou em quarentena nunca são selecionados.

```powershell
python project2_from_project1/scripts/freeze_supervised_dataset.py `
  --converted-jsonl data/secure/mcrc.jsonl `
  --suitability-report data/secure/mcrc.suitability.json `
  --output-directory data/secure/mcrc-supervised `
  --seed 42 --test-fraction 0.2 --folds 5 `
  --maximum-sentence-distance 1
```

O diretório contém `test.jsonl`, um par `fold-XX-train.jsonl` e
`fold-XX-validation.jsonl` por fold, e `manifest.json`. As partições são por
`document_id`: nenhum documento, menção ou relação atravessa treino, validação
e teste. O teste é um holdout fixo de aproximadamente 20%; os demais
documentos aparecem uma única vez em validação e nos treinos dos outros folds.

Cada candidato armazena endpoints, tipos, relação compatível, rótulo ouro e
distância entre sentenças. A janela é inclusiva: `1` permite pares na mesma ou
em sentenças adjacentes; `-1` remove o limite. Relações ouro compatíveis em tipo, mas
fora da janela, não viram candidatos; são contadas por rótulo em
`gold_outside_window` no manifesto, em vez de serem silenciosamente descartadas.
Relações incompatíveis em tipo ou duplicadas continuam causando erro.

Todos os candidatos compatíveis são mantidos em validação e teste. Em treino,
no máximo dois `NO_RELATION` são amostrados por relação positiva, de modo
determinístico a partir da semente e do fold. O manifesto registra IDs das
partições, configuração, semente, políticas, contagens e hashes, sem texto
clínico.
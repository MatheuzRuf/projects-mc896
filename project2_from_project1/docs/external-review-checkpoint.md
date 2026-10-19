# Ponto de pausa da revisão externa MultiCaRe

Em 2026-10-19, por decisão do usuário, as revisões foram pausadas após a
última rodada em andamento. Este é um checkpoint auditável, **não** um teste
externo congelado e não autoriza a execução da avaliação da issue 13.

Os artefatos clínicos permanecem em diretórios locais ignorados pelo Git. As
revisões estão em `artifacts/external-multicare-continuation-pool140-v1/`. A
preparação foi reproduzida com a versão final do script em
`artifacts/external-multicare-continuation-pool140-v3/`; seus três outputs são
idênticos aos do snapshot anterior e o manifesto, que agora vincula o código e
todas as fontes, tem SHA-256
`4d2e4433f026536546b5373ed4e53b679bd249d9b8c337b7f13c4181d5cf3d93`.
O pacote tem SHA-256
`f8a485f661471a651fc4de166425941f58924fcbd78af9ff6fc4194f0703de85` e a
seleção, `07d5cc9b85b6274f58555cf86321006d4daef40a8d1d8157898273491d2c794d`.

As trilhas cegas cumulativas validadas são:

- A, 110 candidatos: revisão
  `63b6185f396f6676916a99235bc5ed9ac8045d4c07f69cbea5b69c769549b0d7` e
  proveniência
  `db69f042b4a8fc6e607a7756a3a8929a9ddc6d42a48f4f3c6bc9bd5372a3b7f3`;
- B, 70 candidatos: revisão
  `b2aea894177a0e874a602257a553c73b98ab833cf3ce5b947cd574a5ae0f1949` e
  proveniência
  `1bd111cef2f4986f5e3f99b9682e88f62cb8956f08f88ebf68132899864c7466`;
- C, 75 candidatos: revisão
  `5c610ffd99eb1a46dc42bf528280fce9e500bfc720946f4498f14538014ab912` e
  proveniência
  `9f4ba9a19d7aa5581150a30bc0eea40d80dac17ed357b07f3e233746ee070de5`.

O maior prefixo com os três votos completos contém 70 candidatos. A montagem
provisória com suporte calculado somente sobre documentos elegíveis está em
`artifacts/external-multicare-fresh-gold-through70-eligible-v3/`:
`provisional-gold.jsonl` tem SHA-256
`6a0b76871fc75a028d6f2e9d1fe6e7b847b2eba9af5f244aa1ff211276846fc9`,
`decisions.json` tem
`d01825bdeba654c3d9ebf0e48cacc3a75112d91677d935c0cdcd635e0718969c` e o
manifesto tem
`6c637109914610bf13d607f3b5618010849cf6ac260e2637edc694e187e94071`.

Sete desses 70 candidatos são inelegíveis segundo a triagem de fonte. Nos 63
documentos elegíveis há 357 `Condition`, 340 `Factor`, 1.879 `Finding`, 264
`Modifier`, 341 `HAS_CONDITION`, 337 `HAS_FACTOR`, 1.489 `HAS_FINDING`, 341
`NEGATES` e 9 `CAUSES`. Os mínimos de documentos e entidades foram atingidos,
mas `CAUSES` permanece abaixo de 20 e o teto de 100 documentos elegíveis não
foi alcançado. Portanto, a regra de parada ainda não permite congelamento.

A triagem da fonte cobre 111 candidatos e registra 11 exclusões. O 100º
elegível é o candidato 110. Se o trabalho for retomado, B deve continuar no
candidato 71 e C no 76, sempre a partir dos hashes acima; A já cobre o corte
necessário. Depois de completar B e C até 110, é preciso montar novamente,
aplicar a regra de parada, congelar e verificar os hashes antes de qualquer
execução de modelo no conjunto externo.

O último parágrafo de `external-continuation-pool-amendment.md` registra o
estado anterior de 10 exclusões e corte 109 e permanece byte a byte preservado
por já estar vinculado às revisões. Ele foi substituído pela correção
`external-source-eligibility-completion.md`; para qualquer retomada, valem 11
exclusões e o corte 110 registrados neste checkpoint.

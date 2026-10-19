# Teste externo: protocolo de trabalho pendente

Este documento prepara a issue 12; **não é um pré-registro nem um manifesto
de congelamento**. A fonte indicada pelo usuário é MultiCaRe. A decisão de
escopo em `external-final-scope-decision.md` usa casos novos da fonte completa
como teste final e conserva a amostra local apenas como piloto.
Não usar este documento como evidência de que há gold independente, de que a
auditoria está completa ou de que o teste foi congelado.

## Fonte indicada e uso anterior

`project1/data/raw/cases.csv` contém 56 relatos com IDs únicos em 50 artigos;
`project1/data/raw/metadata.csv` contém os metadados dos 50 artigos. As cópias
em `project2_from_project1/data/raw/` são idênticas byte a byte. Os SHA-256 dos
arquivos de casos e metadados são, respectivamente,
`59a87e21cce7f511e99776e60e00138806b0f736b66273f73bad1e3957e99dee` e
`e2425091d44f0a79687026b58c760161a5eb61d8c65c947863f24d518a0a7ece`.
O Projeto 1 já aplicou seus extratores e produziu grafos sobre essa amostra;
esse uso anterior deve ser declarado e avaliado quanto à eventual influência
sobre os braços que serão comparados. A documentação do braço canônico de
entidades exclui explicitamente os vocabulários legados.

A amostra local tem somente 56 relatos e foi usada no Projeto 1. A versão
completa MultiCaRe v1.0 foi obtida do Zenodo. O pacote inicial de 100
candidatos foi ampliado para uma reserva de 140, preservando exatamente seu
prefixo, em `artifacts/external-multicare-continuation-pool140-v1/`. A regra e
a ampliação estão em `external-multicare-continuation-rule.md` e
`external-continuation-pool-amendment.md`; seus artigos não se repetem na
amostra local. Nenhum dos 56 casos locais conta para o teste final.

Uma auditoria preliminar na direção MultiCaRe → MCRC classificou os 56 relatos
como `released` no limiar 0,85. O relatório `multicare-versus-mcrc-pmid.json`
e o `manifest-pmid.json`, com hashes das entradas e do código, estão em
`artifacts/external-test-preliminary-audit/`. Eles não
comprovam ausência de duplicatas entre candidatos nem esclarecem se o uso
anterior do Projeto 1 influenciou os braços comparados. Esse relatório é
anterior à regra da reserva nova e não constitui o congelamento final.
Antes da correção de URLs PubMed, a auditoria não reconhecia nenhum PMID nos
53 documentos MCRC. Após a correção, reconhece 10. Os demais ainda podem ter
identificadores ausentes, portanto a cobertura deve ser verificada antes de
considerar o relatório definitivo. O sinal textual não elimina essa lacuna.

Há também uma divergência de bytes na dependência gold: o `mcrc.jsonl` atual
usa CRLF e tem SHA-256 `5104d36f…`, coincidente com o manifesto das partições,
enquanto os 233 registros de adjudicação exigem `28844ee5…`, que é o hash do
mesmo arquivo com finais de linha LF. O contrato valida hashes de bytes; é
necessário reconciliar a fonte, decisões e partições antes de executar os
braços, sem sobrescrever os artefatos existentes.

## Antes da seleção

Regra do piloto registrada antes das revisões de anotação: usar as 56 linhas do `cases.csv` identificado pelo
hash acima, com uma narrativa `case_text` por `case_id`, ordenadas
lexicograficamente por `case_id`. Exigir ID único, texto não vazio, um único
registro correspondente em `metadata.csv` e termos de uso compatíveis. Manter
os casos distintos do mesmo artigo como documentos distintos, sem presumir que
eles descrevem pacientes diferentes; auditar duplicatas e resolver vínculos de
paciente antes da inclusão. Excluir sobreposições conclusivas e não incluir
quarentenas; avançar sempre ao próximo ID na ordem. O comando
`scripts/prepare_external_multicare.py` aplicou a ordem e a auditoria textual;
o snapshot está em `artifacts/external-multicare-preparation-v1/` e tem status
`annotation_packet_not_gold_not_frozen`. A verificação de vínculos de paciente
e termos de reutilização ainda precisa de revisão antes do congelamento.

Registrar e fixar a regra documental antes de consultar resultados de modelos
sobre os candidatos. Ela deve especificar fonte e versão,
consulta completa, datas, idioma, licença e disponibilidade do texto, unidade
documental e de paciente, critérios de inclusão e exclusão, ordenação e regra
para substituir itens excluídos ou em quarentena. Guardar a lista completa de
candidatos na ordem original, a data de obtenção, os metadados e o hash do
snapshot. A ordem não pode depender de previsões, métricas ou suporte estimado
pelos modelos. Registrar também quando e onde o protocolo foi fixado.

Registrar os termos de reutilização de cada artigo com base em seus metadados
e na fonte original. A amostra local contém licenças distintas; não presumir
que todas autorizam os mesmos usos ou a redistribuição do texto.

No piloto, texto e offsets vêm diretamente de `cases.csv`. Na reserva nova,
vêm de `cases.parquet` da versão completa MultiCaRe v1.0; em ambos os casos,
o pacote preserva o texto-fonte sem normalização. O guia
de tipos, relações, revisões e adjudicação foi registrado em
`annotations/external-multicare-v1/GUIDELINES.md` antes das revisões.
Registrar a identidade e o tipo de cada revisor, o método de resolução de
desacordos e os controles de acesso em cada execução.
O guia de projeção existente resolve conflitos de candidatos já anotados;
sozinho ele não especifica a anotação exaustiva de novos documentos e relações.
As instruções completas para essa anotação devem ser fixadas antes da revisão.

## Auditoria e anotação

Auditar cada candidato MultiCaRe contra o MCRC e quaisquer outros dados usados
para treinamento e seleção, além de verificar duplicatas entre os próprios
candidatos. Registrar os hashes das referências e da entrada auditada,
configuração, IDs, evidências e decisões. A auditoria atual em
`src/leakage_audit.py` compara uma coleção protegida por chamada; é necessário
criar uma coleção para o MCRC e executar contra ela. Os fingerprints locais do
MultiCaRe representam os próprios candidatos: usá-los como referência de
exclusão contra a mesma amostra produziria uma auto-correspondência, não uma
auditoria válida contra treinamento. Seguir a
[ADR 0010](adr/0010-quarentenar-sobreposicoes-nao-resolvidas.md): excluir
coincidências conclusivas e não selecionar itens em quarentena. Registrar
exclusões, motivos e a reposição segundo a ordem pré-registrada.

Anotar cada documento liberado com os quatro tipos `Condition`, `Finding`,
`Factor` e `Modifier` e as cinco relações positivas `HAS_CONDITION`,
`HAS_FINDING`, `HAS_FACTOR`, `NEGATES` e `CAUSES`, seguindo as definições e
restrições dos contratos e das ADRs 0004 e 0007. `Patient` é uma âncora lógica,
não um quinto tipo de menção. Preservar texto e offsets, IDs e proveniência;
não inferir relações sem evidência. Cada documento deve ter duas anotações
independentes, sem acesso ao voto anterior, e uma decisão documentada para
divergências. Registrar identificadores, natureza dos revisores e justificativas;
revisões por modelo não equivalem a revisões humanas independentes.

As anotações e seu suporte só podem orientar a regra de parada, não a escolha
de modelo, features, regras ou hiperparâmetros. Manter anotações separadas das
configurações dos modelos e registrar qualquer acesso ou execução.

## Regra de parada e congelamento

Após a adjudicação de cada documento, contar entidades por tipo e relações
positivas por classe. O conjunto exige pelo menos 50 documentos novos e 100
entidades de cada tipo. Prosseguir na ordem registrada até que os mínimos de
documentos e entidades sejam atingidos e todas as cinco classes tenham pelo
menos 20 relações positivas, ou até o teto de 100 documentos. Ao atingir o teto,
marcar previamente como inconclusiva cada classe
com menos de 20 relações. Se algum tipo tiver menos de 100 entidades no teto,
o critério obrigatório de entidades não está satisfeito: não declarar o
conjunto conforme nem alterar silenciosamente a regra.

Somente após passar os critérios, congelar e registrar IDs e hashes dos textos,
anotações finais e revisões, esquema e diretrizes, relatório completo de
leakage, ordem e exclusões, contagens, classes inconclusivas, data e manifesto.
O manifesto deve vincular os hashes das entradas e saídas e a versão do código,
sem incluir texto clínico desnecessário. Verificar novamente os hashes antes da
primeira execução dos braços e manter o conjunto fechado a alterações.

## Estado atual e pendências

- O piloto local tem uma revisão completa dos 56 casos e revisões B/C parciais;
  o material permanece preservado, mas não entra no teste final. Três itens
  sem unidade clínica individual foram identificados no piloto.
- A reserva nova contém 140 candidatos liberados pela auditoria MCRC, com licença
  `CC BY` ou `CC0` e um caso `_01` por artigo. A triagem clínica dos primeiros
  111 foi concluída e excluiu 11 unidades; o 100º elegível é o candidato 110.
- As trilhas cegas foram pausadas por decisão do usuário em A110, B70 e C75. O
  maior prefixo adjudicado tem 70 candidatos, dos quais 63 são elegíveis, e 9
  relações `CAUSES`. Os hashes e o ponto de retomada estão em
  `external-review-checkpoint.md`; ainda não há gold ou manifesto congelado.
- Para retomar, completar B desde o candidato 71 e C desde o 76, adjudicar até
  o ponto de parada e então congelar. Registrar a limitação de revisores somente
  por modelo.

Até essas etapas serem realizadas, a issue 12 permanece aberta e a avaliação da
issue 13 continua bloqueada por ela.

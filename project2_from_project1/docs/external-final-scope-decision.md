# Escopo do teste externo final MultiCaRe

Decisão de 2026-10-19: os 56 relatos de `project1/data/raw/cases.csv` ficam
como **piloto metodológico**, fora do teste final. O Projeto 1 já executou
extratores sobre eles e seus resultados constam no repositório. Mesmo sem
gold plano prévio, esse histórico impede chamá-los de totalmente intocados.
Os votos independentes, a seleção e os relatórios do piloto são preservados
como artefatos de auditoria, sem entrarem no suporte ou nas métricas finais.

O teste final começa no primeiro ID liberado do pacote
`artifacts/external-multicare-continuation-pool140-v1/annotation-packet.jsonl`, preparado
da versão completa MultiCaRe segundo
`docs/external-multicare-continuation-rule.md`. O pacote exclui artigos da
amostra local. A reserva foi ampliada de 100 para 140 candidatos, sem mudar o
prefixo, conforme `external-continuation-pool-amendment.md`. Aplicam-se as mesmas regras
de menções e relações do guia v1 e a mesma revisão independente com terceiro
voto cego por item divergente. Excluir narrativas sem um paciente clínico
individual, registrar motivo e avançar ao próximo ID liberado.

Após cada documento elegível adjudicado, contar os quatro tipos de entidade e
as cinco relações. Congelar no primeiro prefixo com pelo menos 50 documentos,
100 menções de cada tipo e 20 relações de cada classe; caso contrário, alcançar
o teto de 100 documentos e declarar classes com menos de 20 relações
inconclusivas. Menos de 100 entidades de algum tipo no teto impede afirmar que
a issue 12 foi concluída. O teste permanece fechado para ajustes de modelos.

Esta decisão foi motivada pelo uso anterior do lote local, não por previsões de
modelos externos. A revisão A dos cinco primeiros candidatos novos começou
antes desta decisão; seus votos e a cronologia permanecem registrados. O
suporte do piloto local já era conhecido quando a regra de continuação foi
escrita, o que também será declarado no manifesto de congelamento.
Como a ordem é lexical por `case_id`, o prefixo final se concentra nos primeiros
IDs elegíveis da versão MultiCaRe, e não representa uma amostra aleatória de
todo o corpus. Essa limitação acompanha a interpretação das métricas finais.

# Reserva adicional para substituições na seleção externa

Em 2026-10-19, a triagem clínica por texto-fonte identificou itens inelegíveis
entre os 100 primeiros candidatos liberados, incluindo coortes agregadas,
revisão de literatura, relato veterinário e narrativas multipaciente. Assim,
100 candidatos brutos não asseguram o teto de 100 **documentos elegíveis** da
issue 12.

Aplica-se o mesmo código, fonte, hashes, ordenação, filtros e auditoria de
`external-multicare-continuation-rule.md` até obter 140 candidatos liberados.
O pacote inicial de 100 e suas proveniências permanecem preservados. Os 100
primeiros documentos do pacote ampliado devem ser byte a byte iguais ao
prefixo anterior; só um sufixo adicional é anexado. Votos já emitidos não são
editados. Revisões cumulativas posteriores vinculam-se ao hash do pacote
ampliado e registram a troca de snapshot no manifesto de proveniência.
Verificação realizada: os primeiros 336.062 bytes do pacote de 140 são
idênticos aos 336.062 bytes completos do pacote de 100. Os SHA-256 dos pacotes
são, respectivamente, `f8a485f661471a651fc4de166425941f58924fcbd78af9ff6fc4194f0703de85`
e `1c6a730de5b2239b6b44642c7a8cceecad87c134c08da3ca049fbbb32690cbbe`.
O pacote local usado apenas como referência conservadora na checagem de
duplicatas tem SHA-256 `997d490dd5742dd61f14d7d29c2b05fc87eff7138a2bfaa39023b3e11fc9f420`.

A regra de parada continua em 100 documentos **elegíveis**, no primeiro prefixo
que atinja os suportes ou no teto. Candidatos além do necessário não entram no
teste nem nas contagens. Esta ampliação decorre apenas da elegibilidade da
fonte, antes de qualquer previsão de modelo externo; a cronologia da triagem
e dos votos deve constar no congelamento.

A triagem de fonte foi concluída para os 111 primeiros candidatos sem consultar
previsões dos modelos avaliados. Dez foram excluídos por não representarem um
relato clínico individual humano. O 100º documento elegível é, portanto, o
candidato 109 (`PMC10032400_01`); os candidatos 110 e 111 ficam como margem de
auditoria e não entram no teto, salvo mudança previamente documentada nas
decisões de elegibilidade.

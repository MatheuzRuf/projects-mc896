# 08 — Executar k-NN sobre candidatos previstos

**What to build:** adaptar o classificador n-gramas + k-NN já selecionado para treinar com candidatos gold dos documentos de treino e classificar, sem correção, candidatos gerados por cada reconhecedor de entidades.

**Blocked by:** 06 — Gerar candidatos não rotulados das entidades previstas.

**Status:** ready-for-agent

- [ ] O vocabulário n-grama e os vizinhos são ajustados somente com candidatos gold do treino.
- [ ] Candidatos previstos não precisam nem expõem rótulos gold para serem transformados em features.
- [ ] O mesmo modelo relacional do fold é aplicado aos candidatos do dicionário e do CRF.
- [ ] Toda previsão é a relação hipotética ou `NO_RELATION` e preserva a fração de votos sem chamá-la de confiança.
- [ ] A execução produz previsões e manifesto completo mantendo alinhamento um-para-um com os candidatos.

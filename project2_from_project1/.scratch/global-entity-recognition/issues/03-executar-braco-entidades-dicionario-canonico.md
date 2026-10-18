# 03 — Executar o braço de entidades por dicionário canônico

**What to build:** executar um reconhecedor lexical comparável que aprende seu dicionário somente das menções da partição de treino e produz entidades canônicas, métricas estritas e manifesto para uma partição nunca consultada durante a construção do vocabulário.

**Blocked by:** 02 — Produzir a projeção gold plana adjudicada.

**Status:** ready-for-agent

- [ ] O dicionário contém somente formas observadas no treino do fold e registra o hash de sua origem.
- [ ] Superfícies ambíguas usam o tipo mais frequente do treino e empates causam abstenção.
- [ ] Correspondências mais longas têm prioridade e a saída é plana, não sobreposta e offset-preserving.
- [ ] A execução produz `EntityPrediction`, avaliação estrita e manifesto sem ler rótulos da partição prevista.
- [ ] O relatório inclui micro-F1, macro-F1, F1 por tipo e diagnósticos de OOV, limite e comprimento.

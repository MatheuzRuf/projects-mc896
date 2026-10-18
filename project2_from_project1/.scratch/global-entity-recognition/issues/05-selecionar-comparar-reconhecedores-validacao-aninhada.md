# 05 — Selecionar e comparar reconhecedores por validação aninhada

**What to build:** executar dicionário e semi-Markov CRF nos mesmos cinco folds externos documentais, selecionando a regularização do CRF somente em três folds internos, e publicar uma comparação exploratória completa sem usar o teste MCRC consumido.

**Blocked by:** 03 — Executar o braço de entidades por dicionário canônico; 04 — Executar o braço de entidades por semi-Markov CRF.

**Status:** ready-for-agent

- [ ] Nenhum documento atravessa treino interno, validação interna ou validação externa na mesma rodada.
- [ ] A busca considera somente `lambda` L2 em `0.01`, `0.1`, `1` e `10` e seleciona por macro-F1 estrita.
- [ ] Empates usam menor variância e depois regularização mais forte de maneira determinística.
- [ ] Dicionário e CRF são avaliados nos mesmos folds e nenhum resultado remove um braço da comparação.
- [ ] O resumo agrega métricas, variância, convergência, tempos, análises de erro e manifestos dos folds.
- [ ] O teste MCRC consumido não é aberto nem usado para escolher qualquer decisão.

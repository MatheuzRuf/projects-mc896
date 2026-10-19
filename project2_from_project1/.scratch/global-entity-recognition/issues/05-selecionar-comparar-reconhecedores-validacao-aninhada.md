# 05 — Selecionar e comparar reconhecedores por validação aninhada

**What to build:** executar dicionário e semi-Markov CRF nos mesmos cinco folds externos documentais, selecionando a regularização do CRF somente em três folds internos, e publicar uma comparação exploratória completa sem usar o teste MCRC consumido.

**Blocked by:** 03 — Executar o braço de entidades por dicionário canônico; 04 — Executar o braço de entidades por semi-Markov CRF.

**Status:** done

**Implementation note (2026-10-19):** `src/entity_recognition/nested_cv.py`,
CLI `scripts/run_entity_nested_cv.py` e `tests/test_entity_nested_cv.py`. O
estudo completo rodou em 3,96 h (65 treinos do CRF). Macro-F1 estrita externa:
dicionário 0,121 ± 0,013, CRF 0,285 ± 0,044. λ escolhido: 0,1 em quatro folds e
1 em um. Nenhum modelo final convergiu dentro de 250 iterações. Resultados e
leitura em `docs/entity-recognizer-comparison.md`.

- [x] Nenhum documento atravessa treino interno, validação interna ou validação externa na mesma rodada.
- [x] A busca considera somente `lambda` L2 em `0.01`, `0.1`, `1` e `10` e seleciona por macro-F1 estrita.
- [x] Empates usam menor variância e depois regularização mais forte de maneira determinística.
- [x] Dicionário e CRF são avaliados nos mesmos folds e nenhum resultado remove um braço da comparação.
- [x] O resumo agrega métricas, variância, convergência, tempos, análises de erro e manifestos dos folds.
- [x] O teste MCRC consumido não é aberto nem usado para escolher qualquer decisão.

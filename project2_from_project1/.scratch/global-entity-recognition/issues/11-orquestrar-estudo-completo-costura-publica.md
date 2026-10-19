# 11 — Orquestrar o estudo completo pela costura pública

**What to build:** oferecer uma entrada pública única que execute uma fixture sintética e as partições de desenvolvimento do texto bruto ao grafo para todos os braços, publicando contratos, avaliações, análises e manifestos sem gate de desempenho.

**Blocked by:** 05 — Selecionar e comparar reconhecedores por validação aninhada; 09 — Avaliar a matriz fatorial de relações end-to-end; 10 — Projetar e avaliar o Knowledge Graph versionado.

**Status:** done

**Implementation note (2026-10-19):** entrada única `scripts/run_full_study.py`
(`src/study_pipeline.py`) com fixture sintética (`src/synthetic_fixture.py`,
`--synthetic`) e costura testada em `tests/test_full_study.py`. Um CRF não
finito marca apenas os braços `crf+*` do fold afetado
(`skipped_failed_recognizer`). A execução real em `artifacts/study-v1`
reutilizou, por cache indexado por hashes, os treinos do ticket 05 e
reproduziu exatamente os resultados dos tickets 09 e 10. O manifesto raiz lista
401 artefatos filhos. Guia operacional em `docs/running-the-study.md`. Para
um manifesto com `dirty=false`, repetir a execução depois do commit (cerca de
70 s com cache).

- [x] Uma fixture sintética percorre dicionário e CRF, candidatos, regras e k-NN, avaliações e grafo pela mesma interface pública.
- [x] A execução real de desenvolvimento usa as partições documentais congeladas e não abre o teste MCRC consumido.
- [x] Todos os braços produzem artefatos separados e permanecem no resumo mesmo quando inferiores ou não convergidos.
- [x] O manifesto raiz referencia commit, dependências, configurações, partições, tempos e hashes de todos os artefatos filhos.
- [x] Uma falha numericamente inválida identifica somente o braço afetado e não é mascarada por fallback.
- [x] Documentação operacional permite repetir o fluxo em CPU em outra máquina.

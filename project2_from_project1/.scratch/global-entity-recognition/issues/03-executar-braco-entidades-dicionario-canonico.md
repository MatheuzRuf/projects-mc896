# 03 — Executar o braço de entidades por dicionário canônico

**What to build:** executar um reconhecedor lexical comparável que aprende seu dicionário somente das menções da partição de treino e produz entidades canônicas, métricas estritas e manifesto para uma partição nunca consultada durante a construção do vocabulário.

**Blocked by:** 02 — Produzir a projeção gold plana adjudicada.

**Status:** done

**Implementation note (2026-10-18):** dicionário, avaliador estrito
compartilhado, executor por fold e CLI `scripts/run_entity_dictionary.py`
implementados e testados com fixtures sintéticas
(`tests/test_entity_dictionary.py`). A projeção gold oficial do ticket 02 já existe e o executor roda nos dados reais. Ver `docs/entity-dictionary.md`.

- [x] O dicionário contém somente formas observadas no treino do fold e registra o hash de sua origem.
- [x] Superfícies ambíguas usam o tipo mais frequente do treino e empates causam abstenção.
- [x] Correspondências mais longas têm prioridade e a saída é plana, não sobreposta e offset-preserving.
- [x] A execução produz `EntityPrediction`, avaliação estrita e manifesto sem ler rótulos da partição prevista.
- [x] O relatório inclui micro-F1, macro-F1, F1 por tipo e diagnósticos de OOV, limite e comprimento.

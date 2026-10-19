# 10 — Projetar e avaliar o Knowledge Graph versionado

**What to build:** transformar previsões end-to-end congeladas em um grafo JSONL v2 com camada de menção e camada conceitual, medir nós e arestas contra a projeção gold e oferecer exportações compatíveis para visualização e consumidores legados.

**Blocked by:** 01 — Expandir os contratos supervisionados versionados; 09 — Avaliar a matriz fatorial de relações end-to-end.

**Status:** done

**Implementation note (2026-10-19):** projeção em duas camadas
(`src/supervised_graph.py`), estudo por fold e braço (`src/graph_study.py`), CLI
`scripts/run_graph_projection.py` e `tests/test_supervised_graph.py`. As
exportações CSV seguem o formato legado e funcionam com `visualize_case`; as
cores de `Condition`, `Factor` e `Modifier` foram acrescentadas à paleta. F1 de
arestas conceituais: dicionário+regras 0,080, dicionário+k-NN 0,125, CRF+regras
0,152, CRF+k-NN 0,208. Ver `docs/knowledge-graph-v2.md`.

- [x] Cada menção prevista permanece identificável por documento, offsets, tipo, método e versão.
- [x] Conceitos usam documento, tipo e superfície normalizada, sem fundir superfícies iguais de tipos diferentes.
- [x] A agregação preserva os IDs e relações das menções que sustentam cada nó e aresta conceitual.
- [x] Nós e arestas conceituais recebem métricas exatas secundárias sem substituir as métricas de menções e relações.
- [x] CSV e NetworkX são derivados do JSONL canônico e não perdem referências essenciais de proveniência.
- [x] A projeção é determinística e produz manifesto com esquema, configuração e hashes.

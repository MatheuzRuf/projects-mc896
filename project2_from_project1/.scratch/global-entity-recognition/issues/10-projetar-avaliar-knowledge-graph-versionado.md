# 10 — Projetar e avaliar o Knowledge Graph versionado

**What to build:** transformar previsões end-to-end congeladas em um grafo JSONL v2 com camada de menção e camada conceitual, medir nós e arestas contra a projeção gold e oferecer exportações compatíveis para visualização e consumidores legados.

**Blocked by:** 01 — Expandir os contratos supervisionados versionados; 09 — Avaliar a matriz fatorial de relações end-to-end.

**Status:** ready-for-agent

- [ ] Cada menção prevista permanece identificável por documento, offsets, tipo, método e versão.
- [ ] Conceitos usam documento, tipo e superfície normalizada, sem fundir superfícies iguais de tipos diferentes.
- [ ] A agregação preserva os IDs e relações das menções que sustentam cada nó e aresta conceitual.
- [ ] Nós e arestas conceituais recebem métricas exatas secundárias sem substituir as métricas de menções e relações.
- [ ] CSV e NetworkX são derivados do JSONL canônico e não perdem referências essenciais de proveniência.
- [ ] A projeção é determinística e produz manifesto com esquema, configuração e hashes.

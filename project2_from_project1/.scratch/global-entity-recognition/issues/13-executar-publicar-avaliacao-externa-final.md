# 13 — Executar e publicar a avaliação externa final

**What to build:** executar todos os braços congelados no teste externo uma única vez e publicar a comparação final de entidades, relações e grafo com proveniência completa, resultados desfavoráveis preservados e classes sem suporte declaradas inconclusivas.

**Blocked by:** 11 — Orquestrar o estudo completo pela costura pública; 12 — Selecionar, anotar e congelar o teste externo.

**Status:** completed-with-user-authorized-provisional-snapshot

**Entrega (2026-10-19):** por decisão explícita do usuário, os 63 documentos
elegíveis disponíveis foram aceitos como escopo desta execução. A revisão não
atinge a regra de congelamento originalmente planejada; portanto o resultado é
publicado como snapshot provisório autorizado pelo usuário, e não como teste
externo definitivo. Ver `docs/external-final-evaluation.md`.

- [x] Configurações, vocabulários, regras e hiperparâmetros foram congelados antes da leitura das anotações externas.
- [x] Dicionário/CRF × regras/k-NN foram executados sem ajuste posterior e o teto gold-entity foi reportado separadamente.
- [x] O relatório inclui métricas estritas de entidades e relações, métricas secundárias do grafo e decomposição de erros.
- [x] Braços inferiores, CRF finito não convergido e resultados parcialmente inconclusivos permanecem visíveis.
- [x] `CAUSES`, abaixo do suporte pré-registrado, está identificado como inconclusivo sem agregação que esconda a limitação.
- [x] Manifestos e hashes permitem reproduzir os artefatos publicados; a retomada de pontuação verifica as predições já congeladas.

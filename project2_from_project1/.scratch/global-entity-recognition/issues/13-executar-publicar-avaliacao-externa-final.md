# 13 — Executar e publicar a avaliação externa final

**What to build:** executar todos os braços congelados no teste externo uma única vez e publicar a comparação final de entidades, relações e grafo com proveniência completa, resultados desfavoráveis preservados e classes sem suporte declaradas inconclusivas.

**Blocked by:** 11 — Orquestrar o estudo completo pela costura pública; 12 — Selecionar, anotar e congelar o teste externo.

**Status:** ready-for-agent

- [ ] Configurações, vocabulários, regras e hiperparâmetros são congelados antes da leitura das anotações externas.
- [ ] Dicionário/CRF × regras/k-NN são executados sem ajuste posterior e o teto gold-entity é reportado separadamente.
- [ ] O relatório final inclui métricas estritas de entidades e relações, métricas secundárias do grafo e decomposição de erros.
- [ ] Braços inferiores, não convergidos de forma finita ou parcialmente inconclusivos permanecem visíveis.
- [ ] Classes abaixo do suporte pré-registrado são identificadas como inconclusivas, sem agregação que esconda a limitação.
- [ ] Manifestos e hashes permitem reproduzir exatamente cada resultado publicado.

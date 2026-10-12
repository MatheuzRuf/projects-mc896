# 01 — Isolar e validar o novo pacote

**What to build:** tornar a evolução do projeto executável como um pacote independente, preservando o comportamento legado relevante e eliminando dependências acidentais do `project1`.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] A suíte do novo pacote pode ser descoberta e executada isoladamente.
- [ ] Nenhum módulo ou teste do novo pacote importa implementações internas do `project1`.
- [ ] Os testes legados copiados para o novo pacote continuam verdes depois da migração de namespace.
- [ ] O `project1` permanece funcionalmente inalterado.
- [ ] A configuração de empacotamento e testes reconhece explicitamente o novo pacote.


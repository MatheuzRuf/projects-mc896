# 06 — Produzir o dataset supervisionado congelado

**What to build:** transformar um corpus externo aprovado em um dataset supervisionado reproduzível, com partições documentais, candidatos compatíveis e política de negativos registrada.

**Blocked by:** 05 — Decidir a adequação do MCRC.

**Status:** ready-for-agent

- [ ] Aproximadamente 20% dos documentos aprovados formam um teste fixo e protegido.
- [ ] Os documentos restantes formam folds de validação cruzada agrupados por documento.
- [ ] Nenhum documento, menção ou relação atravessa as fronteiras entre treino, validação e teste.
- [ ] Documentos excluídos ou em quarentena não aparecem em nenhuma partição.
- [ ] Pares candidatos respeitam compatibilidade de tipos e a configuração de janela, incluindo relações intersentenciais.
- [ ] O treino recebe no máximo dois negativos por relação positiva, enquanto validação e teste mantêm todos os candidatos elegíveis.
- [ ] Mesma entrada, configuração e semente reproduzem partições e candidatos idênticos.
- [ ] O manifesto registra IDs das partições, semente, política de candidatos, política de negativos e contagens.


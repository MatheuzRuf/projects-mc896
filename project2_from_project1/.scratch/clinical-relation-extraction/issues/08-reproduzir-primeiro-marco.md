# 08 — Reproduzir o primeiro marco de ponta a ponta

**What to build:** permitir que uma execução documentada e limpa demonstre o marco completo de preparação e avaliação, produzindo todos os manifestos e relatórios necessários para iniciar os extratores experimentais posteriores.

**Blocked by:** 06 — Produzir o dataset supervisionado congelado; 07 — Avaliar previsões pelo contrato comum.

**Status:** ready-for-agent

- [ ] Uma interface documentada executa a preparação desde o corpus seguro e a coleção protegida até o dataset congelado.
- [ ] Uma interface documentada avalia previsões compatíveis contra o ouro preparado.
- [ ] A execução reúne configuração, semente, versões, hashes, partições, contagens, política de negativos, métricas, matriz de confusão, tempos e relatório de leakage.
- [ ] Uma nova execução com as mesmas entradas reproduz os artefatos determinísticos esperados.
- [ ] Dados brutos, textos clínicos, modelos e artefatos volumosos permanecem fora do controle de versão.
- [ ] A suíte automatizada comprova as duas costuras principais e os invariantes focados.
- [ ] O relatório final declara o marco pronto sem alegar treinamento definitivo de kNN, regra experimental, híbrido, POS tagging ou dependency parsing.

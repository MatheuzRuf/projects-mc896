# 02 — Preparar e recarregar um documento clínico completo

**What to build:** permitir que um documento clínico representativo atravesse validação, serialização em JSONL e recarga sem perder casos, pacientes, menções complexas ou relações nativas e canônicas.

**Blocked by:** 01 — Isolar e validar o novo pacote.

**Status:** ready-for-agent

- [ ] Um documento válido pode conter `ClinicalCase`, `Patient`, menções e relações com identificadores estáveis.
- [ ] Uma menção preserva um ou mais intervalos `[start, end)`, aninhamento possível, múltiplos tipos nativos e tipo canônico separado.
- [ ] Uma relação preserva direção, rótulo nativo, rótulo canônico e referências válidas às suas extremidades.
- [ ] O vínculo `ClinicalCase --DESCRIBES--> Patient` é representado sem tratar os dois conceitos como idênticos.
- [ ] A negação liga explicitamente um modificador ao achado negado e rejeita combinações positivas incompatíveis.
- [ ] Uma fixture completa mantém conteúdo semanticamente equivalente após gravar e recarregar o JSONL.
- [ ] Entradas com intervalos, referências ou relações inválidas falham com mensagens úteis.


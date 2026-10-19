# MultiCaRe v1: regra de continuidade da seleção externa

Registrada em 2026-10-19, antes de consultar qualquer previsão dos modelos no
teste externo. A revisão das anotações dos 56 relatos locais mostrou suporte
insuficiente de `CAUSES`; esta extensão mantém o limite de 100 documentos da
issue 12. Ela não altera o pacote, os votos ou os hashes dos 56 relatos.

A fonte é o registro MultiCaRe v1.0 [Zenodo 10079370](https://zenodo.org/records/10079370),
arquivos `cases.parquet` (MD5 `b1f7f7a14844e6ea66b5f14f52197e4a`) e
`metadata.parquet` (MD5 `d0db089d8e4fe5deb998ede53abcc0b9`). O texto do
caso é usado sem normalização. A continuação percorre `case_id` em ordem
lexicográfica, depois dos 56 IDs locais já registrados. Exclui todos os 50
artigos da amostra local, aceita somente o primeiro caso `_01` de cada novo
artigo, licenças `CC BY` ou `CC0`, e narrativas entre 300 e 5000 caracteres.
Os SHA-256 locais dos parquets são `1f988382b57379d6e39c285895510b4d210bb704a4b8b9c7535349170eb16da1`
e `71a0f2d56bbac4721bc99f25b488848ff9d665c3a9be361e3fbd02b96107ba80`,
respectivamente. O snapshot tem 100 candidatos liberados, o que dá margem
para substituir itens inelegíveis antes do teto total de 100 documentos.
Esses filtros foram definidos por unidade de paciente, possibilidade de
reutilização e viabilidade de revisão, sem consultar rótulos ou resultados de
modelo. Artigos sem narrativa clínica individual, narrativas não inglesas,
duplicatas e casos com sobreposição ou quarentena frente ao MCRC são excluídos
com motivo registrado; o próximo `case_id` ocupa a posição. A auditoria textual
usa os mesmos limiares do pacote local: 0,85 para a comparação com MCRC e
contenção de shingles de cinco tokens para duplicatas entre candidatos.

Os relatos locais considerados não elegíveis após a leitura da fonte também
serão excluídos apenas da seleção final, com justificativa e votos originais
preservados. A seleção e a regra de parada dependem do suporte das anotações;
nenhuma decisão de arquitetura, feature ou hiperparâmetro pode usá-las. O
registro desta extensão ocorreu depois de observar suporte parcial dos rótulos,
o que deve ser declarado no manifesto final; não é um pré-registro cego quanto
à distribuição de classes.

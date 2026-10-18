# Derivar uma segmentação canônica plana sem alterar o gold

O reconhecedor end-to-end usará um semi-Markov CRF para escolher globalmente
menções contíguas, não sobrepostas e com exatamente um tipo canônico. Essa
segmentação será uma projeção derivada e auditável das anotações do corpus,
nunca uma substituição do contrato lossless definido pela ADR 0008. A camada
original continuará preservando duplicidades, sobreposições, spans complexos e
tipos nativos; conflitos que não puderem ser projetados sem ambiguidade deverão
ser registrados pela política de preparação do reconhecedor.

Duplicatas exatas serão colapsadas na projeção. Sobreposições e conflitos de
tipo serão adjudicados por dois revisores independentes, com consenso ou
terceiro voto para divergências, preservando decisão, justificativa e
proveniência. O futuro ticket dessa adjudicação deverá começar com um aviso
explícito para uso de modelo com alto nível de reasoning.

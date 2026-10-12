# Converter o corpus pickle para formato seguro

O pickle publicado pelo Medical Case Report Corpus será tratado apenas como artefato externo de ingestão, fixado por commit e SHA-256, inspecionado antes da abertura e convertido uma vez para JSONL. O corpus original e os textos convertidos permanecerão fora do Git, e o pipeline experimental consumirá somente o formato convertido, reduzindo o risco de execução arbitrária e tornando a entrada auditável e portátil.

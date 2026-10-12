# Isolar a evolução experimental do Project 1

A evolução será mantida em um pacote irmão chamado `project2_from_project1`, enquanto `project1` permanecerá funcional e sem alterações indiscriminadas. Escolhemos a separação porque as novas regras de preparação de dados, avaliação supervisionada e auditoria de leakage formam um experimento distinto; reutilização ocorrerá por contratos explícitos, evitando que testes ou imports da evolução executem silenciosamente o pacote original.

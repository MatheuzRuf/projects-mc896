# Separar ClinicalCase e Patient

O novo grafo representará a unidade documental como `ClinicalCase` e a pessoa descrita como `Patient`, ligados por `DESCRIBES`. Identificadores, artigo e proveniência pertencem ao caso; condições, achados e fatores explicitamente atribuídos pertencem ao paciente. Essa separação substitui a âncora artificial única do Project 1 e evita confundir identidade documental com entidade clínica.

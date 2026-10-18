# Reconhecimento global de entidades clínicas

Status: entrevista de design concluída; pronta para especificação.

## Intenção

Substituir, em uma fase de implementação posterior, o reconhecimento de
entidades baseado exclusivamente em dicionário por um reconhecedor estatístico
capaz de escolher globalmente quais palavras formam uma menção e qual é seu
tipo canônico. A saída alimentará a geração de candidatos, o classificador de
relações e a projeção no Knowledge Graph.

Esta nota registra decisões para uma especificação futura. Ela não autoriza
nem descreve trabalho já implementado.

## Fatos observados

- O pipeline n-gramas + k-NN atual classifica relações entre entidades gold;
  ele não reconhece entidades em texto bruto.
- A construção de grafo herdada do Project 1 ainda reconhece entidades por
  correspondência exata com termos e aliases dos vocabulários.
- O MCRC convertido possui 53 documentos e 4.294 menções: 347 `Condition`,
  3.248 `Finding`, 363 `Factor` e 336 `Modifier`.
- Ao agrupar por documento, spans e tipo canônico, restam 3.296 alvos únicos;
  998 registros repetidos podem ser colapsados para o reconhecimento de
  entidades sem alterar o corpus original.
- O corpus possui 269 pares de menções com spans distintos sobrepostos e 68
  grupos em que o mesmo span recebe tipos canônicos diferentes.
- Nenhuma menção do MCRC convertido atual é descontínua ou possui mais de um
  tipo nativo, embora o contrato interno permita ambos os casos.
- Aproximadamente 99% das menções possuem até nove palavras por separação em
  espaços; 31 menções possuem mais de dez palavras e existem casos extremos
  de até 25 palavras.
- Nove registros de menção atravessam limites produzidos pelo separador
  simples de sentenças atualmente usado. A inspeção confirmou que todos são
  falsos limites após abreviações ou unidades, como `H. influenzae`,
  `M. tuberculosis`, `S. aureus`, `E. histolytica` e `mg. twice`; não são
  menções semanticamente intersentenciais.
- A auditoria do pickle verificado confirmou que os textos de todas as 4.294
  anotações coincidem exatamente com os intervalos registrados na fonte. Não
  há evidência de que o conversor tenha deslocado os offsets.
- Um tokenizador preliminar que separa palavras e pontuação preservando offsets
  alinha 3.265 dos 3.296 alvos únicos. Os 31 restantes incluem espaços nas
  bordas ou fragmentos de palavra e precisam ser resolvidos na adjudicação. Sob
  essa tokenização, menções alinhadas chegam a 35 tokens, de modo que o limite
  anterior de 25, estimado por espaços, não cobre a representação proposta.
- O tokenizador legado não devolve offsets e normaliza o texto antes da
  tokenização. Ele não pode alinhar previsões ao texto original sem um novo
  contrato de tokens com offsets.
- O nó do grafo possui um `NodeType` único. Cada aresta possui um tipo de
  relação único, enquanto um mesmo nó pode participar de várias arestas.
- O `NodeType` legado contém `Finding`, mas não contém `Condition`, `Factor`
  nem `Modifier`. Portanto, os quatro tipos canônicos supervisionados ainda
  não podem ser projetados diretamente no grafo legado sem uma decisão de
  evolução ou mapeamento do esquema.
- O teste fixo do MCRC já foi consumido na avaliação do classificador de
  relações e não deve orientar novas decisões de modelo.
- O ambiente local usa Python 3.14 no Windows e não possui PyTorch, JAX ou
  TensorFlow instalados. A documentação oficial informa suporte atual do
  PyTorch no Windows para Python 3.10 a 3.14, tornando-o uma opção compatível
  para diferenciação automática em CPU:
  <https://pytorch.org/get-started/locally/>.
- O repositório possui o classificador de relações k-NN e o extrator legado de
  regras, mas eles ainda não formam uma matriz fatorial sob o mesmo contrato.
  O método híbrido aparece no vocabulário do estudo, mas não está implementado.
- Os cinco folds atuais separam documentos: cada um dos 42 documentos de
  desenvolvimento aparece uma vez em validação e nos treinos dos outros folds.
  O protocolo atual do k-NN usa esses mesmos folds para escolher configuração,
  portanto seus resultados de seleção são exploratórios e não substituem um
  teste novo e intocado.
- O extrator por dicionário legado agrega todas as ocorrências do mesmo conceito
  e tipo em um único objeto com várias posições. O contrato do reconhecedor
  supervisionado, por outro lado, precisa manter cada ocorrência textual como
  uma menção independente antes de qualquer normalização no grafo.
- O contrato `Mention` atual descreve anotação de corpus e exige tipos nativos;
  reutilizá-lo diretamente para previsões obrigaria a inventar proveniência.
  As previsões de entidades precisam de um contrato serializado próprio antes
  de serem projetadas em documentos ou no grafo.
- O construtor legado identifica nós por caso e rótulo normalizado, colapsando
  ocorrências iguais antes da construção das relações. Isso não preserva os
  endpoints textuais necessários à avaliação end-to-end estrita e contraria a
  separação entre menção e conceito já decidida na ADR 0003.
- Os CSVs legados do grafo guardam proveniência em uma coluna textual genérica
  de atributos. O braço supervisionado precisará de uma projeção versionada e
  auditável antes de decidir se ou como exportá-la nesse formato.
- A projeção preliminar por texto normalizado e tipo reúne os 3.296 alvos
  únicos em 2.632 conceitos documentais. Existem 63 combinações de documento e
  superfície que aparecem com mais de um tipo, portanto o tipo canônico precisa
  fazer parte da identidade conceitual para evitar colisões.
- O esquema supervisionado contém apenas 47 relações `CAUSES` nos 53 documentos
  do corpus completo, enquanto as demais relações positivas têm suporte maior.
  Um teste externo com tamanho fixado apenas por número de documentos pode não
  sustentar conclusões por classe e precisa também de um critério mínimo de
  suporte pré-registrado.

## Decisões confirmadas

### ER-D01 — Inferência global por semi-Markov CRF

O reconhecedor escolherá segmentos textuais completos e seus tipos por
inferência global, em vez de classificar palavras ou spans de maneira
independente. A família de modelo escolhida é semi-Markov CRF.

### ER-D02 — Saída canônica plana

A saída operacional será uma segmentação canônica plana: as menções
previstas não se sobrepõem e cada uma recebe exatamente um tipo canônico.
Tokens fora de menções permanecem fora de entidade.

### ER-D03 — Multiplicidade pertence às arestas

Um nó projetado no Knowledge Graph possui um tipo canônico. Ele pode participar
de várias relações, mas cada aresta continua possuindo um único rótulo.

### ER-D04 — Preservar o gold original

As anotações originais, inclusive duplicidades, sobreposições e conflitos de
tipo, não serão modificadas. O treinamento do reconhecedor dependerá de uma
projeção gold plana derivada e auditável.

### ER-D05 — Duas camadas de anotação

O contrato lossless continua sendo a representação fiel do corpus. A
segmentação plana será uma camada derivada, criada para o reconhecedor
end-to-end e para a projeção no grafo. A ADR 0008 permanece válida para a
camada de origem; a ADR 0011 formaliza a fronteira derivada.

### ER-D06 — Inventário supervisionado inicial

O semi-Markov CRF aprenderá os quatro tipos sustentados pelo MCRC:
`Condition`, `Finding`, `Factor` e `Modifier`, além do estado fora de entidade.
Tipos adicionais do Knowledge Graph permanecem fora deste modelo enquanto não
existir supervisão semanticamente compatível.

### ER-D07 — Avaliação sem reutilizar o teste consumido

Seleção de modelo e hiperparâmetros usarão somente os cinco folds dos 42
documentos de desenvolvimento existentes. Uma afirmação final end-to-end
dependerá de um novo teste externo anotado. O teste fixo MCRC já consumido não
orientará decisões do reconhecedor.

### ER-D08 — Dicionário como baseline independente

O reconhecedor por dicionário permanecerá como baseline independente. O
semi-Markov CRF formará um braço experimental end-to-end completo e suas
predições não serão substituídas silenciosamente por fallback do dicionário,
mesmo quando seu resultado for inferior. Tipos não cobertos pelo MCRC
permanecem fora do braço supervisionado inicial.

### ER-D09 — Versionar e ampliar o esquema do grafo

O esquema do Knowledge Graph será versionado para representar diretamente
`Condition`, `Finding`, `Factor` e `Modifier`, bem como as relações canônicas
supervisionadas. Esses tipos não serão forçados para categorias legadas apenas
por semelhança de nomes.

### ER-D10 — Expandir os modelos somente depois do pipeline completo

A primeira entrega end-to-end permanecerá limitada aos tipos de entidades e
relações sustentados pelo MCRC. Depois de o pipeline completo estar funcional
e avaliado, o projeto investigará a expansão de todos os modelos para as
classificações legadas mais específicas, porque elas podem melhorar a leitura
humana do grafo. Cada nova distinção dependerá de anotações compatíveis ou de
um mapeamento semântico explícito, nova versão do esquema e nova avaliação;
legibilidade isolada não comprova equivalência clínica.

### ER-D11 — Adjudicar explicitamente a projeção gold plana

Duplicatas com o mesmo documento, spans e tipo serão colapsadas
automaticamente na projeção. Sobreposições e conflitos de tipo restantes
receberão decisões explícitas e auditáveis, sem prioridade automática por
comprimento ou classe.

### ER-D12 — Inferência por sentença com offsets

O semi-Markov CRF operará por sentença, preservando offsets no texto original.
O separador será determinístico e protegerá iniciais, abreviações clínicas e
unidades. A revisão mostrou que os nove cruzamentos do separador anterior são
falsos limites após abreviações, e não menções intersentenciais reais. Nenhum
cruzamento futuro será descartado silenciosamente.

### ER-D13 — Segmentos de até 40 tokens

O limite inicial será de 40 tokens por menção. Esta decisão substitui o limite
anterior de 25, que havia sido estimado por separação em espaços e não cobria a
tokenização final de palavras e pontuação, na qual o máximo observado é 35.
Violações futuras causarão diagnóstico e serão registradas, em vez de
truncamento silencioso.

### ER-D14 — Modelo principal sem features do dicionário

O semi-Markov CRF principal não receberá correspondências do dicionário como
features. Isso mantém independência metodológica e permite comparar um modelo
aprendido com o baseline lexical sem dependência oculta.

### ER-D15 — Comparação sem porta de promoção

O objetivo é um estudo comparativo, não a escolha antecipada de um sistema de
produção. O braço semi-Markov CRF será executado como pipeline end-to-end
completo mesmo que tenha desempenho inferior ao dicionário. Métricas servem
para comparar e explicar os métodos, não para omitir resultados desfavoráveis.

### ER-D16 — Diferenciação automática para o modelo segmental

A futura implementação usará um framework com diferenciação automática para
a programação dinâmica e o treinamento do semi-Markov CRF. A escolha concreta
será PyTorch em CPU. A programação dinâmica do semi-Markov CRF permanecerá no
projeto, usando autograd e operações numericamente estáveis, como `logsumexp`,
sem delegar o núcleo do modelo a uma biblioteca estruturada externa.

### ER-D17 — Adjudicação humana em ticket dedicado

A projeção gold plana será produzida por dois revisores independentes. Casos
em que as decisões divergirem serão resolvidos por consenso ou por um terceiro
voto, preservando a decisão, sua justificativa e a proveniência dos revisores.

Esse trabalho deverá ser descrito em um ticket próprio durante a etapa de
planejamento de implementação. Por exigir avaliação semântica cuidadosa, o
ticket deverá começar com um aviso explícito para que o agente responsável
seja configurado com um modelo de alto nível de reasoning antes de executar a
adjudicação. O ticket não será criado durante esta entrevista de design.

### ER-D18 — Features categóricas esparsas

O primeiro semi-Markov CRF usará features categóricas esparsas equivalentes a
one-hot, sem materializar vetores densos. O conjunto inicial cobrirá identidade
e forma dos tokens, prefixos e sufixos, caixa, números, pontuação, contexto
local e comprimento do segmento. Embeddings treináveis ou biomédicos
pré-treinados não integrarão este primeiro braço, podendo ser avaliados depois
como variantes separadas.

### ER-D19 — Matriz fatorial sob contrato comum

O estudo separará os efeitos do reconhecimento de entidades e da classificação
de relações. O nível de entidades comparará um dicionário canônico com o
semi-Markov CRF. O nível de relações comparará regras, k-NN e, quando existir,
o método híbrido sobre entidades gold. A avaliação end-to-end combinará cada
reconhecedor com cada método de relações que respeite o contrato canônico.

Entidades gold serão tratadas como teto experimental, e não como um pipeline
end-to-end. O pipeline legado incompatível poderá ser apresentado como
referência histórica, mas não será misturado às métricas fatoriais. O
dicionário comparável deverá produzir os mesmos quatro tipos canônicos e ser
construído ou adaptado sem consultar a partição avaliada.

### ER-D20 — PyTorch com núcleo segmental no projeto

A implementação futura usará PyTorch em CPU para diferenciação automática. A
recorrência segmental, o cálculo da função de partição e a decodificação serão
implementados no próprio projeto, de forma legível e testável. Bibliotecas
externas poderão apoiar operações tensoriais, mas não ocultarão o algoritmo
central cuja comparação faz parte do estudo.

### ER-D21 — Dicionário canônico restrito ao treino

O baseline de dicionário canônico será reconstruído em cada fold usando
somente as formas de superfície presentes nas menções gold da respectiva
partição de treino. Cada forma será associada ao tipo mais frequente no treino;
empates de tipo causarão abstenção. Na extração, correspondências mais longas
terão prioridade e a saída será plana e não sobreposta. O teste, a validação e
os vocabulários legados ou externos não contribuirão para esse dicionário.

### ER-D22 — Validação cruzada aninhada por documento

O desenvolvimento usará validação cruzada aninhada com separação por
documento. Os cinco folds externos fornecerão estimativas de desempenho; em
cada rodada, apenas subdivisões dos documentos de treino externo poderão
selecionar a intensidade L2 dentro da grade pré-registrada. Um novo corpus
externo e intocado continuará necessário para uma conclusão final, pois o
teste MCRC existente já foi consumido.

### ER-D23 — Distribuição natural no braço principal

O braço principal do semi-Markov CRF preservará a distribuição natural das
sequências. Ele não reamostrará sentenças nem aplicará pesos de classe à
likelihood. Regularização e métricas macro e por tipo tratarão o risco de
domínio da classe `Finding`. Uma variante ponderada, se investigada, será
registrada como outro braço experimental, sem substituir o principal.

### ER-D24 — Correspondência estrita como avaliação principal

Uma entidade prevista será correta somente quando seus offsets e tipo forem
idênticos aos da projeção gold plana. Uma relação end-to-end será correta
somente quando os dois endpoints, seus tipos e o rótulo da relação coincidirem
exatamente. Serão reportados micro-F1, macro-F1 e F1 por tipo ou relação.
Métricas por token e por sobreposição poderão aparecer apenas como diagnóstico,
sem sustentar a comparação principal.

### ER-D25 — Candidatos derivados apenas das previsões

No pipeline end-to-end, os pares candidatos serão gerados exclusivamente a
partir das menções previstas, da compatibilidade entre seus tipos e da janela
existente de até uma sentença. O nó lógico `Patient` será criado de modo
determinístico quando exigido pelo esquema. Menções, endpoints e pares gold não
participarão dessa execução; serão usados somente pelo avaliador depois que as
predições estiverem congeladas.

### ER-D26 — Tokenização de palavras e pontuação com offsets

O reconhecedor usará uma tokenização determinística de palavras, números e
pontuação que preserve os intervalos no texto original. Espaços periféricos
presentes em uma anotação poderão ser removidos somente na projeção gold
derivada, com registro da transformação. Os 31 alvos únicos que ainda não se
alinham a limites de token serão enviados à adjudicação humana. O tokenizador
por espaços e a modelagem por caracteres foram rejeitados para o braço
principal.

### ER-D27 — Segmentação de sentenças protegida contra abreviações

O separador de sentenças será determinístico, preservará offsets e não criará
limites após iniciais, abreviações clínicas ou unidades reconhecíveis, como
`H.`, `M.`, `S.`, `E.` e `mg.` nos contextos observados. As regras serão gerais,
versionadas e auditáveis; não dependerão de menções gold durante a inferência.

### ER-D28 — Semi-Markov CRF linear completo

O score global combinará features esparsas dos segmentos, transições entre
tipos e parâmetros para início e fim da sequência. Segmentos de entidade
poderão ter de um a 40 tokens. Cada token fora de entidade será representado
por um segmento externo unitário, evitando múltiplas segmentações latentes que
correspondam à mesma saída observável.

### ER-D29 — Otimização determinística com L-BFGS e L2

O modelo linear será inicializado em zero e otimizado em full-batch com L-BFGS.
A intensidade da regularização L2 e qualquer outro hiperparâmetro ajustável
serão escolhidos somente nas subdivisões internas da validação cruzada
aninhada. Critérios de convergência e falhas numéricas serão registrados nos
artefatos experimentais.

### ER-D30 — Contrato separado para entidades previstas

Cada reconhecedor produzirá um JSONL de `EntityPrediction`, separado do corpus
gold. Cada registro conterá `document_id`, identificador determinístico da
menção, intervalo de caracteres, tipo canônico, índice de sentença, método e
versão do modelo. O artefato não duplicará o texto clínico e não preencherá
campos de anotação nativa que não existem em uma previsão.

### ER-D31 — Vocabulário explícito de features restrito ao treino

O vocabulário de features será ajustado exclusivamente nos documentos de
treino de cada rodada. Features lexicais precisarão ocorrer ao menos duas vezes
para receber parâmetro próprio. Features desconhecidas serão ignoradas, sem
hashing; forma, caixa, números, pontuação, contexto e comprimento continuarão
fornecendo sinais não lexicais para tokens fora do vocabulário.

### ER-D32 — Grade pequena e pré-registrada de L2

Cada fold externo usará três folds internos por documento para comparar
`lambda` L2 em `{0.01, 0.1, 1, 10}`. A seleção maximizará macro-F1 estrita de
entidades. Empates favorecerão primeiro a menor variância entre folds internos
e depois a regularização mais forte. Templates, limite de segmento, otimizador
e tolerâncias não serão ajustados implicitamente nessa busca.

### ER-D33 — Novo teste externo anotado e congelado

A conclusão final dependerá de novos relatos clínicos abertos e ausentes do
MCRC. Os documentos passarão por auditoria de duplicidade, receberão anotação
dos quatro tipos e cinco relações sob revisão independente e serão congelados
antes da execução dos modelos. i2b2/n2c2 não substituirá esse teste por não
oferecer diretamente o mesmo esquema e por exigir acesso e mapeamentos
adicionais. Até esse conjunto existir, resultados serão declarados
exploratórios.

### ER-D34 — Manifesto completo por execução

Toda execução registrará commit, hashes de dados, identificadores documentais
das partições, configuração, versões das dependências, controles de
determinismo, estado de convergência, duração e hashes das saídas. Métricas sem
esse manifesto não integrarão a comparação oficial do estudo.

### ER-D35 — Relações ajustadas sobre entidades gold de treino

No braço fatorial principal, regras e modelos de relações serão configurados ou
treinados usando somente menções gold dos documentos de treino. Na inferência
end-to-end, o mesmo método de relações receberá os candidatos produzidos por
cada reconhecedor, sem correção gold. Isso mantém o classificador de relações
fixo entre os braços e permite atribuir diferenças à origem das entidades.
Treino sobre previsões out-of-fold poderá existir apenas como variante
experimental explicitamente separada.

### ER-D36 — Grafo com camadas de menção e conceito

As relações serão previstas primeiro entre ocorrências textuais individuais.
Uma projeção posterior poderá agregar menções de tipo e identidade normalizada
compatíveis em conceitos, preservando em cada nó conceitual os IDs, offsets e
proveniência das menções integrantes. A agregação nunca apagará a camada de
menção usada pela avaliação end-to-end.

### ER-D37 — JSONL versionado como contrato do grafo

O grafo canônico supervisionado será serializado em JSONL versionado, com
registros estruturados de nós, arestas e proveniência. NetworkX poderá consumir
esse contrato para visualização, e os CSVs legados serão uma exportação
adaptada, não a fonte de verdade do novo braço.

### ER-D38 — Conclusão operacional sem limiar de desempenho

Um braço será considerado completo quando executar do texto bruto ao grafo em
todas as partições previstas, validar os contratos e produzir previsões,
métricas, análises de erro e manifestos reproduzíveis. Não haverá F1 mínimo nem
remoção de um braço por desempenho inferior.

### ER-D39 — Decomposição pré-definida dos erros

A análise separará entidades ausentes, espúrias, com tipo incorreto e com
limites incorretos; estratificará resultados por OOV, tipo e comprimento; e
comparará relações com entidades gold e previstas. No end-to-end, distinguirá
erros herdados do reconhecimento, erros próprios da relação e perdas ou fusões
introduzidas pela projeção no grafo. Exemplos qualitativos complementarão, mas
não substituirão, essa decomposição reproduzível.

### ER-D40 — Identidade conceitual tipada e avaliação secundária do grafo

Um conceito será identificado pelo documento, tipo canônico e superfície
normalizada por Unicode, `casefold` e consolidação de espaços. O primeiro braço
não aplicará stemming, sinônimos ou ontologias externas. Nós e arestas
conceituais projetados das previsões serão comparados aos projetados do gold
por correspondência exata e F1 de nós e arestas. Essa avaliação será
secundária às métricas estritas de menções e relações.

### ER-D41 — Contrato não rotulado para candidatos end-to-end

Os candidatos derivados de entidades previstas serão serializados em JSONL
como `RelationCandidate`, contendo documento, ID determinístico, endpoints,
tipos, relação hipotética e distância entre sentenças. O contrato não terá
campo `label`: rótulo desconhecido durante a inferência não será representado
como `NO_RELATION`. O avaliador consultará o gold somente depois de congeladas
as previsões.

### ER-D42 — Marginal segmental opcional e explicitamente não calibrada

Uma `EntityPrediction` poderá registrar `marginal_probability`, calculada pelo
forward-backward do semi-Markov CRF. O valor será descrito como posterior do
modelo, não como confiança calibrada, e não filtrará a segmentação Viterbi
principal. O baseline de dicionário deixará esse campo nulo.

### ER-D43 — Não convergência finita permanece observável

O L-BFGS terá limite de 250 iterações. Uma execução que atingir esse limite com
valores finitos será avaliada e marcada com `converged=false`, sem ser apagada
do estudo. `NaN` ou infinito interromperá somente a execução afetada e gerará
diagnóstico estruturado. Não haverá troca automática de otimizador.

### ER-D44 — Suficiência pré-registrada do novo teste externo

O teste externo terá no mínimo 50 novos documentos e 100 entidades de cada
tipo. A amostragem sequencial continuará até obter ao menos 20 relações
positivas de cada classe ou alcançar o teto pré-registrado de 100 documentos.
Classes ainda abaixo desse suporte serão declaradas inconclusivas. A regra de
parada será aplicada antes de qualquer comparação de desempenho.

## Compatibilidade com decisões existentes

A ADR 0008 determina que menções complexas sejam preservadas sem descarte para
adaptar o corpus ao modelo. A nova saída operacional plana não deve substituir
o contrato lossless nem ser apresentada como o gold original. A ADR 0011
resolve essa fronteira ao definir a segmentação plana como projeção derivada e
auditável, mantendo a anotação original intacta.

## Encaminhamentos posteriores que não reabrem o design

- criar, na etapa de tickets, o trabalho de adjudicação humana da projeção gold;
  o ticket deverá começar com aviso explícito para uso de modelo com alto nível
  de reasoning;
- converter esta entrevista em uma especificação verificável;
- decompor a especificação em tickets dependentes antes da implementação;
- implementar e executar todos os braços, sem condicionar sua presença ao
  desempenho observado;
- selecionar e anotar o novo teste externo sob a regra ER-D44.

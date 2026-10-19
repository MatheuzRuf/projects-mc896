# Guia de adjudicação da projeção gold plana (`flat-gold-v1`)

Este guia foi fixado antes das revisões e é idêntico para todos os revisores.
Cada caso reúne candidatos de menção do MCRC que não podem entrar juntos numa
segmentação plana: sobreposições, mesmo trecho com tipos diferentes ou bordas
ajustadas. O revisor escolhe o conjunto de candidatos que melhor representa as
entidades do texto sob o esquema supervisionado.

## Tipos canônicos

- **Condition**: doença, distúrbio, síndrome ou diagnóstico nomeado como
  entidade patológica do paciente ou em discussão diagnóstica.
- **Finding**: sintoma, sinal, achado de exame físico, de imagem, laboratorial
  ou histológico, teste e seu resultado, estrutura anatômica observada.
- **Factor**: característica ou circunstância do paciente que não é um achado
  atual: idade, sexo, história familiar ou pessoal, hábitos, exposições,
  gestação, tratamentos prévios como contexto.
- **Modifier**: marcador de negação ou normalidade que modifica um achado,
  como ausência, normal ou negativo.

O esquema de relações liga `Patient → Condition/Finding/Factor`
(`HAS_*`), `Modifier → Finding` (`NEGATES`) e `Condition → Finding` (`CAUSES`).

## Regras, em ordem

1. **Seleção plana.** Os candidatos escolhidos não podem se sobrepor. É
   permitido escolher vários candidatos disjuntos do mesmo caso.
2. **Validade.** Só escolha um candidato se seu trecho for, no contexto, uma
   entidade completa e coerente com o tipo.
3. **Tipo no mesmo trecho.** Escolha o tipo que corresponde ao papel do trecho
   neste contexto, pelas definições acima. Participação em relações gold é
   evidência do papel dado pelos anotadores: um alvo de `HAS_CONDITION` ou
   origem de `CAUSES` foi tratado como `Condition`, um alvo de `NEGATES` ou de
   `CAUSES` como `Finding`. Use essa evidência, mas não a siga quando
   contradiz claramente o texto.
4. **Aninhamento.** Entre um candidato longo e candidatos internos:
   - prefira a decomposição em candidatos internos disjuntos quando ela
     preserva unidades que o esquema separa, como um `Modifier` que nega um
     `Finding`, ou quando os internos participam de relações gold e o longo
     não;
   - prefira o candidato longo quando ele é o conceito clínico completo e os
     internos são fragmentos sem significado próprio no contexto, ou quando só
     o longo participa de relações gold;
   - em empate real, prefira o conjunto que mantém mais relações gold
     representáveis, e depois o candidato mais curto que ainda nomeia o
     conceito completo.
5. **Bordas ajustadas.** Num caso de candidato único com remoção de espaços ou
   expansão até o token completo, escolha o candidato se a superfície ajustada
   for a entidade pretendida; caso contrário, não escolha nenhum.
6. **Seleção vazia** apenas quando nenhum candidato é entidade válida.

## Justificativa

- Uma ou duas frases específicas do caso, explicando o critério aplicado.
- **Não copie texto clínico.** Refira-se aos candidatos de forma abstrata ("o
  candidato mais longo", "o candidato Finding", "o interno que é alvo de
  NEGATES"). A coleta rejeita justificativas que contenham a superfície de um
  candidato.

## Independência

Cada revisor trabalha sem acesso a votos de outros revisores. O terceiro voto
é dado somente nos casos em que os dois primeiros divergem, também sem acesso
aos votos anteriores. Os revisores desta versão são modelos de linguagem
(`reviewer_kind = model`), conforme a decisão registrada em
[`README.md`](README.md).

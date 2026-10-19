# MultiCaRe external text annotation v1

This guide is fixed before annotation of the local 56 case narratives. The
source is the sorted `case_id` sequence in
`artifacts/external-multicare-preparation-v1/selection.json`. Each line of the
private `annotation-packet.jsonl` is one complete `ClinicalDocument` with no
mentions or gold relations. Text and character offsets are immutable. Reviewers
must not consult Project 1 extracted nodes/edges, predictions from the new
models, or the other reviewer's labels.

## What to annotate

Read every sentence of every selected narrative, including history, physical
examination, investigations, treatment and follow-up. Mark each clinical mention
in its original character span `[start, end)`. Include every supported occurrence,
even when the same concept appears repeatedly. Prefer the shortest complete
clinical expression; do not include articles, generic verbs, surrounding
punctuation or unrelated units. Do not merge disjoint occurrences. A mention
must occupy complete tokens in one sentence and be at most 40 tokens. Resolve
nested spans into one flat, non-overlapping set, recording the reason for any
ambiguous choice. Do not infer a mention merely from medical knowledge.

Use exactly the four supervised types from `flat-gold-v1`:

- `Condition`: a named disease, disorder, syndrome or diagnosis, including a
  suspected or historical diagnosis when explicitly named. Do not label a
  symptom as a disease merely because it is clinically important.
- `Finding`: symptom, sign, test, test result, imaging, physical, laboratory or
  histological observation. Include an observed numeric value with its unit
  when that is part of the finding. A normal measurement remains a Finding.
- `Factor`: age, sex, personal or family history, habits, exposure, pregnancy
  or prior treatment as patient context. Current medication administration
  alone is not a Factor under this schema.
- `Modifier`: the explicit text that negates or marks normality of a Finding,
  such as `no`, `without`, `normal` or `negative`. An unobserved implicit
  negation is not a Modifier.

Create one logical `Patient` for the person described by the case and one
`ClinicalCase` for the source narrative. Record their IDs and source article.
Do not create a Patient text span. Do not infer additional patients from family
members or from narrative quotations.

## Relations

Annotate directed, explicit positive relations only:

- `Patient → Condition` as `HAS_CONDITION` for a condition attributed to the
  patient. Do not convert a ruled-out diagnosis to a positive relation.
- `Patient → Finding` as `HAS_FINDING` for an observed finding attributed to the
  patient. If that occurrence is negated, record `NEGATES` and omit
  `HAS_FINDING` for that same occurrence.
- `Patient → Factor` as `HAS_FACTOR` for patient context expressed in the text.
- `Modifier → Finding` as `NEGATES` only when the modifier scopes over that
  finding. For a list, link the modifier to each finding it scopes over.
- `Condition → Finding` as `CAUSES` only when the narrative explicitly states a
  causal link. Temporal proximity and general medical plausibility are not
  enough.

Do not create `NO_RELATION` gold edges; candidate generation supplies negative
pairs later. Preserve every explicit positive relation even if a downstream
model's sentence window cannot represent it. Record any projection loss
separately. The `ClinicalDocument` contract validates endpoint types and
forbids a negated Finding from simultaneously receiving `HAS_FINDING`.

## Review and adjudication

Two independent review runs each label the full document from its original
text and this guide. Record a reviewer ID, `reviewer_kind=model`, model/version,
date, packet hash and guide hash with each review. Each reviewer writes a
separate JSONL of complete `ClinicalDocument` records and a separate provenance
manifest. Keep these records private with the source text. A reviewer must not
receive the other's labels or rationales. Model reviews are separate runs and
may still share systematic errors; report this limitation.

Only after both reviews are sealed, compare spans, types and directed relations.
Resolve every difference with a third independent review that sees the source
and guide, not the initial votes. Record the three decisions and an explanation
for the final choice. Validate that the adjudicated result reproduces the final
decision, all offsets slice the source text exactly, mentions are flat, and
relationships satisfy the contract. No heuristic agreement pass counts as a
review. Do not replace original votes after adjudication.

## Stopping and access boundary

Follow the fixed `case_id` order. After each adjudicated document, tally the
four entity types and five positive relation classes. Stop only after at least
50 documents, at least 100 mentions of **each** entity type and at least 20
positive relations of **each** class. Otherwise continue through the
preregistered source, up to 100 documents. The local source has only 56 cases;
additional documents would require a documented source and continuation rule
before they are inspected. At the cap, mark classes with fewer than 20
relations inconclusive. Fewer than 100 mentions of an entity type means the
required entity criterion was not met.

The text, reviewer labels and support counts are evaluation data only. Do not
use them to select model architecture, vocabulary, features, rules or
hyperparameters. Freeze final IDs, texts, labels, reviews, guide, audit,
selection and hashes before the first model evaluation on this set.

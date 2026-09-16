---
name: design-documents
description: Write or revise a design document under docs/design_docs/. Use when proposing a new capability, not when fixing a bug.
---

# Design documents

Long is fine. A design document is scanned and returned to, not read
straight through. What must be scannable is the specification.

- Normative statements come first in a section and stand alone. Rationale
  goes in a marked block below, which a reader can skip.
- Rejected alternatives and superseded drafts go in one `Decisions`
  section, cited from the places they affect. Never re-argued in place.
- A principle is stated once. Later sections cite it by name.
- Do not pre-empt objections. Drop "worth noting", "not an accident",
  "deliberately", "this is not a stylistic preference". State the decision
  and let it stand.
- Open questions go at the top or in their own section, never
  mid-paragraph.

## Calibration

The two design documents in `docs/design_docs/` run 2,574 and 12,390 words,
both at 28 words per sentence, with no hedging phrases in the first and one
in the second. Sentence length is the thing to fix; aim for twenty words.
Do not import the hedging habit that these documents do not have.

`docs/design_docs/template.rst` is the prescribed structure: a
`Requirement`, `Algorithm Design`, `Implementation` and `Testing` section
per topic, with the same topic name used in each. It also says requirements
"should not discuss technical software issues, but rather focus on model
capability", which is the rule most often broken.

## Enough

From `docs/design_docs/cached_outputs.rst`. The heading states the
requirement and the body is one or two normative sentences.

> Requirement: updating cached outputs
> ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
>
> There should be a documented process for creating cached outputs for steps
> and uploading them.
>
> Requirement: either "normal" or "cached" versions of a step
> ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
>
> We **do not** require the ability to set up a "normal" and a "cached"
> version of the same step within a ``compass`` test case or suite.

Five requirements in that document take about 250 words between them, and
saying which capability is *not* required is a requirement worth writing.

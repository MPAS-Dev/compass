---
name: plan-documents
description: Write a plan for work not yet started, for a colleague or the user to approve before implementation begins.
---

# Plan documents

The reader is deciding whether to let you proceed.

- Open questions and anything needing a decision go at the top.
- The steps, in order, one line each.
- Do not justify each step. Do not list the files you will touch. Do not
  restate the codebase back.
- If a step needs a paragraph to explain, it belongs in a design document.

## Enough

Plans are approved in conversation rather than committed, so there is no
human example in this repository to copy. The following is constructed.

> **Open:** should the `anvil` rows come out of the machines table now, or
> wait until the config file is restored?
>
> 1. Fix the land-ice draft used for pressure in the thin-film cases.
> 2. Restore the culled thin-film region in `initial_state`.
> 3. Take the smoothing from ice thickness, not draft, in both cases.
> 4. Update the `isomip_plus` docs in both guides.
> 5. Rerun the `pr` suite on Chrysalis against a `main` baseline.

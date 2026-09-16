---
name: pr-descriptions
description: Write or update a pull request description, including drafting pr_description.md before opening a PR. Use when opening a pull request or editing its body.
---

# Pull request descriptions

The reader is deciding whether to review.

- What changed and why, in a few sentences. Not how.
- Anything needing a reviewer decision goes in its own short list near the
  top, never mid-paragraph.
- A list of changed behaviours is fine. A trace of the mechanism is not.
- No commit list. No testing; that goes in a separate `Testing` comment.
- Link the issue or upstream pull request that gives context.
- Several fixes usually means several pull requests.

## Calibration

Measured over merged pull requests from 2023 and 2024, before any agent
wrote here, excluding bots. Descriptions run 32 to 36 words at the median,
57 to 60 at the seventy-fifth percentile, 94 to 106 at the ninetieth, and
230 at the longest. Roughly one description in ten is empty; a title that
says it all is an acceptable description.

## Enough

A bug fix, stated and done, from #609:

> The code for changing permissions was being given an incorrect directory
> and then was failing silently. This merge fixes the path and also removes
> the silent failure (an error will be raised if the status of the download
> directory cannot be determined).
>
> This merge also adds the database root and the root for each affected core
> to the list of directories to chmod/chown anytime any files get downloaded.
>
> closes #586
> closes #608

A port, in one sentence, from #519:

> This PR moves the generation of an initial state for the ocean global
> cosine bell test case from init mode to the initial state compass step.

A change with a caveat the reviewer needs, from #716:

> This PR replaces the `mode_init` MPAS-Ocean run in the `initial_state`
> step with local computations. The initial state is unchanged for the
> configuration that is currently used in the drying_slope cases. I did not
> retain all of the configuration options that were previously present in
> `mode_init`, e.g., the idealized transect.
>
> I added some buffer space in the y-dimension (the new config options `Ly`
> as differentiated from `Ly_analysis`) because the 1km case did not include
> the full 25km stretch of wetting and drying.

## Too much

Constructed, in the style of a real agent-written description elsewhere.
Five fixes each got a section, and each section traced its mechanism:

> `compass.ocean.tests.global_ocean` gained a module-scope import of
> `compass.ocean.mesh.remap_topography`, which runs
> `compass/ocean/mesh/__init__.py`, which imports the step classes, which
> import `compass.ocean.tests.global_ocean` back. The module could no longer
> be imported on its own; it worked only when something else imported
> `compass.ocean.mesh` first.

Someone deciding whether to review does not need the cycle traced. One
sentence would do; the rest belongs in the commit message. Five fixes is
also five pull requests.

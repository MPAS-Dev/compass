---
name: testing-comments
description: Write a Testing comment on a pull request, recording what was run and what the results were. Use after running suites, test cases or linting for a PR.
---

# Testing comments

What you ran, where, and whether it passed.

- Name the suite or test case, the machine and the compiler. One sentence.
- Give the work directory or the baseline you compared against.
- Say the result. Bit-for-bit, passed, or the numbers if they matter.
- Use a table or a list only when there are several runs to compare.
- Do not restate in prose what a table or a pasted result already shows.
- Failures unrelated to the branch go under their own heading at the end.

## Calibration

Measured over `Testing` comments from 2023 and 2024, before any agent wrote
here. They run 38 to 40 words at the median and 92 to 152 at the ninetieth
percentile. The shortest useful ones are under 25.

## Enough

From #663 and #816:

> ## Testing
>
> I ran the `pr` suite on Chrysalis using `main` as a baseline and all tests
> passed with BFB results.

> ## Testing
>
> I ran the `pr` suite with these changes, using the current `master` as a
> baseline. All tests passed.

With the work directory, from #753:

> ## Testing
>
> This mesh has been run through `files_for_e3sm` and the output is in:
> ```
> /lcrc/group/e3sm/ac.xylar/compass_1.2/chrysalis/e3smv3-meshes/sowisc12to30e3r2
> ```
>
> I have verified that the land-ice mask includes "land-locked" cells that
> were formerly a problem for sea-ice, based on the fix in #752

Saying what you could not check, from #791:

> ## Testing
>
> I successfully ran the `pr` and `nightly` suites on Chrysalis with Intel
> and Open-MPI. I didn't compare with a baseline because none is available
> for the Icos meshes.

When the results are worth pasting, paste them and stop, from #771:

> Tests are passing on Chrysalis (Intel/OpenMPI):
> ```
> Test Runtimes:
> 06:22 PASS ocean_hurricane_DEQU120at30cr10rr2_mesh_fblts
> 46:58 PASS ocean_hurricane_DEQU120at30cr10rr2_init_fblts
> 06:21 PASS ocean_hurricane_DEQU120at30cr10rr2_sandy_fblts
> Total runtime 59:43
> PASS: All passed successfully!
> ```
> and the documentation looks good.

## Too much

Constructed, in the style of a real agent-written comment elsewhere, which
pasted the results table and then said the same thing again in prose:

> Every test case now runs to completion. The five diffs are all of the form
> `File ... does not exist`: `main` crashed before writing those outputs, so
> there is nothing to compare against. Every comparison that had a file on
> both sides passed. A clean like-for-like comparison for those five cases
> needs a fresh baseline once this lands.

The table already showed the passes and the five missing files. The only
new sentence is the last one.

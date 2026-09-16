---
name: review-comments
description: Write a review comment, review findings, or a reply to review feedback on a GitHub pull request. Use when reviewing code, reporting what testing someone else's branch turned up, or answering a reviewer's question.
---

# Review comments

The reader is deciding what to change.

- Put each finding as an inline comment on the line it concerns, one point
  each. That is where colleagues put them, and it is why their review
  bodies are short.
- The review body summarizes: what you ran, and the verdict. Two or three
  sentences.
- Use a list in the body only for requests that span files.
- No section on what already works. One line for all of it, if any.
- Say what you could not check.

## Calibration

Measured over review comments from 2023 and 2024, before any agent wrote
here. Review bodies run 21 to 23 words at the median, 80 to 92 at the
ninetieth percentile, and 383 at the longest. Inline comments run 14 words
at the median, 58 at the ninetieth percentile, and 517 at the longest.

## Enough

Inline, one point and a suggestion, from #547 and #738:

> I'd be inclined to call this test group `BaroclinicGyre` rather than
> `MitgcmBaroclinicGyre`. I think shorter is better, and I doubt we'll be
> implementing multiple baroclinic gyre cases.

> Ok. Why don't you leave a comment that mentions that they refer to the old
> METIS partitioning so others know why they are there?

Inline, a question rather than an assertion, from #817:

> Is the tilde correct here? Won't this overwrite all the good data and
> leave the missing/bad data in place?

A body that is the verdict and nothing else, from #523:

> Thanks for doing this, @xylar. I'm approving based on my testing on Cori,
> in which I built spack with Albany and successfully ran the MALI
> `full_integration` test suite. I also had a quick look through the code
> diffs and nothing jumped out at me, although I can't pretend that I
> understand all of it.

In the body, when several requests span the whole change, from #637:

> @scalandr, this is excellent!
>
> I have some small changes and then there are a few other things to do:
> * the test group and all its classes and methods need to be added to the
>   `ocean/api.rst` in the developer's guide
> * It would be great if you could build the documentation locally (see
>   checklist above) to make sure it looks right.
> * Document the testing you did in a comment in the PR

## Too much

Constructed, in the style of a real agent-written review elsewhere, which
spent its first 444 words on "How this was reviewed", "What the previous
review asked for" and four paragraphs of "What works", then traced each
finding's mechanism:

> The bin-centre array is static: computed once in the step's `setup()` from
> `nBins`, `minLat` and `maxLat`, and never updated. But it is added to the
> output stream like any other field, so every output file in every run
> carries a copy of the same 61 numbers.

The finding is that a static array is written to every output file. Say
that inline, on the line that adds it, and stop.

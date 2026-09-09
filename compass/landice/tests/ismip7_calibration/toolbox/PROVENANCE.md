# Vendored `parameter_selection_toolbox.py`

`parameter_selection_toolbox.py` in this directory is an **unmodified, verbatim
copy** of the ISMIP7 parameter-selection toolbox. It implements the objective
function of the ISMIP7 AIS ice-ocean protocol (Reese et al., Sect. 4.2) and the
downstream `deltaT` fit.

## Where it came from

| | |
|---|---|
| Repository | <https://github.com/ismip/ismip7-antarctic-ocean-forcing> |
| Path | `parameterisations/parameter_selection_toolbox.py` |
| Last commit to touch the file | `132beb155e63fd07957e28407a83cb9d2c954447` ("Clean up toolbox code", 2026-06-23) |
| Repository `main` when copied | `84d17692935867d62b9c8e26f076a9869d74be31` (2026-09-08) |
| SHA256 | `3d016c04987d7c66c2603e2c3110d8a6a68cee0b6193c7f47bccf64aaa1ce029` |

**The upstream tags do not track the toolbox.** The newest toolbox tag,
`param-toolbox-v1`, points at a commit from 2026-04-20, which is *older* than
the last change to this file. Pinning to that tag would pin the wrong code, so
the commit is recorded instead. If upstream starts tagging toolbox updates
reliably, record the tag here as well.

## Why it is vendored rather than depended upon

There is no conda-forge or PyPI package that contains this module. It lives at
the top level of `parameterisations/`, outside the `i7aof` package that the
upstream repository does distribute, so even installing that package would not
provide it. The alternatives considered were a git submodule for a single file,
and reimplementing the objective function in Compass. Vendoring keeps exact
provenance without either cost.

## The integrity check

`__init__.py` verifies the SHA256 above on import, and
`compass/landice/tests/ismip7_calibration/tests/test_toolbox.py` asserts it in
CI. A local edit or a partial update therefore fails loudly, rather than
silently changing published calibration numbers.

## How to update

1. Copy the new upstream file over this one, unmodified.
2. Update the commit, `main` and SHA256 rows above (`sha256sum` the file).
3. Update `_EXPECTED_SHA256` in `__init__.py`.
4. Run the `replication` test case. It must still reproduce the published 8 km
   percentiles 4.75e-5 / 8.5e-5 / 1.375e-4 exactly. If it does not, the change
   is not a refactor and the calibration results need revisiting.

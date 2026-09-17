# Compass unit tests

Unit tests for compass's own Python code, run with `pytest` from the root of
the repository:

```bash
pytest
```

These are **unit** tests: they must run in seconds, on any machine, with no
input datasets, no MPAS build and no network access. Anything that needs
those belongs in a compass test case instead, where the framework can stage
inputs and compare against a baseline.

The tests are collected by CI on every pull request, so a test that depends on
data outside the repository will fail there even if it passes locally.

The layout mirrors the package: `tests/landice/ismip7_calibration/test_terms.py`
tests `compass/landice/tests/ismip7_calibration/terms.py`.

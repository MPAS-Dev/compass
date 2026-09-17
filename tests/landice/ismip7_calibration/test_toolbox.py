"""
Tests for the vendored ISMIP7 parameter-selection toolbox.

The vendored file must stay byte-for-byte identical to upstream.  The
published calibration numbers depend on it, so an accidental edit or a
partial update has to fail loudly rather than quietly change results.
"""

import pytest

from compass.landice.tests.ismip7_calibration import toolbox


def test_vendored_toolbox_matches_the_recorded_checksum():
    """The copy on disk must be the upstream revision PROVENANCE.md names."""
    assert toolbox.file_sha256() == toolbox.EXPECTED_SHA256


def test_check_integrity_passes_for_the_shipped_copy():
    """The import-time check must not raise for an unmodified checkout."""
    toolbox.check_integrity()


def test_check_integrity_reports_a_mismatch(monkeypatch):
    """A modified file must raise, naming both checksums."""
    monkeypatch.setattr(toolbox, 'EXPECTED_SHA256', '0' * 64)

    with pytest.raises(RuntimeError, match='does not match the recorded'):
        toolbox.check_integrity()


def test_provenance_is_recorded():
    """The upstream commit and URL must be pinned in the module."""
    assert len(toolbox.UPSTREAM_COMMIT) == 40
    assert toolbox.UPSTREAM_URL.startswith('https://github.com/ismip/')


def test_the_functions_the_calibration_calls_are_present():
    """
    Guard against an upstream refactor silently removing something the
    calibration drives.
    """
    module = toolbox.parameter_selection_toolbox
    for name in ('calculate_objective_function', 'optimise_deltaT',
                 'select_optimal_deltaT',
                 'select_subensemble_using_optimal_deltaT'):
        assert callable(getattr(module, name))

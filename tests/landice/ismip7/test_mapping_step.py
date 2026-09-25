"""Tests for shared ISMIP7 mapping-step behavior."""

from configparser import ConfigParser
from types import SimpleNamespace

from compass.landice.tests.ismip7_forcing import mapping_step
from compass.landice.tests.ismip7_forcing.fracture.build_mapping_file import (
    _resolve_grid_source,
)


class _Logger:
    def info(self, message):
        pass


def _step(tmp_path, mapping_files_path):
    config = ConfigParser()
    config['ismip7'] = {
        'mapping_files_path': str(mapping_files_path),
        'output_base_path': str(tmp_path / 'output'),
        'mali_mesh_file': 'mesh.nc',
    }
    return SimpleNamespace(config=config, logger=_Logger())


def test_mapping_step_reuses_cached_file(tmp_path, monkeypatch):
    """An existing cached mapping is linked rather than rebuilt."""
    cache = tmp_path / 'cache'
    cache.mkdir()
    expected = cache / 'map.nc'
    expected.touch()
    step = _step(tmp_path, cache)
    calls = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        mapping_step, 'build_mapping_file',
        lambda *args, **kwargs: calls.append((args, kwargs)))

    mapping_step.build_and_cache_mapping(
        step, 'source.nc', 'map.nc', 'bilinear')

    assert (tmp_path / 'map.nc').is_symlink()
    assert (tmp_path / 'map.nc').resolve() == expected
    assert len(calls) == 1


def test_mapping_step_caches_new_file(tmp_path, monkeypatch):
    """A newly built mapping is copied to the reusable cache directory."""
    step = _step(tmp_path, 'NotAvailable')

    def _build(config, logger, source_file, mapping_file, **kwargs):
        (tmp_path / mapping_file).write_text('weights')

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mapping_step, 'build_mapping_file', _build)

    mapping_step.build_and_cache_mapping(
        step, 'source.nc', 'map.nc', 'bilinear')

    cached = tmp_path / 'output' / 'mapping_files' / 'map.nc'
    assert cached.read_text() == 'weights'


def test_fracture_mapping_uses_xy_grid_donor(tmp_path):
    """The mapping source is not the ungridded excess-melt file."""
    version_path = (
        tmp_path / 'AIS' / 'CESM2-WACCM' / 'ssp585' /
        'fracture' / 'v2.1')
    version_path.mkdir(parents=True)
    (version_path / 'excess_melt_data.nc').touch()
    expected = version_path / 'lake_properties_data.nc'
    expected.touch()

    config = ConfigParser()
    config['ismip7'] = {
        'base_path_ismip7': str(tmp_path),
        'ice_sheet': 'ais',
        'model': 'CESM2-WACCM',
        'scenario': 'ssp585',
    }
    config['ismip7_fracture'] = {'version': 'latest'}

    source = _resolve_grid_source(config)

    assert source.files == (str(expected),)

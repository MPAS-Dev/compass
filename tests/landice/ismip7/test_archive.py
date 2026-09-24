"""Tests for ISMIP7 native-archive path resolution."""

from configparser import ConfigParser

import pytest

from compass.landice.ismip7.archive import (
    mapping_file_name,
    resolve_atmosphere_source,
    resolve_fracture_source,
    resolve_ocean_source,
    resolve_version_directory,
)


def _config(root, ice_sheet='gis', model='CESM2-WACCM',
            scenario='ssp585'):
    config = ConfigParser()
    config['ismip7'] = {
        'base_path_ismip7': str(root),
        'ice_sheet': ice_sheet,
        'model': model,
        'scenario': scenario,
        'mali_mesh_name': 'mesh',
    }
    config['ismip7_atmosphere'] = {
        'product': 'auto',
        'resolution': 'auto',
        'version': 'latest',
    }
    config['ismip7_ocean_thermal'] = {
        'resolution': 'auto',
        'version': 'latest',
    }
    config['ismip7_fracture'] = {'version': 'latest'}
    return config


def _forcing_file(root, relative_path):
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_latest_version_uses_numeric_components(tmp_path):
    """Dotted version components are sorted numerically, not as strings."""
    _forcing_file(tmp_path, 'v2.9/field.nc')
    expected = _forcing_file(tmp_path, 'v2.10/field.nc')

    _, version, files = resolve_version_directory(tmp_path)

    assert version == 'v2.10'
    assert files == (str(expected),)


def test_latest_does_not_fall_back_from_incomplete_version(tmp_path):
    """An incomplete newest transfer is reported instead of hidden."""
    _forcing_file(tmp_path, 'v1/field.nc')
    (tmp_path / 'v2').mkdir()

    with pytest.raises(FileNotFoundError, match="Version 'v2' contains no"):
        resolve_version_directory(tmp_path)


def test_atmosphere_latest_is_resolved_per_variable(tmp_path):
    """Variables can select different latest versions on the same grid."""
    base = 'GIS/CESM2-WACCM/ssp585'
    _forcing_file(
        tmp_path, f'{base}/SDBN1-4000m/acabf/v4/acabf_2000.nc')
    acabf = _forcing_file(
        tmp_path, f'{base}/SDBN1-1000m/acabf/v3/acabf_2000.nc')
    ts = _forcing_file(
        tmp_path, f'{base}/SDBN1-1000m/ts/v2/ts_2000.nc')
    config = _config(tmp_path)

    acabf_source = resolve_atmosphere_source(config, 'acabf')
    ts_source = resolve_atmosphere_source(config, 'ts')

    assert acabf_source.version == 'v3'
    assert acabf_source.files == (str(acabf),)
    assert ts_source.version == 'v2'
    assert ts_source.files == (str(ts),)
    assert acabf_source.resolution == '1000m'


def test_variable_version_override_takes_precedence(tmp_path):
    """A variable-specific version can pin one atmosphere dataset."""
    base = 'GIS/CESM2-WACCM/ssp585/SDBN1-1000m/acabf'
    expected = _forcing_file(tmp_path, f'{base}/v2/acabf_2000.nc')
    _forcing_file(tmp_path, f'{base}/v3/acabf_2000.nc')
    config = _config(tmp_path)
    config['ismip7_atmosphere']['acabf_version'] = 'v2'

    source = resolve_atmosphere_source(config, 'acabf')

    assert source.version == 'v2'
    assert source.files == (str(expected),)


def test_auto_product_uses_model_default(tmp_path):
    """Auto distinguishes the primary model product from dEBM2."""
    base = 'AIS/MRI-ESM2-0/ssp585'
    expected = _forcing_file(
        tmp_path, f'{base}/GEMB-SDBN1-2000m/ts/v2/ts_2000.nc')
    _forcing_file(tmp_path, f'{base}/dEBM2-8000m/ts/v1/ts_2000.nc')
    config = _config(
        tmp_path, ice_sheet='ais', model='MRI-ESM2-0')

    source = resolve_atmosphere_source(config, 'ts')

    assert source.product == 'GEMB-SDBN1'
    assert source.files == (str(expected),)


def test_gis_ocean_resolution_and_version(tmp_path):
    """GrIS ocean resolution and version are selected from the archive."""
    base = 'GIS/CESM2-WACCM/ssp585'
    expected = _forcing_file(
        tmp_path, f'{base}/ocean-1000m/tf/v2/tf_2000.nc')
    _forcing_file(
        tmp_path, f'{base}/ocean-4000m/tf/v3/tf_2000.nc')
    config = _config(tmp_path)

    source = resolve_ocean_source(config)

    assert source.resolution == '1000m'
    assert source.version == 'v2'
    assert source.files == (str(expected),)


def test_ais_ocx_ocean_choice_layout(tmp_path):
    """AIS OCX choices resolve below OCX/ocean rather than a variable dir."""
    expected = _forcing_file(
        tmp_path, 'AIS/OCX/ocean/main/v2/tf_AIS_OCX_main.nc')
    config = _config(
        tmp_path, ice_sheet='ais', model='None', scenario='OCX')

    source = resolve_ocean_source(config, choice='main')

    assert source.version == 'v2'
    assert source.forcing_group == 'OCX_main'
    assert source.files == (str(expected),)


def test_fracture_uses_native_model_scenario_path(tmp_path):
    """Fracture data resolves below the selected model and scenario."""
    expected = _forcing_file(
        tmp_path,
        'AIS/CESM2-WACCM/ssp585/fracture/v2.1/excess_melt_data.nc')
    config = _config(tmp_path, ice_sheet='ais')

    source = resolve_fracture_source(config, 'excess_melt_*.nc')

    assert source.version == 'v2.1'
    assert source.files == (str(expected),)


def test_mapping_name_identifies_source_grid(tmp_path):
    """Mapping files cannot collide across source resolutions."""
    config = _config(tmp_path)

    name = mapping_file_name(
        config, 'atm', 'atmosphere_SDBN1-1000m', 'conserve')

    assert name == (
        'map_ismip7_gis_atm_atmosphere_sdbn1_1000m_to_mesh_conserve.nc')

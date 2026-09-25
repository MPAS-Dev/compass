"""Resolve forcing files in the native ISMIP7 archive hierarchy."""

import glob
import os
import re
from dataclasses import dataclass

from compass.landice.ismip7.ice_sheet_params import get_params

_ICE_SHEET_DIRECTORIES = {
    'ais': 'AIS',
    'gis': 'GIS',
}

_DEFAULT_ATMOSPHERE_PRODUCTS = {
    'CESM2-WACCM': 'SDBN1',
    'MRI-ESM2-0': 'GEMB-SDBN1',
    'RACMO2.3p2-ERA': 'SDBN1',
}

_VERSION_PATTERN = re.compile(r'^v(\d+(?:\.\d+)*)$')
_GRID_PATTERN = re.compile(r'^(.+)-(\d+)m$')


@dataclass(frozen=True)
class ForcingSource:
    """A resolved collection of ISMIP7 forcing files."""

    directory: str
    files: tuple[str, ...]
    version: str
    model: str
    scenario: str
    forcing_group: str
    label: str
    source_grid: str
    product: str | None = None
    resolution: str | None = None


def resolve_atmosphere_source(config, variable):
    """
    Resolve one atmosphere variable from the native archive hierarchy.

    ``latest`` is evaluated independently for each variable because archive
    versions are not necessarily synchronized across variables.
    """
    params = get_params(config)
    scenario = config.get('ismip7', 'scenario')
    configured_model = config.get('ismip7', 'model')

    if params['atm_model'] is None:
        model = configured_model
        forcing_group = f'{model}_{scenario}'
        base_path = os.path.join(_ice_sheet_path(config), model, scenario)
    else:
        model = params['atm_model']
        forcing_group = scenario
        base_path = os.path.join(_ice_sheet_path(config), scenario, model)

    section_name = 'ismip7_atmosphere'
    section = config[section_name]
    product = section.get('product', fallback='auto')
    resolution = section.get('resolution', fallback='auto')
    grid_name, product, resolution = _resolve_atmosphere_grid(
        base_path, model, product, resolution)

    variable_path = os.path.join(base_path, grid_name, variable)
    version = _requested_version(config, section_name, variable)
    version_path, version, files = resolve_version_directory(
        variable_path, version, f'{variable}_*.nc')

    return ForcingSource(
        directory=version_path,
        files=files,
        version=version,
        model=model,
        scenario=scenario,
        forcing_group=forcing_group,
        label=f'{model}_{scenario}',
        source_grid=f'atmosphere_{grid_name}',
        product=product,
        resolution=resolution,
    )


def resolve_ocean_source(config, choice=None):
    """Resolve ocean thermal forcing from the native archive hierarchy."""
    params = get_params(config)
    ice_sheet = config.get('ismip7', 'ice_sheet')
    scenario = config.get('ismip7', 'scenario')
    configured_model = config.get('ismip7', 'model')
    section_name = 'ismip7_ocean_thermal'
    section = config[section_name]
    requested_version = section.get('version', fallback='latest')

    if params.get('ocean_choice_layout', False):
        if choice is None:
            raise ValueError('An ocean choice is required for AIS OCX.')
        model = configured_model
        grid_name = 'ocean'
        base_path = os.path.join(
            _ice_sheet_path(config), scenario, 'ocean', choice)
        forcing_group = f'{scenario}_{choice}'
        label = forcing_group
        resolution = None
    else:
        if params['ocean_model'] is None:
            model = configured_model
            base_path = os.path.join(
                _ice_sheet_path(config), model, scenario)
            forcing_group = f'{model}_{scenario}'
        else:
            model = params['ocean_model']
            base_path = os.path.join(
                _ice_sheet_path(config), scenario, model)
            forcing_group = scenario

        if ice_sheet == 'ais':
            grid_name = 'ocean'
            resolution = None
        else:
            requested_resolution = section.get(
                'resolution', fallback='auto')
            grid_name, resolution = _resolve_ocean_grid(
                base_path, requested_resolution)
        base_path = os.path.join(base_path, grid_name, 'tf')
        label = f'{model}_{scenario}'

    version_path, version, files = resolve_version_directory(
        base_path, requested_version, 'tf_*.nc')

    return ForcingSource(
        directory=version_path,
        files=files,
        version=version,
        model=model,
        scenario=scenario,
        forcing_group=forcing_group,
        label=label,
        source_grid=grid_name,
        resolution=resolution,
    )


def resolve_fracture_source(config, file_pattern):
    """Resolve one fracture product, keeping all pathways on one version."""
    model = config.get('ismip7', 'model')
    scenario = config.get('ismip7', 'scenario')
    base_path = os.path.join(
        _ice_sheet_path(config), model, scenario, 'fracture')
    requested_version = config.get(
        'ismip7_fracture', 'version', fallback='latest')
    version_path, version, files = resolve_version_directory(
        base_path, requested_version, file_pattern)

    return ForcingSource(
        directory=version_path,
        files=files,
        version=version,
        model=model,
        scenario=scenario,
        forcing_group=f'{model}_{scenario}',
        label=f'{model}_{scenario}',
        source_grid='fracture',
    )


def resolve_version_directory(base_path, requested_version='latest',
                              file_pattern='*.nc'):
    """
    Resolve an explicit version or the numerically latest version directory.

    Version names may contain any number of numeric components, so ``v2.10``
    correctly sorts after ``v2.9``.  If the latest directory exists but does
    not contain the expected files, an error is raised instead of silently
    falling back to an older version.
    """
    versions = []
    if os.path.isdir(base_path):
        for name in os.listdir(base_path):
            path = os.path.join(base_path, name)
            match = _VERSION_PATTERN.fullmatch(name)
            if match is not None and os.path.isdir(path):
                key = tuple(int(value) for value in match.group(1).split('.'))
                versions.append((key, name))

    if requested_version.lower() == 'latest':
        if not versions:
            raise FileNotFoundError(
                f'No version directories were found in:\n  {base_path}')
        _, version = max(versions)
    else:
        version = requested_version
        if _VERSION_PATTERN.fullmatch(version) is None:
            raise ValueError(
                f"Invalid version '{version}'. Use 'latest' or a version "
                f"such as 'v2' or 'v2.1'.")

    version_path = os.path.join(base_path, version)
    if not os.path.isdir(version_path):
        available = ', '.join(name for _, name in sorted(versions)) or 'none'
        raise FileNotFoundError(
            f"Version '{version}' was not found in:\n  {base_path}\n"
            f'Available versions: {available}')

    files = tuple(sorted(glob.glob(os.path.join(version_path, file_pattern))))
    if not files:
        raise FileNotFoundError(
            f"Version '{version}' contains no files matching:\n"
            f'  {os.path.join(version_path, file_pattern)}')

    return version_path, version, files


def mapping_file_name(config, component, source_grid, method_remap):
    """Build a mapping filename that identifies the selected source grid."""
    ice_sheet = config.get('ismip7', 'ice_sheet')
    mali_mesh_name = config.get('ismip7', 'mali_mesh_name')
    grid = re.sub(r'[^A-Za-z0-9]+', '_', source_grid).strip('_').lower()
    return (f'map_ismip7_{ice_sheet}_{component}_{grid}_to_'
            f'{mali_mesh_name}_{method_remap}.nc')


def _ice_sheet_path(config):
    ice_sheet = config.get('ismip7', 'ice_sheet')
    if ice_sheet not in _ICE_SHEET_DIRECTORIES:
        raise ValueError(
            f"Unknown ice_sheet '{ice_sheet}'. Must be one of: "
            f'{list(_ICE_SHEET_DIRECTORIES)}')
    archive_root = config.get('ismip7', 'base_path_ismip7')
    return os.path.join(archive_root, _ICE_SHEET_DIRECTORIES[ice_sheet])


def _requested_version(config, section, variable):
    option = f'{variable}_version'
    if config.has_option(section, option):
        return config.get(section, option)
    return config[section].get('version', fallback='latest')


def _resolve_atmosphere_grid(base_path, model, requested_product,
                             requested_resolution):
    candidates = []
    if os.path.isdir(base_path):
        for name in os.listdir(base_path):
            match = _GRID_PATTERN.fullmatch(name)
            path = os.path.join(base_path, name)
            if match is not None and os.path.isdir(path):
                product, metres = match.groups()
                candidates.append((product, int(metres), name))

    product = requested_product
    if product.lower() == 'auto':
        preferred = _DEFAULT_ATMOSPHERE_PRODUCTS.get(model)
        products = sorted({candidate[0] for candidate in candidates})
        if preferred in products:
            product = preferred
        elif len(products) == 1:
            product = products[0]
        else:
            choices = ', '.join(products) or 'none'
            raise ValueError(
                f"Could not select an atmosphere product automatically in:\n"
                f'  {base_path}\nAvailable products: {choices}. Set '
                f"'product' in [ismip7_atmosphere].")

    product_candidates = [candidate for candidate in candidates
                          if candidate[0] == product]
    resolution = _select_resolution(
        base_path, product_candidates, requested_resolution, 'atmosphere')
    grid_name = f'{product}-{resolution}'
    return grid_name, product, resolution


def _resolve_ocean_grid(base_path, requested_resolution):
    candidates = []
    if os.path.isdir(base_path):
        for name in os.listdir(base_path):
            match = re.fullmatch(r'ocean-(\d+)m', name)
            path = os.path.join(base_path, name)
            if match is not None and os.path.isdir(path):
                candidates.append(('ocean', int(match.group(1)), name))

    resolution = _select_resolution(
        base_path, candidates, requested_resolution, 'ocean')
    return f'ocean-{resolution}', resolution


def _select_resolution(base_path, candidates, requested_resolution,
                       component):
    if requested_resolution.lower() == 'auto':
        if not candidates:
            raise FileNotFoundError(
                f'No {component} resolution directories were found in:\n'
                f'  {base_path}')
        metres = min(candidate[1] for candidate in candidates)
    else:
        normalized = requested_resolution.lower().removesuffix('m')
        if not normalized.isdigit():
            raise ValueError(
                f"Invalid {component} resolution '{requested_resolution}'. "
                f"Use 'auto' or a value such as '1000m'.")
        metres = int(normalized)

    matches = [candidate for candidate in candidates
               if candidate[1] == metres]
    if not matches:
        available = ', '.join(
            f'{candidate[1]}m' for candidate in sorted(candidates)) or 'none'
        raise FileNotFoundError(
            f"Resolution '{metres}m' was not found in:\n  {base_path}\n"
            f'Available resolutions: {available}')
    return f'{metres}m'

import os

from compass.landice.ismip7.archive import (
    mapping_file_name,
    resolve_ocean_source,
    resolve_version_directory,
)
from compass.landice.ismip7.ice_sheet_params import get_params
from compass.landice.tests.ismip7_forcing.mapping_step import (
    build_and_cache_mapping,
    setup_mapping_step,
)
from compass.step import Step


class BuildMappingFile(Step):
    """Build weights from selected ocean grids to the MALI mesh."""

    def __init__(self, test_case):
        super().__init__(test_case=test_case, name='build_mapping_file')

    def setup(self):
        setup_mapping_step(self)

    def run(self):
        config = self.config
        section = config['ismip7']

        if section.getboolean('process_ocean_thermal'):
            self._build_scenario_mapping()
        if section.getboolean('process_ocean_climatology'):
            self._build_climatology_mapping()

    def _build_scenario_mapping(self):
        config = self.config
        params = get_params(config)
        section = config['ismip7_ocean_thermal']
        method_remap = section.get('method_remap')

        if params.get('ocean_choice_layout', False):
            choice = _first_ocean_choice(section.get('ocean_choice'))
            source = resolve_ocean_source(config, choice=choice)
        else:
            source = resolve_ocean_source(config)

        mapping_file = mapping_file_name(
            config, 'ocean', source.source_grid, method_remap)
        self.logger.info(f'Using ocean grid from {source.files[0]}')
        build_and_cache_mapping(
            self, source.files[0], mapping_file, method_remap)

    def _build_climatology_mapping(self):
        config = self.config
        section = config['ismip7_ocean_climatology']
        method_remap = section.get('method_remap')
        base_path = section.get('base_path_climatology')
        version = section.get('version', fallback='latest')
        _, _, files = resolve_version_directory(
            os.path.join(base_path, 'tf'), version, 'tf_*.nc')
        mapping_file = mapping_file_name(
            config, 'ocean', 'climatology', method_remap)

        self.logger.info(f'Using climatology grid from {files[0]}')
        build_and_cache_mapping(
            self, files[0], mapping_file, method_remap)


def _first_ocean_choice(raw_choices):
    """Return one representative AIS OCX choice for mapping generation."""
    choices = [choice.strip() for choice in raw_choices.split(',')
               if choice.strip()]
    if any(choice.lower() == 'all' for choice in choices):
        return 'main'
    if not choices:
        raise ValueError('No ocean_choice specified for AIS OCX forcing.')
    return choices[0]

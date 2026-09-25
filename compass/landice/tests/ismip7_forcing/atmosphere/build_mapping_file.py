from compass.landice.ismip7.archive import (
    mapping_file_name,
    resolve_atmosphere_source,
)
from compass.landice.tests.ismip7_forcing.mapping_step import (
    build_and_cache_mapping,
    setup_mapping_step,
)
from compass.step import Step


class BuildMappingFile(Step):
    """Build weights from the selected atmosphere grid to the MALI mesh."""

    def __init__(self, test_case):
        super().__init__(test_case=test_case, name='build_mapping_file')

    def setup(self):
        setup_mapping_step(self)

    def run(self):
        config = self.config
        method_remap = config.get('ismip7_atmosphere', 'method_remap')
        source = resolve_atmosphere_source(config, 'acabf')
        mapping_file = mapping_file_name(
            config, 'atm', source.source_grid, method_remap)

        self.logger.info(f'Using atmosphere grid from {source.files[0]}')
        build_and_cache_mapping(
            self, source.files[0], mapping_file, method_remap)

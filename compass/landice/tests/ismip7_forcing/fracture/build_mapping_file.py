from compass.landice.ismip7.archive import (
    mapping_file_name,
    resolve_fracture_source,
)
from compass.landice.tests.ismip7_forcing.mapping_step import (
    build_and_cache_mapping,
    setup_mapping_step,
)
from compass.step import Step


class BuildMappingFile(Step):
    """Build weights from the fracture grid to the MALI mesh."""

    def __init__(self, test_case):
        super().__init__(test_case=test_case, name='build_mapping_file')

    def setup(self):
        setup_mapping_step(self)

    def run(self):
        config = self.config
        section = config['ismip7_fracture']
        methods = {
            section.get('method_remap_shelf_collapse'),
            section.get('method_remap_excess_melt'),
            section.get('method_remap_lake_properties'),
        }
        if all(method is None or method.lower() == 'none' for method in methods):
            self.logger.info('No fracture mappings requested; skipping.')
            return
        source = _resolve_grid_source(config)

        for method_remap in sorted(methods):
            if method_remap is None or method_remap.lower() == 'none':
                continue
            mapping_file = mapping_file_name(
                config, 'fracture', source.source_grid, method_remap)
            self.logger.info(f'Using fracture grid from {source.files[0]}')
            build_and_cache_mapping(
                self, source.files[0], mapping_file, method_remap)


def _resolve_grid_source(config):
    """Select a fracture file with native x/y coordinates as grid donor."""
    errors = []
    for pattern in ('lake_properties_*.nc',
                    'ice_shelf_collapse_mask_*.nc'):
        try:
            return resolve_fracture_source(config, pattern)
        except FileNotFoundError as exc:
            errors.append(str(exc))
    raise FileNotFoundError(
        'No fracture source with x/y coordinates was found.\n' +
        '\n'.join(errors))

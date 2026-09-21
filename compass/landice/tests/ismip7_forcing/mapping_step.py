"""Shared support for ISMIP7 forcing mapping-file steps."""

import os
import shutil

from compass.landice.ismip7.mapping import build_mapping_file


def setup_mapping_step(step):
    """Add the MALI mesh input and request the ESMF task allocation."""
    section = step.config['ismip7']
    base_path_mali = section.get('base_path_mali')
    mali_mesh_file = section.get('mali_mesh_file')

    step.add_input_file(
        filename=mali_mesh_file,
        target=os.path.join(base_path_mali, mali_mesh_file))

    step.ntasks = section.getint('esmf_ntasks')
    step.min_tasks = 1


def build_and_cache_mapping(step, source_file, mapping_file, method_remap):
    """Reuse or build a mapping file, then cache a newly built file."""
    config = step.config
    logger = step.logger
    section = config['ismip7']
    mapping_files_path = section.get('mapping_files_path')
    output_base_path = section.get('output_base_path')
    mali_mesh_file = section.get('mali_mesh_file')

    reused = False
    if mapping_files_path != 'NotAvailable':
        cached_file = os.path.abspath(
            os.path.join(mapping_files_path, mapping_file))
        if os.path.isfile(cached_file):
            logger.info(f'Reusing mapping file {cached_file}')
            if os.path.lexists(mapping_file):
                os.remove(mapping_file)
            os.symlink(cached_file, mapping_file)
            reused = True

    build_mapping_file(
        config, logger, source_file, mapping_file,
        mali_mesh_file=mali_mesh_file,
        method_remap=method_remap,
        ntasks=step.ntasks)

    if reused:
        return

    mapping_files_dir = os.path.join(output_base_path, 'mapping_files')
    os.makedirs(mapping_files_dir, exist_ok=True)
    destination = os.path.join(mapping_files_dir, mapping_file)
    if os.path.realpath(mapping_file) != os.path.realpath(destination):
        logger.info(f'Caching mapping file in {mapping_files_dir}')
        shutil.copy2(mapping_file, destination)

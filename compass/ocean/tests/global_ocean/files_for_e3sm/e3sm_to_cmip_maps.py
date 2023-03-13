import os

from mpas_tools.logging import check_call

from compass.io import symlink
from compass.ocean.tests.global_ocean.files_for_e3sm.files_for_e3sm_step import (  # noqa: E501
    FilesForE3SMStep,
)


class E3smToCmipMaps(FilesForE3SMStep):
    """
    A step for creating mapping files from the MPAS-Ocean mesh to a standard
    CMIP6 mesh
    """
    def __init__(self, test_case):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.files_for_e3sm.FilesForE3SM
            The test case this step belongs to
        """  # noqa: E501

        super().__init__(test_case, name='e3sm_to_cmip_maps', ntasks=36,
                         min_tasks=1)

        self.add_input_file(filename='ocean.scrip.nc',
                            target='../scrip/ocean.scrip.nc')

        # add both scrip files, since we don't know in advance which to use
        self.add_input_file(
            filename='cmip6_180x360_scrip.20181001.nc',
            target='cmip6_180x360_scrip.20181001.nc',
            database='map_database')

        self.add_input_file(
            filename='cmip6_720x1440_scrip.20181001.nc',
            target='cmip6_720x1440_scrip.20181001.nc',
            database='map_database')

        self.add_output_file(filename='map_mpas_to_cmip6_aave.nc')
        self.add_output_file(filename='map_mpas_to_cmip6_mono.nc')
        self.add_output_file(filename='map_mpas_to_cmip6_nco.nc')

    def run(self):
        """
        Run this step of the testcase
        """
        super().run()
        # more verbose creation date for clarity
        creation_date = f'20{self.creation_date}'
        make_e3sm_to_cmip_maps(self.config, self.logger, self.mesh_short_name,
                               creation_date, self.ntasks)


def make_e3sm_to_cmip_maps(config, logger, mesh_short_name, creation_date,
                           ntasks):
    """
    Make mapping file from the MPAS-Ocean mesh to the CMIP6 grid

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for this test case

    logger : logging.Logger
        A logger for output from the step

    mesh_short_name : str
        The E3SM short name of the mesh

    creation_date : str
        The date to append to the mapping files

    ntasks : int
        The number of parallel tasks to use for remapping
    """

    link_dir = '../assembled_files/diagnostics/maps'

    try:
        os.makedirs(link_dir)
    except FileExistsError:
        pass

    src_scrip_filename = 'ocean.scrip.nc'
    cmip6_grid_res = config.get('files_for_e3sm', 'cmip6_grid_res')
    if cmip6_grid_res == '180x360':
        dst_scrip_filename = 'cmip6_180x360_scrip.20181001.nc'
    elif cmip6_grid_res == '720x1440':
        dst_scrip_filename = 'cmip6_720x1440_scrip.20181001.nc'
    else:
        raise ValueError(f'Unexpected cmip6_grid_res: {cmip6_grid_res}')

    parallel_executable = config.get('parallel', 'parallel_executable')
    # split the parallel executable into constituents in case it includes flags
    parallel_command = parallel_executable.split(' ')
    parallel_system = config.get('parallel', 'system')
    if parallel_system == 'slurm':
        parallel_command.extend(['-n', f'{ntasks}'])
    elif parallel_system == 'single_node':
        if ntasks > 1:
            parallel_command.extend(['-n', f'{ntasks}'])
    else:
        raise ValueError(f'Unexpected parallel system: {parallel_system}')
    parallel_command = ' '.join(parallel_command)

    map_methods = dict(aave='conserve', mono='fv2fv_flx', nco='nco')
    for suffix, map_method in map_methods.items():
        local_map_filename = f'map_mpas_to_cmip6_{suffix}.nc'
        args = ['ncremap', f'--mpi_pfx={parallel_command}',
                f'--alg_typ={map_method}',
                f'--grd_src={src_scrip_filename}',
                f'--grd_dst={dst_scrip_filename}',
                f'--map={local_map_filename}']
        check_call(args, logger=logger)

        map_filename = \
            f'map_{mesh_short_name}_to_cmip6_{cmip6_grid_res}_{suffix}.{creation_date}.nc'  # noqa: E501

        symlink(os.path.abspath(local_map_filename),
                f'{link_dir}/{map_filename}')

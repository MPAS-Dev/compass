import os
import xarray
import numpy as np
from glob import glob

from mpas_tools.logging import check_call

from compass.io import symlink
from compass.step import Step
from compass.ocean.tests.global_ocean.files_for_e3sm.graph_partition import \
    get_core_list


class SeaiceGraphPartition(Step):
    """
    A step for creating graph partition files for the sea-ice mesh
    """
    def __init__(self, test_case, restart_filename):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.files_for_e3sm.FilesForE3SM
            The test case this step belongs to

        restart_filename : str
            A restart file from the end of the dynamic adjustment test case to
            use as the basis for an E3SM initial condition
        """

        super().__init__(test_case, name='seaice_graph_partition', ntasks=1,
                         min_tasks=1, openmp_threads=1)

        self.add_input_file(filename='README', target='../README')
        self.add_input_file(filename='mesh.nc',
                            target=f'../{restart_filename}')

        for filename in ['icePresent_QU60km_polar.nc',
                         'seaice_QU60km_polar.nc']:
            self.add_input_file(filename=filename,
                                target=filename,
                                database='partition',
                                database_component='seaice')

        # for now, we won't define any outputs because they include the mesh
        # short name, which is not known at setup time.  Currently, this is
        # safe because no other steps depend on the outputs of this one.

    def run(self):
        """
        Run this step of the testcase
        """
        logger = self.logger

        with xarray.open_dataset('mesh.nc') as ds:
            mesh_short_name = ds.attrs['MPAS_Mesh_Short_Name']
            mesh_prefix = ds.attrs['MPAS_Mesh_Prefix']
            prefix = f'MPAS_Mesh_{mesh_prefix}'
            creation_date = ds.attrs[f'{prefix}_Version_Creation_Date']

        assembled_dir = f'../assembled_files/inputdata/ice/mpas-seaice/' \
                        f'{mesh_short_name}'
        try:
            os.makedirs(assembled_dir)
        except OSError:
            pass

        args = ['prepare_seaice_partitions',
                '-i', 'seaice_QU60km_polar.nc',
                '-p', 'icePresent_QU60km_polar.nc',
                '-m', 'mesh.nc',
                '-o', '.']
        check_call(args, logger)

        ncells = sum(1 for _ in open('graph.info'))
        cores = get_core_list(ncells=ncells)
        logger.info(f'Creating graph files between {np.amin(cores)} and '
                    f'{np.amax(cores)}')

        for ncores in cores:
            args = ['create_seaice_partitions',
                    '-m', 'mesh.nc',
                    '-o', '.',
                    '-p', f'mpas-seaice.graph.info.{creation_date}'
                    '-g', 'gpmetis',
                    '-n', f'{ncores}']
            check_call(args, logger)

        # create link in assembled files directory
        files = glob('mpas-seaice.graph.info.*')
        for file in files:
            symlink(f'../../../../../seaice_graph_partition/{file}',
                    f'{assembled_dir}/{file}')

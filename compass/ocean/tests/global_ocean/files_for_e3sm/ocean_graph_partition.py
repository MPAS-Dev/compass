import os
import xarray
import numpy as np
from glob import glob

from mpas_tools.logging import check_call

from compass.io import symlink
from compass.step import Step
from compass.ocean.tests.global_ocean.files_for_e3sm.graph_partition import \
    get_core_list


class OceanGraphPartition(Step):
    """
    A step for creating graph partition files for the ocean mesh
    """
    def __init__(self, test_case, mesh, restart_filename):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.files_for_e3sm.FilesForE3SM
            The test case this step belongs to

        mesh : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case that creates the mesh used by this test case

        restart_filename : str
            A restart file from the end of the dynamic adjustment test case to
            use as the basis for an E3SM initial condition
        """

        super().__init__(test_case, name='ocean_graph_partition', ntasks=1,
                         min_tasks=1, openmp_threads=1)

        self.add_input_file(filename='README', target='../README')
        self.add_input_file(filename='restart.nc',
                            target=f'../{restart_filename}')

        mesh_path = mesh.get_cull_mesh_path()
        self.add_input_file(
            filename='graph.info',
            work_dir_target=f'{mesh_path}/culled_graph.info')

        # for now, we won't define any outputs because they include the mesh
        # short name, which is not known at setup time.  Currently, this is
        # safe because no other steps depend on the outputs of this one.

    def run(self):
        """
        Run this step of the testcase
        """
        logger = self.logger

        with xarray.open_dataset('restart.nc') as ds:
            mesh_short_name = ds.attrs['MPAS_Mesh_Short_Name']
            mesh_prefix = ds.attrs['MPAS_Mesh_Prefix']
            prefix = f'MPAS_Mesh_{mesh_prefix}'
            creation_date = ds.attrs[f'{prefix}_Version_Creation_Date']

        try:
            os.makedirs(
                f'../assembled_files/inputdata/ocn/mpas-o/{mesh_short_name}')
        except OSError:
            pass

        symlink('graph.info', f'mpas-o.graph.info.{creation_date}')

        ncells = sum(1 for _ in open('graph.info'))
        cores = get_core_list(ncells=ncells)
        logger.info(f'Creating graph files between {np.amin(cores)} and '
                    f'{np.amax(cores)}')
        for ncores in cores:
            args = ['gpmetis', f'mpas-o.graph.info.{creation_date}',
                    f'{ncores}']
            check_call(args, logger)

        # create link in assembled files directory
        files = glob('mpas-o.graph.info.*')
        dest_path = \
            f'../assembled_files/inputdata/ocn/mpas-o/{mesh_short_name}'
        for file in files:
            symlink(f'../../../../../ocean_graph_partition/{file}',
                    f'{dest_path}/{file}')

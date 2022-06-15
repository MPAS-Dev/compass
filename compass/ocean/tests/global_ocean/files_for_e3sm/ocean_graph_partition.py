import os
import xarray
import numpy as np
from glob import glob

from mpas_tools.logging import check_call

from compass.io import symlink
from compass.step import Step


class OceanGraphPartition(Step):
    """
    A step for creating an E3SM ocean initial condition from the results of
    a dynamic-adjustment process to dissipate fast waves
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
                            target='../{}'.format(restart_filename))

        mesh_path = mesh.mesh_step.path
        self.add_input_file(
            filename='graph.info',
            work_dir_target='{}/culled_graph.info'.format(mesh_path))

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
            prefix = 'MPAS_Mesh_{}'.format(mesh_prefix)
            creation_date = ds.attrs['{}_Version_Creation_Date'.format(prefix)]

        try:
            os.makedirs('../assembled_files/inputdata/ocn/mpas-o/{}'.format(
                mesh_short_name))
        except OSError:
            pass

        symlink('graph.info', 'mpas-o.graph.info.{}'.format(creation_date))

        nCells = sum(1 for _ in open('graph.info'))
        min_graph_size = int(nCells / 6000)
        max_graph_size = int(nCells / 100)
        logger.info('Creating graph files between {} and {}'.format(
            min_graph_size, max_graph_size))
        n_power2 = 2**np.arange(1, 21)
        n_multiples12 = 12 * np.arange(1, 9)

        n = n_power2
        for power10 in range(3):
            n = np.concatenate([n, 10**power10 * n_multiples12])

        for index in range(len(n)):
            if min_graph_size <= n[index] <= max_graph_size:
                args = ['gpmetis', 'mpas-o.graph.info.{}'.format(creation_date),
                        '{}'.format(n[index])]
                check_call(args, logger)

        # create link in assembled files directory
        files = glob('mpas-o.graph.info.*')
        dest_path = '../assembled_files/inputdata/ocn/mpas-o/{}'.format(
            mesh_short_name)
        for file in files:
            symlink('../../../../../ocean_graph_partition/{}'.format(file),
                    '{}/{}'.format(dest_path, file))

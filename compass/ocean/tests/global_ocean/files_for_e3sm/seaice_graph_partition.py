import os
from glob import glob

import numpy as np
import xarray as xr
from mpas_tools.logging import check_call
from pyremap import MpasMeshDescriptor, Remapper

from compass.io import symlink
from compass.ocean.tests.global_ocean.files_for_e3sm.files_for_e3sm_step import (  # noqa: E501
    FilesForE3SMStep,
)
from compass.ocean.tests.global_ocean.files_for_e3sm.graph_partition import (
    get_core_list,
)


class SeaiceGraphPartition(FilesForE3SMStep):
    """
    A step for creating graph partition files for the sea-ice mesh
    """
    def __init__(self, test_case):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.files_for_e3sm.FilesForE3SM
            The test case this step belongs to
        """  # noqa: E501

        super().__init__(test_case, name='seaice_graph_partition', ntasks=36,
                         min_tasks=1)

        for filename in ['icePresent_QU60km_polar.nc',
                         'seaice_QU60km_polar.nc']:
            self.add_input_file(filename=filename,
                                target=filename,
                                database='partition',
                                database_component='seaice')

        # for now, we won't define any outputs because they include the mesh
        # short name, which is not known at setup time.  Currently, this is
        # safe because no other steps depend on the outputs of this one.

    def setup(self):
        """
        setup input files based on config options
        """
        super().setup()
        graph_filename = self.config.get('files_for_e3sm', 'graph_filename')
        if graph_filename != 'autodetect':
            graph_filename = os.path.normpath(os.path.join(
                self.test_case.work_dir, graph_filename))
            self.add_input_file(filename='graph.info', target=graph_filename)

    def run(self):
        """
        Run this step of the testcase
        """
        super().run()
        logger = self.logger
        creation_date = self.creation_date

        with xr.open_dataset('restart.nc') as ds:
            ncells = ds.sizes['nCells']

        cores = get_core_list(ncells=ncells)
        logger.info(f'Creating graph files between {np.amin(cores)} and '
                    f'{np.amax(cores)}')

        mapping_filename = _make_mapping_file(
            in_mesh_filename='seaice_QU60km_polar.nc',
            in_mesh_name='QU60km',
            out_mesh_filename='restart.nc',
            out_mesh_name=self.mesh_short_name,
            ntasks=self.ntasks,
            config=self.config,
            logger=logger,
            method='bilinear')

        args = ['prepare_seaice_partitions',
                '-i', 'seaice_QU60km_polar.nc',
                '-p', 'icePresent_QU60km_polar.nc',
                '-m', 'restart.nc',
                '-o', '.',
                '-w', mapping_filename]
        check_call(args, logger)

        args = ['create_seaice_partitions',
                '-m', 'restart.nc',
                '-o', '.',
                '-p', f'mpas-seaice.graph.info.{creation_date}',
                '-g', 'gpmetis',
                '--plotting',
                '-n']
        args = args + [f'{ncores}' for ncores in cores]
        check_call(args, logger)

        # create link in assembled files directory
        inputdata_dir = os.path.join(self.seaice_inputdata_dir, 'partitions')
        try:
            os.makedirs(inputdata_dir)
        except FileExistsError:
            pass
        files = glob('mpas-seaice.graph.info.*')
        for file in files:
            symlink(os.path.abspath(file),
                    f'{inputdata_dir}/{file}')


def _make_mapping_file(in_mesh_filename, in_mesh_name, out_mesh_filename,
                       out_mesh_name, ntasks, config, logger,
                       method):

    parallel_executable = config.get('parallel', 'parallel_executable')

    mapping_file_name = f'map_{in_mesh_name}_to_{out_mesh_name}_{method}.nc'

    in_descriptor = MpasMeshDescriptor(in_mesh_filename, in_mesh_name)
    out_descriptor = MpasMeshDescriptor(out_mesh_filename, out_mesh_name)

    remapper = Remapper(in_descriptor, out_descriptor, mapping_file_name)

    remapper.build_mapping_file(method=method, mpiTasks=ntasks,
                                tempdir='.', logger=logger,
                                esmf_parallel_exec=parallel_executable)

    return mapping_file_name

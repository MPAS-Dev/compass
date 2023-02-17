import os
from glob import glob

import numpy as np
import xarray as xr
from mpas_tools.logging import check_call

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

        super().__init__(test_case, name='seaice_graph_partition')

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

        args = ['prepare_seaice_partitions',
                '-i', 'seaice_QU60km_polar.nc',
                '-p', 'icePresent_QU60km_polar.nc',
                '-m', 'restart.nc',
                '-o', '.']
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
        files = glob('mpas-seaice.graph.info.*')
        for file in files:
            symlink(os.path.abspath(file),
                    f'{self.seaice_inputdata_dir}/{file}')

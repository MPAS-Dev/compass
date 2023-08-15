import os
from glob import glob

import numpy as np
from mpas_tools.logging import check_call

from compass.io import symlink
from compass.ocean.tests.global_ocean.files_for_e3sm.files_for_e3sm_step import (  # noqa: E501
    FilesForE3SMStep,
)
from compass.ocean.tests.global_ocean.files_for_e3sm.graph_partition import (
    get_core_list,
)


class OceanGraphPartition(FilesForE3SMStep):
    """
    A step for creating graph partition files for the ocean mesh
    """
    def __init__(self, test_case):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.files_for_e3sm.FilesForE3SM
            The test case this step belongs to
        """  # noqa: E501

        super().__init__(test_case, name='ocean_graph_partition')

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
        config = self.config
        creation_date = self.creation_date

        if not os.path.exists('graph.info'):
            graph_filename = config.get('files_for_e3sm', 'graph_filename')
            if graph_filename == 'autodetect':
                raise ValueError('No graph file was provided in the '
                                 'graph_filename config option.')
            graph_filename = os.path.normpath(os.path.join(
                self.test_case.work_dir, graph_filename))
            if not os.path.exists(graph_filename):
                raise FileNotFoundError('The graph file given in '
                                        'graph_filename could not be found.')
            if graph_filename != 'graph.info':
                symlink(graph_filename, 'graph.info')

        symlink('graph.info', f'mpas-o.graph.info.{creation_date}')

        ncells = sum(1 for _ in open('graph.info'))
        max_cells_per_core = config.getint('files_for_e3sm',
                                           'max_cells_per_core')
        min_cells_per_core = config.getint('files_for_e3sm',
                                           'min_cells_per_core')
        cores = get_core_list(ncells=ncells,
                              max_cells_per_core=max_cells_per_core,
                              min_cells_per_core=min_cells_per_core)

        logger.info(f'Creating graph files between {np.amin(cores)} and '
                    f'{np.amax(cores)}')
        for ncores in cores:
            if ncores > ncells:
                raise ValueError('Can\t have more tasks than cells in a '
                                 'partition file.')
            out_filename = f'mpas-o.graph.info.{creation_date}.part.{ncores}'
            if os.path.exists(out_filename):
                continue
            if ncores == 1:
                args = ['touch', f'mpas-o.graph.info.{creation_date}.part.1']
            else:
                args = ['gpmetis', f'mpas-o.graph.info.{creation_date}',
                        f'{ncores}']
            check_call(args, logger)

        # create link in assembled files directory

        inputdata_dir = os.path.join(self.ocean_inputdata_dir, 'partitions')
        try:
            os.makedirs(inputdata_dir)
        except FileExistsError:
            pass
        files = glob('mpas-o.graph.info.*')
        for file in files:
            symlink(os.path.abspath(file),
                    f'{inputdata_dir}/{file}')

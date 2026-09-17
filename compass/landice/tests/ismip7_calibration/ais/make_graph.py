"""
Build the graph partition the MALI melt diagnostics run on.
"""

import os

from compass.model import make_graph_file, partition
from compass.step import Step


class MakeGraph(Step):
    """
    A step that builds a graph partition file for the MALI mesh.

    The melt-diagnostic runs all use the same mesh and the same number of
    tasks, so the partition is built once here rather than once per run.
    Doing it in the test group means the group works on any MALI mesh,
    rather than only on meshes that already have a partition file
    distributed alongside them.
    """

    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.landice.tests.ismip7_calibration.ais.Ais
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='make_graph')

    def setup(self):
        """
        Set up this step of the test case
        """
        section = self.config['ismip7_calibration']
        ntasks = section.getint('ntasks')
        self.add_input_file(
            filename='mesh.nc',
            target=(f'{section.get("base_path_mali")}/'
                    f'{section.get("mali_mesh_file")}'))
        self.add_output_file(filename=f'graph.info.part.{ntasks}')

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        ntasks = config.getint('ismip7_calibration', 'ntasks')

        if not os.path.exists('graph.info'):
            self.logger.info('Building graph.info from the MALI mesh')
            make_graph_file(mesh_filename='mesh.nc',
                            graph_filename='graph.info')

        self.logger.info(f'Partitioning the graph for {ntasks} tasks')
        partition(ntasks, config, self.logger, graph_file='graph.info')

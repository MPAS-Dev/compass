import os

from compass.landice.tests.mismipplus.tasks import (
    approx_cell_count,
    exact_cell_count,
    get_ntasks_from_cell_count,
)
from compass.model import make_graph_file, run_model
from compass.step import Step


class RunModel(Step):
    """
    A step for performing forward MALI runs as part of MISMIP+ test cases.

    Attributes
    ----------
    suffixes : list of str, optional
        a list of suffixes for namelist and streams files produced
        for this step.  Most steps most runs will just have a
        ``namelist.landice`` and a ``streams.landice`` (the default) but
        the ``restart_run`` step of the ``restart_test`` runs the model
        twice, the second time with ``namelist.landice.rst`` and
        ``streams.landice.rst``

    resolution : float
        The nominal distance [m] between horizontal grid points (dcEdge).
    """
    def __init__(self, test_case, name, subdir=None, resolution=None,
                 ntasks=1, min_tasks=None, openmp_threads=1, suffixes=None):
        """
        Create a new test case

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        name : str
            the name of the test case

        subdir : str, optional
            the subdirectory for the step.  The default is ``name``

        resolution : float, optional
            The nominal distance [m] between horizontal grid points (dcEdge).
            For the ``SpinUp`` testcase no value should be passed. Instead the
            value will be parsed from the configuration options at the
            time of ``compass setup``.

        ntasks : int, optional
            the number of tasks the step would ideally use.  If fewer tasks
            are available on the system, the step will run on all available
            tasks as long as this is not below ``min_tasks``

        min_tasks : int, optional
            the number of tasks the step requires.  If the system has fewer
            than this number of tasks, the step will fail

        openmp_threads : int, optional
            the number of OpenMP threads the step will use

        suffixes : list of str, optional
            a list of suffixes for namelist and streams files produced
            for this step.  Most run steps will just have a
            ``namelist.landice`` and a ``streams.landice`` (the default) but
            the ``restart_run`` step of the ``restart_test`` runs the model
            twice, the second time with ``namelist.landice.rst`` and
            ``streams.landice.rst``
        """

        if suffixes is None:
            suffixes = ['landice']
        self.suffixes = suffixes

        if resolution is None:
            resolution = 0.0
        self.resolution = resolution

        if min_tasks is None:
            min_tasks = ntasks

        super().__init__(test_case=test_case,
                         name=name,
                         subdir=subdir,
                         ntasks=ntasks, min_tasks=min_tasks,
                         openmp_threads=openmp_threads)

        for suffix in suffixes:
            self.add_namelist_file(
                'compass.landice.tests.mismipplus', 'namelist.landice',
                out_name='namelist.{}'.format(suffix))

            self.add_streams_file(
                'compass.landice.tests.mismipplus', 'streams.landice',
                out_name='streams.{}'.format(suffix))

        self.add_input_file(filename='albany_input.yaml',
                            package='compass.landice.tests.mismipplus',
                            copy=True)

        self.add_model_as_input()

        self.add_output_file(filename='output.nc')
        self.add_output_file(filename='globalStats.nc')

    def setup(self):
        """
        Set the number of MPI tasks based on a tentative scaling of
        a the `ncells_at_1km_res` heuristic from the config file
        based on the desired resolution.
        """

        config = self.config
        # use the resolution at the time of ``compass setup``
        resolution = self.resolution

        # default gutter_length is 0. This encompasses `SmokeTest`, which does
        # not use the gutter_length option.
        gutter_length = 0.0

        if self.test_case.name == "spin_up":
            gutter_length = config.getfloat('mesh', 'gutter_length')

        # approximate the number of cells in the mesh, based on the requested
        # resolution and gutter_length.
        cell_count = approx_cell_count(resolution, gutter_length)

        # find optimal and minimum number of task for the desired resolution
        ntasks, min_tasks = get_ntasks_from_cell_count(config, cell_count)

        # set values as attributes
        self.ntasks, self.min_tasks = (ntasks, min_tasks)

        super().setup()

    def constrain_resources(self, available_resources):
        """
        Update resources at runtime from config options
        """

        config = self.config

        mesh_filename = os.path.join(self.work_dir, 'landice_grid.nc')

        # find the number of cells from the mesh created as part of the
        # `setup_mesh` step.
        cell_count = exact_cell_count(mesh_filename)

        # find optimal and minimum number of task for the cell count
        ntasks, min_tasks = get_ntasks_from_cell_count(config, cell_count)

        # set values as attributes
        self.ntasks, self.min_tasks = (ntasks, min_tasks)

        super().constrain_resources(available_resources)

    def run(self):
        """
        Run this step of the test case
        """

        config = self.config

        if self.test_case.name == "spin_up":
            # read the density value from config file and update
            # namelist. This will only work for the `SpinUp` test case.
            ice_density = config['mesh'].getfloat('ice_density')
            self.update_namelist_at_runtime(
                {'config_ice_density': f'{ice_density}'})

        make_graph_file(mesh_filename=self.mesh_file,
                        graph_filename='graph.info')

        for suffix in self.suffixes:
            run_model(step=self, namelist='namelist.{}'.format(suffix),
                      streams='streams.{}'.format(suffix))

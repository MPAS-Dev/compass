from compass.model import make_graph_file, run_model
from compass.step import Step


class RunModel(Step):
    """
    A step for performing forward MALI runs as part of low-res 20km Greenland
    test cases.

    Attributes
    ----------
    suffixes : list of str, optional
        a list of suffixes for namelist and streams files produced
        for this step.  Most steps most runs will just have a
        ``namelist.landice`` and a ``streams.landice`` (the default) but
        the ``restart_run`` step of the ``restart_test`` runs the model
        twice, the second time with ``namelist.landice.rst`` and
        ``streams.landice.rst``
    """
    def __init__(self, test_case, velo_solver, name='run_model', subdir=None,
                 ntasks=1, min_tasks=None, openmp_threads=1, suffixes=None):
        """
        Create a new test case

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        velo_solver : {'sia', 'FO'}
            The velocity solver to use for the test case

        name : str, optional
            the name of the test case

        subdir : str, optional
            the subdirectory for the step.  The default is ``name``

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
            for this step.  Most steps most runs will just have a
            ``namelist.landice`` and a ``streams.landice`` (the default) but
            the ``restart_run`` step of the ``restart_test`` runs the model
            twice, the second time with ``namelist.landice.rst`` and
            ``streams.landice.rst``
        """
        self.velo_solver = velo_solver
        assert velo_solver in {'sia', 'FO'}, \
            "Value of velo_solver must be one of {'sia', 'FO'}"

        if suffixes is None:
            suffixes = ['landice']
        self.suffixes = suffixes
        if min_tasks is None:
            min_tasks = ntasks
        super().__init__(test_case=test_case, name=name, subdir=subdir,
                         ntasks=ntasks, min_tasks=min_tasks,
                         openmp_threads=openmp_threads)

        # download and link the mesh
        self.add_input_file(filename='landice_grid.nc',
                            target='gis_20km_r01.20210824.nc', database='')

        for suffix in suffixes:
            self.add_namelist_file(
                'compass.landice.tests.greenland', 'namelist.landice',
                out_name='namelist.{}'.format(suffix))
            options = {'config_velocity_solver': "'{}'".format(velo_solver)}
            self.add_namelist_options(options=options,
                                      out_name='namelist.{}'.format(suffix))

            self.add_streams_file(
                'compass.landice.tests.greenland', 'streams.landice',
                out_name='streams.{}'.format(suffix))

        if velo_solver == 'FO':
            self.add_input_file(filename='albany_input.yaml',
                                package='compass.landice.tests.greenland',
                                copy=True)
            self.add_input_file(filename='albany_schoof_input.yaml',
                                package='compass.landice.tests.greenland',
                                copy=True)

        self.add_model_as_input()

        self.add_output_file(filename='output.nc')

    # no setup() is needed

    def run(self):
        """
        Run this step of the test case
        """
        make_graph_file(mesh_filename='landice_grid.nc',
                        graph_filename='graph.info')
        for suffix in self.suffixes:
            run_model(step=self, namelist='namelist.{}'.format(suffix),
                      streams='streams.{}'.format(suffix))

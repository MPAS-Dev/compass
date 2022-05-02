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
    """
    def __init__(self, test_case, calv_dt_frac=None, name='run_model', subdir=None,
                 cores=1, min_cores=None, threads=1, suffixes=None):
        """
        Create a new test case

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        calv_dt_frac : float, optional
            the value to use for calving dt fraction

        name : str, optional
            the name of the test case

        subdir : str, optional
            the subdirectory for the step.  The default is ``name``

        cores : int, optional
            the number of cores the step would ideally use.  If fewer cores
            are available on the system, the step will run on all available
            cores as long as this is not below ``min_cores``

        min_cores : int, optional
            the number of cores the step requires.  If the system has fewer
            than this number of cores, the step will fail

        threads : int, optional
            the number of threads the step will use

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
        if min_cores is None:
            min_cores = cores
        super().__init__(test_case=test_case, name=name, subdir=subdir,
                         cores=cores, min_cores=min_cores, threads=threads)

        # download and link the mesh
        self.mesh_file = 'MISMIP_2km_20220502.nc'
        self.add_input_file(filename=self.mesh_file, target=self.mesh_file,
                            database='')

        for suffix in suffixes:
            self.add_namelist_file(
                'compass.landice.tests.mismipplus', 'namelist.landice',
                out_name='namelist.{}'.format(suffix))

            self.add_streams_file(
                'compass.landice.tests.mismipplus', 'streams.landice',
                out_name='streams.{}'.format(suffix))

            if calv_dt_frac is not None:
                options = {'config_adaptive_timestep_calvingCFL_fraction':
                               f"{calv_dt_frac}"}
                self.add_namelist_options(options=options,
                                          out_name='namelist.{}'.format(suffix))

        self.add_input_file(filename='albany_input.yaml',
                            package='compass.landice.tests.mismipplus',
                            copy=True)

        self.add_model_as_input()

        self.add_output_file(filename='output.nc')
        self.add_output_file(filename='globalStats.nc')

    # no setup() is needed

    def run(self):
        """
        Run this step of the test case
        """
        make_graph_file(mesh_filename=self.mesh_file,
                        graph_filename='graph.info')
        for suffix in self.suffixes:
            run_model(step=self, namelist='namelist.{}'.format(suffix),
                      streams='streams.{}'.format(suffix))

from compass.model import run_model
from compass.step import Step


class Forward(Step):
    """
    A step for performing forward MPAS-Ocean runs as part of dam break
    test cases.
    """
    def __init__(self, test_case, resolution, name='forward', subdir=None,
                 cores=1, min_cores=None, threads=1):
        """
        Create a new test case

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolution : str
            The resolution of the test case

        name : str
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

        """
        if min_cores is None:
            min_cores = cores

        super().__init__(test_case=test_case, name=name, subdir=subdir,
                         cores=cores, min_cores=min_cores, threads=threads)

        self.add_namelist_file('compass.ocean.tests.dam_break',
                               'namelist.forward')
        if resolution < 1.:
            res_name = f'{int(resolution*1e3)}m'
        else:
            res_name = f'{int(resolution)}km'
        self.add_namelist_file('compass.ocean.tests.dam_break',
                               f'namelist.{res_name}.forward')
        self.add_streams_file('compass.ocean.tests.dam_break',
                              'streams.forward')

        input_path = '../initial_state'
        self.add_input_file(filename='mesh.nc',
                            target=f'{input_path}/culled_mesh.nc')

        self.add_input_file(filename='init.nc',
                            target=f'{input_path}/ocean.nc')

        self.add_input_file(filename='graph.info',
                            target=f'{input_path}/culled_graph.info')

        self.add_model_as_input()

        self.add_output_file(filename='output.nc')

    # no setup() is needed

    def run(self):
        """
        Run this step of the test case
        """

        resolution = self.resolution
        if resolution == 0.04:
            self.update_namelist_at_runtime({'config_dt': "'0000_00:00:00.01'"})
        elif resolution == 0.12:
            self.update_namelist_at_runtime({'config_dt': "'0000_00:00:00.03'"})
        run_model(self)


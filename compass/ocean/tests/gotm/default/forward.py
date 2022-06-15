from compass.model import run_model
from compass.step import Step


class Forward(Step):
    """
    A step for performing forward MPAS-Ocean runs as part of General Ocean
    Turbulence Model (GOTM) test cases.
    """
    def __init__(self, test_case):
        """
        Create a new test case

        Parameters
        ----------
        test_case : compass.ocean.tests.gotm.default.Default
            The test case this step belongs to

        """
        super().__init__(test_case=test_case, name='forward', ntasks=1,
                         min_tasks=1, openmp_threads=1)
        # make sure output is double precision
        self.add_streams_file('compass.ocean.streams', 'streams.output')

        self.add_namelist_file('compass.ocean.tests.gotm.default',
                               'namelist.forward')

        self.add_streams_file('compass.ocean.tests.gotm.default',
                              'streams.forward')

        self.add_input_file(filename='mesh.nc', target='../init/mesh.nc')
        self.add_input_file(filename='init.nc', target='../init/ocean.nc')
        self.add_input_file(filename='graph.info', target='../init/graph.info')

        self.add_input_file(filename='gotmturb.nml', target='gotmturb.nml',
                            package='compass.ocean.tests.gotm.default',
                            copy=True)

        self.add_model_as_input()

        self.add_output_file(filename='output.nc')

    # no setup() is needed

    def run(self):
        """
        Run this step of the test case
        """
        run_model(self)

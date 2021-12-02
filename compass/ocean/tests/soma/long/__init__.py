from compass.testcase import TestCase
from compass.ocean.tests.soma.initial_state import InitialState
from compass.ocean.tests.soma.forward import Forward
from compass.validate import compare_variables, compare_timers


class Long(TestCase):
    """
    A test case for long (3-year) simulations in the SOMA test group

    Attributes
    ----------
    resolution : str
        The resolution of the test case
    """

    def __init__(self, test_group, resolution):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.soma.Soma
            The test group that this test case belongs to

        resolution : str
            The resolution of the test case
        """
        name = 'long'
        self.resolution = resolution
        subdir = f'{resolution}/{name}'
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        step = InitialState(test_case=self, resolution=resolution)
        # add the namelist options specific ot this test case
        package = 'compass.ocean.tests.soma.long'
        for out_name in ['namelist_mark_land.ocean', 'namelist.ocean']:
            step.add_namelist_file(package, 'namelist.init', mode='init',
                                   out_name=out_name)
        self.add_step(step)

        step = Forward(test_case=self, resolution=resolution,
                       with_particles=False)
        step.add_namelist_file(package, 'namelist.forward', mode='forward')
        step.add_streams_file(package, 'streams.forward', mode='forward')

        self.add_step(step)

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        """
        variables = ['bottomDepth', 'layerThickness', 'maxLevelCell',
                     'temperature', 'salinity']
        compare_variables(
            test_case=self, variables=variables,
            filename1='initial_state/initial_state.nc')

        variables = ['temperature', 'layerThickness']
        compare_variables(
            test_case=self, variables=variables,
            filename1='forward/output/output.0001-01-01_00.00.00.nc')


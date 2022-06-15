from compass.validate import compare_variables
from compass.testcase import TestCase
from compass.landice.tests.hydro_radial.setup_mesh import SetupMesh
from compass.landice.tests.hydro_radial.run_model import RunModel
from compass.landice.tests.hydro_radial.visualize import Visualize


class RestartTest(TestCase):
    """
    A test case for performing two MALI runs of a radially symmetric
    hydrological setup, one full run and one run broken into two segments with
    a restart.  The test case verifies that the results of the two runs are
    identical.
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.hydro_radial.Dome
            The test group that this test case belongs to
        """
        super().__init__(test_group=test_group, name='restart_test')

        self.add_step(
            SetupMesh(test_case=self, initial_condition='zero'))

        name = 'full_run'
        step = RunModel(test_case=self, name=name, subdir=name, ntasks=4,
                        openmp_threads=1)
        # modify the namelist options and streams file
        step.add_namelist_file(
            'compass.landice.tests.hydro_radial.restart_test',
            'namelist.full', out_name='namelist.landice')
        step.add_streams_file(
            'compass.landice.tests.hydro_radial.restart_test',
            'streams.full', out_name='streams.landice')
        self.add_step(step)

        input_dir = name
        name = 'visualize_{}'.format(name)
        step = Visualize(test_case=self, name=name, subdir=name,
                         input_dir=input_dir)
        self.add_step(step, run_by_default=False)

        name = 'restart_run'
        step = RunModel(test_case=self, name=name, subdir=name, ntasks=4,
                        openmp_threads=1,
                        suffixes=['landice', 'landice.rst'])

        # modify the namelist options and streams file
        step.add_namelist_file(
            'compass.landice.tests.hydro_radial.restart_test',
            'namelist.restart', out_name='namelist.landice')
        step.add_streams_file(
            'compass.landice.tests.hydro_radial.restart_test',
            'streams.restart', out_name='streams.landice')

        step.add_namelist_file(
            'compass.landice.tests.hydro_radial.restart_test',
            'namelist.restart.rst', out_name='namelist.landice.rst')
        step.add_streams_file(
            'compass.landice.tests.hydro_radial.restart_test',
            'streams.restart.rst', out_name='streams.landice.rst')
        self.add_step(step)

        input_dir = name
        name = 'visualize_{}'.format(name)
        step = Visualize(test_case=self, name=name, subdir=name,
                         input_dir=input_dir)
        self.add_step(step, run_by_default=False)

    # no configure() method is needed

    # no run() method is needed

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        variables = ['waterThickness', 'waterPressure']
        compare_variables(test_case=self, variables=variables,
                          filename1='full_run/output.nc',
                          filename2='restart_run/output.nc')

from compass.validate import compare_variables
from compass.testcase import TestCase
from compass.landice.tests.eismint2.setup_mesh import SetupMesh
from compass.landice.tests.eismint2.run_experiment import RunExperiment


class RestartTest(TestCase):
    """
    A test case for performing two MALI runs of an EISMINT2 setup, one full run
    and one run broken into two segments with a restart.  The test case
    verifies that the results of the two runs are identical.
    """

    def __init__(self, test_group, thermal_solver):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.eismint2.Eismint2
            The test group that this test case belongs to

        thermal_solver : {'temperature', 'enthalpy'}
            The formulation of the thermodynamics to use
        """
        if thermal_solver == 'enthalpy':
            name = 'enthalpy_restart_test'
        elif thermal_solver == 'temperature':
            name = 'restart_test'
        else:
            raise ValueError(
                'Unknown thermal_solver {}'.format(thermal_solver))
        super().__init__(test_group=test_group, name=name)

        self.add_step(
            SetupMesh(test_case=self))

        experiment = 'f'

        name = 'full_run'
        step = RunExperiment(test_case=self, name=name, subdir=name, ntasks=4,
                             openmp_threads=1, experiment=experiment)

        options = {'config_thermal_solver': "'{}'".format(thermal_solver)}

        # modify the namelist options and streams file
        step.add_namelist_file(
            'compass.landice.tests.eismint2.restart_test',
            'namelist.full', out_name='namelist.landice')
        step.add_namelist_options(options, out_name='namelist.landice')
        step.add_streams_file(
            'compass.landice.tests.eismint2.restart_test',
            'streams.full', out_name='streams.landice')
        self.add_step(step)

        name = 'restart_run'
        step = RunExperiment(test_case=self, name=name, subdir=name, ntasks=4,
                             openmp_threads=1, experiment=experiment,
                             suffixes=['landice', 'landice.rst'])

        # modify the namelist options and streams file
        step.add_namelist_file(
            'compass.landice.tests.eismint2.restart_test',
            'namelist.restart', out_name='namelist.landice')
        step.add_namelist_options(options, out_name='namelist.landice')
        step.add_streams_file(
            'compass.landice.tests.eismint2.restart_test',
            'streams.restart', out_name='streams.landice')

        step.add_namelist_file(
            'compass.landice.tests.eismint2.restart_test',
            'namelist.restart.rst', out_name='namelist.landice.rst')
        step.add_namelist_options(options, out_name='namelist.landice.rst')
        step.add_streams_file(
            'compass.landice.tests.eismint2.restart_test',
            'streams.restart.rst', out_name='streams.landice.rst')
        self.add_step(step)

    # no configure() method is needed

    # no run() method is needed

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        variables = ['thickness', 'temperature', 'basalTemperature',
                     'heatDissipation']
        compare_variables(test_case=self, variables=variables,
                          filename1='full_run/output.nc',
                          filename2='restart_run/output.nc')

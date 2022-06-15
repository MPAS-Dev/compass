from compass.validate import compare_variables
from compass.testcase import TestCase
from compass.landice.tests.eismint2.setup_mesh import SetupMesh
from compass.landice.tests.eismint2.run_experiment import RunExperiment


class DecompositionTest(TestCase):
    """
    A test case for performing two MALI runs of a EISMINT2 setup, one with one
    core and one with four.  The test case verifies that the results of the two
    runs are identical.
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
            name = 'enthalpy_decomposition_test'
        elif thermal_solver == 'temperature':
            name = 'decomposition_test'
        else:
            raise ValueError(
                'Unknown thermal_solver {}'.format(thermal_solver))
        super().__init__(test_group=test_group, name=name)

        self.add_step(
            SetupMesh(test_case=self))

        options = {'config_run_duration': "'3000-00-00_00:00:00'",
                   'config_thermal_solver': "'{}'".format(thermal_solver)}

        experiment = 'f'
        for procs in [1, 4]:
            name = '{}proc_run'.format(procs)
            step = RunExperiment(test_case=self, name=name, subdir=name,
                                 ntasks=procs, openmp_threads=1,
                                 experiment=experiment)

            step.add_namelist_options(options)

            step.add_streams_file(
                'compass.landice.tests.eismint2.decomposition_test',
                'streams.landice')
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
                          filename1='1proc_run/output.nc',
                          filename2='4proc_run/output.nc')

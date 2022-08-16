from compass.testcase import TestCase
from compass.landice.tests.eismint2.setup_mesh import SetupMesh
from compass.landice.tests.eismint2.run_experiment import RunExperiment
from compass.landice.tests.eismint2.standard_experiments.visualize import \
    Visualize


class StandardExperiments(TestCase):
    """
    A test case for performing the standard EISMINT2 experiments.
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.eismint2.Eismint2
            The test group that this test case belongs to

        mesh_type : str
            The resolution or type of mesh of the test case
        """
        name = 'standard_experiments'
        super().__init__(test_group=test_group, name=name)

        self.add_step(
            SetupMesh(test_case=self))

        for experiment in ['a', 'b', 'c', 'd', 'f', 'g']:
            name = 'experiment_{}'.format(experiment)
            self.add_step(
                RunExperiment(test_case=self, name=name, subdir=name, ntasks=4,
                              openmp_threads=1, experiment=experiment))

        self.add_step(
            Visualize(test_case=self))

    def configure(self):
        """
        Modify the configuration options for this test case
        """
        # We want to visualize all test cases by default
        self.config.set('eismint2_viz', 'experiment', 'a, b, c, d, f, g')

    # no run() method is needed because we will just do the default: run all
    # the steps

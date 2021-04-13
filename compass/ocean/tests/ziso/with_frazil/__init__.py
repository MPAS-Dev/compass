from compass.testcase import TestCase
from compass.ocean.tests.ziso.initial_state import InitialState
from compass.ocean.tests.ziso.forward import Forward
from compass.ocean.tests import ziso
from compass.validate import compare_variables


class WithFrazil(TestCase):
    """
    The with frazil test case for the ZISO test group simply creates the mesh
    and initial condition, then performs a short forward run including frazil
    formation.

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
        test_group : compass.ocean.tests.ziso.Ziso
            The test group that this test case belongs to

        resolution : str
            The resolution of the test case
        """
        name = 'with_frazil'
        self.resolution = resolution
        subdir = '{}/{}'.format(resolution, name)
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        res_params = {'20km': {'cores': 4, 'min_cores': 2}}

        if resolution not in res_params:
            raise ValueError(
                'Unsupported resolution {}. Supported values are: '
                '{}'.format(resolution, list(res_params)))

        res_params = res_params[resolution]

        self.add_step(
            InitialState(test_case=self, resolution=resolution,
                         with_frazil=True))
        self.add_step(
            Forward(test_case=self, resolution=resolution,
                    cores=res_params['cores'],
                    min_cores=res_params['min_cores'],
                    with_analysis=False, with_frazil=True))

    def configure(self):
        """
        Modify the configuration options for this test case.
        """
        ziso.configure(self.name, self.resolution, self.config)

    def run(self):
        """
        Run each step of the test case
        """
        # run the steps
        super().run()

        # perform validation
        config = self.config
        work_dir = self.work_dir

        steps = self.steps_to_run
        if 'forward' in steps:
            variables = ['temperature', 'layerThickness']
            compare_variables(
                variables, config, work_dir,
                filename1='forward/output/output.0001-01-01_00.00.00.nc')

            variables = ['accumulatedFrazilIceMass',
                         'accumulatedFrazilIceSalinity',
                         'seaIceEnergy', 'frazilLayerThicknessTendency',
                         'frazilTemperatureTendency', 'frazilSalinityTendency',
                         'frazilSurfacePressure',
                         'accumulatedLandIceFrazilMass']
            compare_variables(variables, config, work_dir,
                              filename1='forward/frazil.nc')

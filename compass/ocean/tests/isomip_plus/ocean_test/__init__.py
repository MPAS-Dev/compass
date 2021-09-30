from compass.testcase import TestCase
from compass.ocean.tests.isomip_plus.initial_state import InitialState
from compass.ocean.tests.isomip_plus.ssh_adjustment import SshAdjustment
from compass.ocean.tests.isomip_plus.forward import Forward
from compass.ocean.tests.isomip_plus.streamfunction import Streamfunction
from compass.ocean.tests.isomip_plus.viz import Viz
from compass.ocean.tests.isomip_plus.misomip import Misomip
from compass.validate import compare_variables


class OceanTest(TestCase):
    """
    An ISOMIP+ test case

    Attributes
    ----------
    resolution : float
        The horizontal resolution (km) of the test case

    experiment : {'Ocean0', 'Ocean1', 'Ocean2'}
        The ISOMIP+ experiment

    vertical_coordinate : str
            The type of vertical coordinate (``z-star``, ``z-level``, etc.)

    time_varying_forcing : bool
        Whether the run includes time-varying land-ice forcing
    """

    def __init__(self, test_group, resolution, experiment,
                 vertical_coordinate, time_varying_forcing=False):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.isomip_plus.IsomipPlus
            The test group that this test case belongs to

        resolution : float
            The horizontal resolution (km) of the test case

        experiment : {'Ocean0', 'Ocean1', 'Ocean2'}
            The ISOMIP+ experiment

        vertical_coordinate : str
            The type of vertical coordinate (``z-star``, ``z-level``, etc.)

        time_varying_forcing : bool, optional
            Whether the run includes time-varying land-ice forcing
        """
        if time_varying_forcing:
            name = f'time_varying_{experiment}'
        else:
            name = experiment
        self.resolution = resolution
        self.experiment = experiment
        self.vertical_coordinate = vertical_coordinate
        self.time_varying_forcing = time_varying_forcing

        if resolution == int(resolution):
            res_folder = f'{int(resolution)}km'
        else:
            res_folder = f'{resolution}km'

        subdir = '{}/{}/{}'.format(res_folder, vertical_coordinate, name)
        super().__init__(test_group=test_group, name=name, subdir=subdir)

        self.add_step(
            InitialState(test_case=self, resolution=resolution,
                         experiment=experiment,
                         vertical_coordinate=vertical_coordinate,
                         time_varying_forcing=time_varying_forcing))
        self.add_step(
            SshAdjustment(test_case=self, resolution=resolution))
        self.add_step(
            Forward(test_case=self, name='performance', resolution=resolution,
                    experiment=experiment,
                    run_duration='0000-00-00_01:00:00',
                    time_varying_forcing=time_varying_forcing))

        self.add_step(
            Forward(test_case=self, name='simulation', resolution=resolution,
                    experiment=experiment,
                    run_duration='0000-01-00_00:00:00',
                    time_varying_forcing=time_varying_forcing),
            run_by_default=False)

        self.add_step(
            Streamfunction(test_case=self, resolution=resolution,
                           experiment=experiment),
            run_by_default=False)

        self.add_step(
            Viz(test_case=self, resolution=resolution, experiment=experiment),
            run_by_default=False)

        if resolution in [2., 5.]:
            self.add_step(
                Misomip(test_case=self, resolution=resolution,
                        experiment=experiment),
                run_by_default=False)

    def configure(self):
        """
        Modify the configuration options for this test case.
        """

        resolution = self.resolution
        vertical_coordinate = self.vertical_coordinate
        config = self.config
        experiment = self.experiment

        nx = round(800 / resolution)
        ny = round(100 / resolution)
        dc = 1e3 * resolution

        if experiment in ['Ocean0', 'Ocean2', 'Ocean3']:
            # warm initial conditions
            config.set('isomip_plus', 'init_top_temp', '-1.9')
            config.set('isomip_plus', 'init_bot_temp', '1.0')
            config.set('isomip_plus', 'init_top_sal', '33.8')
            config.set('isomip_plus', 'init_bot_sal', '34.7')
        else:
            # cold initial conditions
            config.set('isomip_plus', 'init_top_temp', '-1.9')
            config.set('isomip_plus', 'init_bot_temp', '-1.9')
            config.set('isomip_plus', 'init_top_sal', '33.8')
            config.set('isomip_plus', 'init_bot_sal', '34.55')

        if experiment in ['Ocean0', 'Ocean1', 'Ocean3']:
            # warm restoring
            config.set('isomip_plus', 'restore_top_temp', '-1.9')
            config.set('isomip_plus', 'restore_bot_temp', '1.0')
            config.set('isomip_plus', 'restore_top_sal', '33.8')
            config.set('isomip_plus', 'restore_bot_sal', '34.7')
        else:
            # cold restoring
            config.set('isomip_plus', 'restore_top_temp', '-1.9')
            config.set('isomip_plus', 'restore_bot_temp', '-1.9')
            config.set('isomip_plus', 'restore_top_sal', '33.8')
            config.set('isomip_plus', 'restore_bot_sal', '34.55')

        config.set('isomip_plus', 'nx', '{}'.format(nx))
        config.set('isomip_plus', 'ny', '{}'.format(ny))
        config.set('isomip_plus', 'dc', '{}'.format(dc))

        approx_cells = 30e3 / resolution ** 2
        # round to the nearest 4 cores
        cores = max(1, 4 * round(approx_cells / 200 / 4))
        min_cores = max(1, round(approx_cells / 2000))

        config.set('isomip_plus', 'forward_cores', '{}'.format(cores))
        config.set('isomip_plus', 'forward_min_cores', '{}'.format(min_cores))
        config.set('isomip_plus', 'forward_threads', '1')

        config.set('vertical_grid', 'coord_type', vertical_coordinate)

        for step_name in self.steps:
            if step_name in ['ssh_adjustment', 'performance', 'simulation']:
                step = self.steps[step_name]
                step.cores = cores
                step.min_cores = min_cores
                step.threads = 1

    def run(self):
        """
        Run each step of the test case
        """
        config = self.config
        # get the these properties from the config options
        for step_name in self.steps_to_run:
            if step_name in ['ssh_adjustment', 'performance', 'simulation']:
                step = self.steps[step_name]
                # get the these properties from the config options
                step.cores = config.getint('isomip_plus', 'forward_cores')
                step.min_cores = config.getint('isomip_plus',
                                               'forward_min_cores')
                step.threads = config.getint('isomip_plus', 'forward_threads')

        # run the steps
        super().run()

    def validate(self):
        """
        Perform validation of variables
        """
        # perform validation
        variables = ['temperature', 'salinity', 'layerThickness',
                     'normalVelocity']
        compare_variables(test_case=self, variables=variables,
                          filename1='performance/output.nc')

        variables = \
            ['ssh', 'landIcePressure', 'landIceDraft', 'landIceFraction',
             'landIceMask', 'landIceFrictionVelocity', 'topDrag',
             'topDragMagnitude', 'landIceFreshwaterFlux',
             'landIceHeatFlux', 'heatFluxToLandIce',
             'landIceBoundaryLayerTemperature', 'landIceBoundaryLayerSalinity',
             'landIceHeatTransferVelocity', 'landIceSaltTransferVelocity',
             'landIceInterfaceTemperature', 'landIceInterfaceSalinity',
             'accumulatedLandIceMass', 'accumulatedLandIceHeat']
        compare_variables(test_case=self, variables=variables,
                          filename1='performance/land_ice_fluxes.nc')

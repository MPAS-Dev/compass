from compass.testcase import TestCase
from compass.ocean.tests.isomip_plus.process_geom import ProcessGeom
from compass.ocean.tests.isomip_plus.planar_mesh import PlanarMesh
from compass.ocean.tests.isomip_plus.cull_mesh import CullMesh
from compass.ocean.tests.isomip_plus.initial_state import InitialState
from compass.ocean.tests.isomip_plus.ssh_adjustment import SshAdjustment
from compass.ocean.tests.isomip_plus.forward import Forward
from compass.ocean.tests.isomip_plus.streamfunction import Streamfunction
from compass.ocean.tests.isomip_plus.viz import Viz
from compass.ocean.tests.isomip_plus.misomip import Misomip
from compass.validate import compare_variables


class IsomipPlusTest(TestCase):
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

    thin_film_present: bool
        Whether a thin film is present under land ice
    """

    def __init__(self, test_group, resolution, experiment,
                 vertical_coordinate, time_varying_forcing=False,
                 time_varying_load=None, thin_film_present=False,
                 tidal_forcing=False):
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

        time_varying_load : {'increasing', 'decreasing', None}, optional
            Only relevant if ``time_varying_forcing = True``.  If
            ``'increasing'``, a doubling of the ice-shelf pressure will be
            applied over one year.  If ``'decreasing'``, the ice-shelf
            thickness will be reduced to zero over one year.  Otherwise,
            the default behavior is that the ice shelf grows from 10% of its
            full thickness to its full thickness over 1 year.

        thin_film_present: bool, optional
            Whether the run includes a thin film below grounded ice

        tidal_forcing: bool, optional
            Whether the run includes a single-period tidal forcing
        """
        name = experiment
        if tidal_forcing:
            name = f'tidal_forcing_{name}'
        if time_varying_forcing:
            if time_varying_load == 'increasing':
                name = f'drying_{name}'
            elif time_varying_load == 'decreasing':
                name = f'wetting_{name}'
            else:
                name = f'time_varying_{name}'
        if thin_film_present:
            name = f'thin_film_{name}'

        self.resolution = resolution
        self.experiment = experiment
        self.vertical_coordinate = vertical_coordinate
        self.time_varying_forcing = time_varying_forcing
        self.time_varying_load = time_varying_load
        self.thin_film_present = thin_film_present

        if resolution == int(resolution):
            res_folder = f'{int(resolution)}km'
        else:
            res_folder = f'{resolution}km'

        subdir = f'{res_folder}/{vertical_coordinate}/{name}'
        super().__init__(test_group=test_group, name=name, subdir=subdir)

        self.add_step(
            ProcessGeom(test_case=self, resolution=resolution,
                        experiment=experiment,
                        thin_film_present=thin_film_present))

        self.add_step(
            PlanarMesh(test_case=self, thin_film_present=thin_film_present))

        self.add_step(
            CullMesh(test_case=self, thin_film_present=thin_film_present,
                     planar=True))

        self.add_step(
            InitialState(test_case=self, resolution=resolution,
                         experiment=experiment,
                         vertical_coordinate=vertical_coordinate,
                         time_varying_forcing=time_varying_forcing,
                         thin_film_present=thin_film_present))
        self.add_step(
            SshAdjustment(test_case=self, resolution=resolution,
                          vertical_coordinate=vertical_coordinate,
                          thin_film_present=thin_film_present))
        if tidal_forcing or time_varying_load in ['increasing', 'decreasing']:
            performance_run_duration = '0000-00-01_00:00:00'
        else:
            performance_run_duration = '0000-00-00_01:00:00'
        self.add_step(
            Forward(test_case=self, name='performance', resolution=resolution,
                    experiment=experiment,
                    run_duration=performance_run_duration,
                    vertical_coordinate=vertical_coordinate,
                    tidal_forcing=tidal_forcing,
                    time_varying_forcing=time_varying_forcing,
                    thin_film_present=thin_film_present))

        self.add_step(
            Forward(test_case=self, name='simulation', resolution=resolution,
                    experiment=experiment,
                    run_duration='0000-01-00_00:00:00',
                    vertical_coordinate=vertical_coordinate,
                    tidal_forcing=tidal_forcing,
                    time_varying_forcing=time_varying_forcing,
                    thin_film_present=thin_film_present),
            run_by_default=False)

        self.add_step(
            Streamfunction(test_case=self, resolution=resolution,
                           experiment=experiment),
            run_by_default=False)

        self.add_step(
            Viz(test_case=self, resolution=resolution, experiment=experiment,
                tidal_forcing=tidal_forcing),
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
        thin_film_present = self.thin_film_present
        time_varying_load = self.time_varying_load
        config = self.config
        experiment = self.experiment

        nx = round(800 / resolution)
        ny = round(100 / resolution)
        dc = 1e3 * resolution
        # Width of the thin film region
        nx_thin_film = 10

        if thin_film_present:
            config.set('isomip_plus', 'min_column_thickness', '1e-3')

        if time_varying_load == 'increasing':
            config.set('isomip_plus_forcing', 'scales', '1.0, 2.0, 2.0')
        if time_varying_load == 'decreasing':
            config.set('isomip_plus_forcing', 'scales', '1.0, 0.0, 0.0')

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

        config.set('isomip_plus', 'nx', f'{nx}')
        config.set('isomip_plus', 'ny', f'{ny}')
        config.set('isomip_plus', 'dc', f'{dc}')
        config.set('isomip_plus', 'nx_thin_film', f'{nx_thin_film}')

        approx_cells = 30e3 / resolution ** 2
        # round to the nearest 4 cores
        ntasks = max(1, 4 * round(approx_cells / 200 / 4))
        min_tasks = max(1, round(approx_cells / 2000))

        config.set('isomip_plus', 'forward_ntasks', f'{ntasks}')
        config.set('isomip_plus', 'forward_min_tasks', f'{min_tasks}')
        config.set('isomip_plus', 'forward_threads', '1')

        config.set('vertical_grid', 'coord_type', vertical_coordinate)

        if vertical_coordinate == 'sigma':
            if time_varying_load in ['increasing', 'decreasing']:
                config.set('vertical_grid', 'vert_levels', '3')
            else:
                # default to 10 vertical levels instead of 36
                config.set('vertical_grid', 'vert_levels', '10')

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

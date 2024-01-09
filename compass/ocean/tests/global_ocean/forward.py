import os
import time
from importlib.resources import contents

from compass.model import run_model
from compass.ocean.tests.global_ocean.metadata import (
    add_mesh_and_init_metadata,
)
from compass.ocean.tests.global_ocean.tasks import get_ntasks_from_cell_count
from compass.step import Step
from compass.testcase import TestCase


class ForwardStep(Step):
    """
    A step for performing forward MPAS-Ocean runs as part of global ocean test
    cases.

    Attributes
    ----------
    mesh : compass.ocean.tests.global_ocean.mesh.Mesh
        The test case that produces the mesh for this run

    init : compass.ocean.tests.global_ocean.init.Init
        The test case that produces the initial condition for this run

    time_integrator : {'split_explicit_ab2', 'RK4'}
        The time integrator to use for the forward run

    dynamic_ntasks : bool
        Whether the target and minimum number of MPI tasks (``ntasks`` and
        ``min_tasks``) are computed dynamically from the number of cells
        in the mesh

    get_dt_from_min_res : bool
        Whether to automatically compute `config_dt` and `config_btr_dt`
        namelist options from the minimum resolution of the mesh
    """
    def __init__(self, test_case, mesh, time_integrator, init=None,
                 name='forward', subdir=None, ntasks=None, min_tasks=None,
                 openmp_threads=None, get_dt_from_min_res=True,
                 land_ice_flux_mode='pressure_only', **kwargs):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        mesh : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case that produces the mesh for this run

        time_integrator : {'split_explicit_ab2', 'RK4'}
            The time integrator to use for the forward run

        init : compass.ocean.tests.global_ocean.init.Init, optional
            The test case that produces the initial condition for this run

        name : str, optional
            the name of the step

        subdir : str, optional
            the subdirectory for the step.  The default is ``name``

        ntasks : int, optional
            the number of tasks the step would ideally use.  If fewer tasks
            are available on the system, the step will run on all available
            tasks as long as this is not below ``min_tasks``

        min_tasks : int, optional
            the number of tasks the step requires.  If the system has fewer
            than this number of tasks, the step will fail

        openmp_threads : int, optional
            the number of OpenMP threads the step will use

        get_dt_from_min_res : bool
            Whether to automatically compute `config_dt` and `config_btr_dt`
            namelist options from the minimum resolution of the mesh

        land_ice_flux_mode : {'pressure_only', 'standalone', 'data'}, optional
            Whether to have no ice-shelf melt fluxes ("pressure_only"),
            prognostic melt ("standalone") or data melt from a
            satellite-derived climatology ("data").
        """
        self.mesh = mesh
        self.init = init
        self.time_integrator = time_integrator
        if min_tasks is None:
            min_tasks = ntasks
        super().__init__(test_case=test_case, name=name, subdir=subdir,
                         ntasks=ntasks, min_tasks=min_tasks,
                         openmp_threads=openmp_threads, **kwargs)

        if (ntasks is None) != (openmp_threads is None):
            raise ValueError('You must specify both ntasks and openmp_threads '
                             'or neither.')

        self.dynamic_ntasks = (ntasks is None and min_tasks is None)
        self.get_dt_from_min_res = get_dt_from_min_res

        # make sure output is double precision
        self.add_streams_file('compass.ocean.streams', 'streams.output')

        self.add_namelist_file(
            'compass.ocean.tests.global_ocean', 'namelist.forward')
        self.add_streams_file(
            'compass.ocean.tests.global_ocean', 'streams.forward')

        if mesh.with_ice_shelf_cavities:
            options = dict(config_check_ssh_consistency='.false.',
                           config_land_ice_flux_mode=f"'{land_ice_flux_mode}'")
            self.add_namelist_options(options)
            if land_ice_flux_mode == 'data':
                filename = 'prescribed_ismf_adusumilli2020.nc'
                target = f'{init.path}/remap_ice_shelf_melt/{filename}'
                self.add_input_file(filename=filename, work_dir_target=target)
                self.add_streams_file(
                    'compass.ocean.tests.global_ocean', 'streams.dismf')

        mesh_package = mesh.package
        mesh_package_contents = list(contents(mesh_package))
        mesh_namelists = ['namelist.forward',
                          f'namelist.{time_integrator.lower()}']
        for mesh_namelist in mesh_namelists:
            if mesh_namelist in mesh_package_contents:
                self.add_namelist_file(mesh_package, mesh_namelist)

        mesh_streams = ['streams.forward',
                        f'streams.{time_integrator.lower()}']
        for mesh_stream in mesh_streams:
            if mesh_stream in mesh_package_contents:
                self.add_streams_file(mesh_package, mesh_stream)

        mesh_path = self.mesh.get_cull_mesh_path()
        self.add_input_file(
            filename='mesh.nc',
            work_dir_target=f'{mesh_path}/culled_mesh.nc')

        if init is not None:
            if mesh.with_ice_shelf_cavities:
                initial_state_target = \
                    f'{init.path}/ssh_adjustment/adjusted_init.nc'
            else:
                initial_state_target = \
                    f'{init.path}/initial_state/initial_state.nc'
            self.add_input_file(filename='init.nc',
                                work_dir_target=initial_state_target)
            self.add_input_file(
                filename='forcing_data.nc',
                work_dir_target=f'{init.path}/initial_state/'
                                f'init_mode_forcing_data.nc')
            self.add_input_file(
                filename='graph.info',
                work_dir_target=f'{init.path}/initial_state/graph.info')

        self.add_model_as_input()

    def setup(self):
        """
        Set the number of MPI tasks and the time step based on config options
        """
        config = self.config
        self.dynamic_ntasks = (self.ntasks is None and self.min_tasks is None)

        if self.dynamic_ntasks:
            mesh_filename = os.path.join(self.work_dir, 'init.nc')
            self.ntasks, self.min_tasks = get_ntasks_from_cell_count(
                config=config, at_setup=True, mesh_filename=mesh_filename)
            self.openmp_threads = config.getint('global_ocean',
                                                'forward_threads')

        if self.get_dt_from_min_res:
            dt, btr_dt = self._get_dts()
            if self.time_integrator == 'split_explicit_ab2':
                self.add_namelist_options({'config_dt': dt,
                                           'config_btr_dt': btr_dt})
            else:
                # RK4, so use the smaller time step
                self.add_namelist_options({'config_dt': btr_dt})
        super().setup()

    def constrain_resources(self, available_resources):
        """
        Update resources at runtime from config options
        """
        config = self.config
        if self.dynamic_ntasks:
            mesh_filename = os.path.join(self.work_dir, 'mesh.nc')
            self.ntasks, self.min_tasks = get_ntasks_from_cell_count(
                config=config, at_setup=False, mesh_filename=mesh_filename)
            self.openmp_threads = config.getint('global_ocean',
                                                'forward_threads')
        super().constrain_resources(available_resources)

    def runtime_setup(self):
        """
        Update the time step based on config options that a user might have
        changed
        """
        if self.get_dt_from_min_res:
            dt, btr_dt = self._get_dts()
            if self.time_integrator == 'split_explicit_ab2':
                self.update_namelist_at_runtime({'config_dt': dt,
                                                 'config_btr_dt': btr_dt})
            else:
                # RK4, so use the smaller time step
                self.update_namelist_at_runtime({'config_dt': btr_dt})
        super().runtime_setup()

    def run(self):
        """
        Run this step of the testcase
        """
        super().run()
        update_pio = self.config.getboolean('global_ocean',
                                            'forward_update_pio')
        run_model(self, update_pio=update_pio)
        add_mesh_and_init_metadata(self.outputs, self.config,
                                   init_filename='init.nc')

    def _get_dts(self):
        """
        Get the time step and barotropic time steps
        """
        config = self.config
        # dt is proportional to resolution: default 30 seconds per km
        dt_per_km = config.getfloat('global_ocean', 'dt_per_km')
        btr_dt_per_km = config.getfloat('global_ocean', 'btr_dt_per_km')
        min_res = config.getfloat('global_ocean', 'min_res')

        dt = dt_per_km * min_res
        # https://stackoverflow.com/a/1384565/7728169
        dt = time.strftime('%H:%M:%S', time.gmtime(dt))

        btr_dt = btr_dt_per_km * min_res
        btr_dt = time.strftime('%H:%M:%S', time.gmtime(btr_dt))

        return dt, btr_dt


class ForwardTestCase(TestCase):
    """
    A parent class for test cases for forward runs with global MPAS-Ocean mesh

    Attributes
    ----------
    mesh : compass.ocean.tests.global_ocean.mesh.Mesh
        The test case that produces the mesh for this run

    init : compass.ocean.tests.global_ocean.init.Init
        The test case that produces the initial condition for this run

    time_integrator : {'split_explicit_ab2', 'RK4'}
        The time integrator to use for the forward run
    """

    def __init__(self, test_group, mesh, init, time_integrator, name):
        """
        Create test case

        Parameters
        ----------
        test_group : compass.ocean.tests.global_ocean.GlobalOcean
            The global ocean test group that this test case belongs to

        mesh : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case that produces the mesh for this run

        init : compass.ocean.tests.global_ocean.init.Init
            The test case that produces the initial condition for this run

        time_integrator : {'split_explicit_ab2', 'RK4'}
            The time integrator to use for the forward run

        name : str
            the name of the test case
        """
        self.mesh = mesh
        self.init = init
        self.time_integrator = time_integrator
        subdir = get_forward_subdir(init.init_subdir, time_integrator, name)
        super().__init__(test_group=test_group, name=name, subdir=subdir)

    def configure(self):
        """
        Modify the configuration options for this test case
        """
        self.init.configure(config=self.config)


def get_forward_subdir(init_subdir, time_integrator, name):
    """
    Get the subdirectory for a test case that is based on a forward run with
    a time integrator
    """
    if time_integrator == 'split_explicit_ab2':
        # this is the default so we won't make a subdir for the time
        # integrator
        subdir = os.path.join(init_subdir, name)
    elif time_integrator == 'RK4':
        subdir = os.path.join(init_subdir, time_integrator, name)
    else:
        raise ValueError(f'Unexpected time integrator {time_integrator}')

    return subdir

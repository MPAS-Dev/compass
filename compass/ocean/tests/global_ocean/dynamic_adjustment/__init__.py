import glob
import importlib.resources
import os
import platform
from datetime import datetime, timedelta

import xarray as xr
from ruamel.yaml import YAML

from compass.ocean.tests.global_ocean.forward import (
    ForwardStep,
    ForwardTestCase,
)
from compass.validate import compare_variables


class DynamicAdjustment(ForwardTestCase):
    """
    A parent test case for performing dynamic adjustment (dissipating
    fast-moving waves) from an MPAS-Ocean initial condition.

    The final stage of the dynamic adjustment is assumed to be called
    ``simulation``, and is expected to have a file ``output.nc`` that can be
    compared against a baseline.

    Attributes
    ----------
    restart_filenames : list of str
        A list of restart files from each dynamic-adjustment step
    """

    def __init__(self, test_group, mesh, init, time_integrator):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.global_ocean.GlobalOcean
            The global ocean test group that this test case belongs to

        mesh : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case that produces the mesh for this run

        init : compass.ocean.tests.global_ocean.init.Init
            The test case that produces the initial condition for this run

        time_integrator : {'split_explicit', 'RK4'}
            The time integrator to use for the forward run
        """
        super().__init__(test_group=test_group, mesh=mesh, init=init,
                         time_integrator=time_integrator,
                         name='dynamic_adjustment')

        if time_integrator == 'split_explicit':
            yaml_filename = 'dynamic_adjustment.yaml'
        else:
            yaml_filename = f'dynamic_adjustment_{time_integrator}.yaml'

        yaml_filename = str(
            importlib.resources.files(mesh.package) / yaml_filename)
        text = importlib.resources.files(mesh.package).joinpath(
            yaml_filename).read_text()
        yaml_data = YAML(typ='rt')
        options_dict = yaml_data.load(text)
        if 'dynamic_adjustment' not in options_dict:
            raise ValueError(f'{yaml_filename} in {mesh.package} does not '
                             f'start with "dynamic_adjustment:" as expected.')
        options_dict = options_dict['dynamic_adjustment']
        if 'land_ice_flux_mode' in options_dict:
            land_ice_flux_mode = options_dict['land_ice_flux_mode']
        else:
            land_ice_flux_mode = 'pressure_only'

        if 'get_dt_from_min_res' not in options_dict:
            raise ValueError(f'{yaml_filename} in {mesh.package} does not '
                             f'have a "get_dt_from_min_res:" option as '
                             f'expected.')

        get_dt_from_min_res = options_dict['get_dt_from_min_res']

        shared_options = {
            'config_AM_globalStats_enable': '.true.',
            'config_AM_globalStats_compute_on_startup': '.true.',
            'config_AM_globalStats_write_on_startup': '.true.',
            'config_use_activeTracers_surface_restoring': '.true.'
        }
        if 'shared' in options_dict:
            # replace the defaults with values from shared options
            shared_options.update(options_dict['shared'])

        if 'steps' not in options_dict:
            raise ValueError(f'{yaml_filename} in {mesh.package} does not '
                             f'have a "steps:" item as expected.')

        start_time = '0001-01-01_00:00:00'
        previous_restart_filename = None
        restart_filenames = list()
        for step_name in options_dict['steps']:
            options = options_dict['steps'][step_name]
            restart_time, restart_filename = self._add_step(
                step_name, options, get_dt_from_min_res, time_integrator,
                yaml_filename, start_time, previous_restart_filename, mesh,
                init, land_ice_flux_mode, shared_options)
            start_time = restart_time
            previous_restart_filename = restart_filename
            restart_filenames.append(restart_filename)

        self.restart_filenames = restart_filenames

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        config = self.config
        variables = ['temperature', 'salinity', 'layerThickness',
                     'normalVelocity']

        compare_variables(test_case=self, variables=variables,
                          filename1='simulation/output.nc')

        temp_max = config.getfloat('dynamic_adjustment', 'temperature_max')
        max_values = {'temperatureMax': temp_max}

        for step_name in self.steps_to_run:
            step = self.steps[step_name]
            step_path = os.path.join(self.work_dir, step.subdir)
            global_stats_path = os.path.join(step_path, 'analysis_members',
                                             'globalStats.*.nc')
            global_stats_path = glob.glob(global_stats_path)
            for filename in global_stats_path:
                ds = xr.open_dataset(filename)
                for var, max_value in max_values.items():
                    max_in_global_stats = ds[var].max().values
                    if max_in_global_stats > max_value:
                        raise ValueError(
                            f'Max of {var} > allowed threshold: '
                            f'{max_in_global_stats} > {max_value} '
                            f'in {filename}')

    def _add_step(self, step_name, options, get_dt_from_min_res,
                  time_integrator, yaml_filename, start_time,
                  previous_restart_filename, mesh, init, land_ice_flux_mode,
                  shared_options):
        required = ['run_duration', 'output_interval', 'restart_interval']
        if not get_dt_from_min_res:
            required.append('dt')
            if time_integrator == 'split_explicit':
                required.append('btr_dt')
        for option in required:
            if option not in options:
                raise ValueError(
                    f'In {yaml_filename} in {mesh.package}, step '
                    f'{step_name} does not have required option {option}.')

        if get_dt_from_min_res:
            omit = ['dt', 'btr_dt']
            for option in omit:
                if option in options:
                    raise ValueError(
                        f'In {yaml_filename} in {mesh.package}, step '
                        f'{step_name} is getting dt from the minimum '
                        f'resolution of the mesh but dt and/or btr_dt '
                        f'are also specified.')

        run_duration = options['run_duration']
        restart_time = _get_restart_time(start_time, run_duration)

        step = ForwardStep(test_case=self, mesh=mesh, init=init,
                           time_integrator=time_integrator, name=step_name,
                           subdir=step_name,
                           land_ice_flux_mode=land_ice_flux_mode,
                           get_dt_from_min_res=get_dt_from_min_res)

        namelist_options = dict(shared_options)
        if previous_restart_filename is None:
            namelist_options['config_do_restart'] = '.false.'
        else:
            namelist_options['config_do_restart'] = '.true.'
        namelist_options['config_start_time'] = f"'{start_time}'"

        for option in ['run_duration', 'dt', 'btr_dt']:
            if option in options:
                namelist_options[f'config_{option}'] = \
                    f"'{options[option]}'"
        if ('Rayleigh_damping_coeff' in options and
                options['Rayleigh_damping_coeff'] != 'None'):
            namelist_options['config_implicit_bottom_drag_type'] = \
                "'constant_and_rayleigh'"
            namelist_options['config_Rayleigh_damping_coeff'] = \
                f"{options['Rayleigh_damping_coeff']}"
        else:
            namelist_options['config_implicit_bottom_drag_type'] = \
                "'constant'"

        step.add_namelist_options(namelist_options)

        stream_replacements = {
            'output_interval': options['output_interval'],
            'restart_interval': options['restart_interval']}
        package = 'compass.ocean.tests.global_ocean.dynamic_adjustment'
        step.add_streams_file(package, 'streams.template',
                              template_replacements=stream_replacements)

        restart_filename = \
            f'restarts/rst.{restart_time.replace(":", ".")}.nc'
        if previous_restart_filename is not None:
            step.add_input_file(filename=f'../{previous_restart_filename}')
        step.add_output_file(filename=f'../{restart_filename}')
        step.add_output_file(filename='output.nc')

        self.add_step(step)

        return restart_time, restart_filename


def _get_restart_time(start_time, run_duration):
    start = datetime.strptime(start_time, '%Y-%m-%d_%H:%M:%S')
    duration = _parse_duration(run_duration)
    restart = start + duration
    if platform.system() == 'Darwin':
        restart_time = restart.strftime('%Y-%m-%d_%H:%M:%S')
    else:
        restart_time = restart.strftime('%4Y-%m-%d_%H:%M:%S')
    return restart_time


def _parse_duration(duration_str):
    days = 0
    hours = 0
    minutes = 0
    split = duration_str.split('_')
    hms = split[-1]
    if len(split) > 1:
        days = int(split[-2])

    split = hms.split(':')
    seconds = int(split[-1])
    if len(split) > 1:
        minutes = int(split[-2])
    if len(split) > 2:
        hours = int(split[-3])
    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)

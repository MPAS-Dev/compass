"""
A single-timestep MALI melt diagnostic for one ocean state and melt form.
"""

import os

import xarray as xr
from mpas_tools.io import write_netcdf

from compass.landice.tests.ismip7_calibration.configure import (
    is_ismip7,
    mali_melt_method,
    uses_spatial_slope,
)
from compass.model import run_model
from compass.step import Step

#: namelist options that must exist in MALI's defaults for each melt form.
#: Compass only *warns* when an option is missing from the defaults, so
#: without an explicit check a MALI build without the ISMIP7 melt method
#: would silently drop these and produce a plausible but wrong calibration.
#: The ismip7_const form requires only the base ISMIP7 options (no PR #195
#: slope options), so it can run on an older MALI build; ismip7_slope requires
#: the full set including the five slope options from PR #195.
REQUIRED_OPTIONS = {
    'ismip7_const': ['config_ismip7_melt_sin_slope',
                     'config_ismip7_melt_coriolis',
                     'config_ismip7_melt_salinity',
                     'config_ismip7_melt_salinity_source',
                     'config_basal_mass_bal_float'],
    'ismip7_slope': ['config_ismip7_melt_sin_slope',
                     'config_ismip7_melt_coriolis',
                     'config_ismip7_melt_salinity',
                     'config_ismip7_melt_salinity_source',
                     'config_basal_mass_bal_float',
                     'config_ismip7_melt_spatially_variable_slope',
                     'config_ismip7_melt_max_slope',
                     'config_ismip7_melt_slope_smoothing_iterations',
                     'config_ismip7_melt_slope_method',
                     'config_ismip7_melt_slope_stencil_rings'],
    'ismip6': ['config_basal_mass_bal_float'],
}

#: the input-stream variable each melt form reads its parameter from
PARAMETER_VARIABLE = {'ismip7_const': 'ismip7shelfMelt_K',
                      'ismip7_slope': 'ismip7shelfMelt_K',
                      'ismip6': 'ismip6shelfMelt_gamma0'}


class RunState(Step):
    """
    A step that runs MALI for a single timestep to diagnose the melt field
    for one ocean state, using one melt form.

    Melt is computed from the geometry and the forcing alone, so no ice
    dynamics, no thermal evolution and no transient are needed.  The velocity
    solver is off, which also means Albany is not required.

    The run takes **one short timestep rather than none**, because MALI
    computes melt inside the timestep rather than in the initial diagnostic
    solve; a zero-length run would produce no melt at all.

    One run per ocean state suffices, not one per parameter value: melt is
    exactly proportional to the melt parameter, so the whole parameter
    ensemble follows by scaling a single run.  That is what turns a ~1300-run
    campaign into 28.

    Attributes
    ----------
    state_name : str
        Name of the ocean state this step runs

    melt_form : str
        ``'ismip7'`` for the Burgard local quadratic or ``'ismip6'`` for the
        non-local quadratic
    """

    def __init__(self, test_case, state_name, melt_form, subdir,
                 parameter_scale=1.0):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.landice.tests.ismip7_calibration.ais.Ais
            The test case this step belongs to

        state_name : str
            Name of the ocean state to run

        melt_form : {'ismip7', 'ismip6'}
            Which melt form to run

        subdir : str
            Subdirectory for this step

        parameter_scale : float, optional
            Multiple of the reference melt parameter to run at.  Only the
            linearity check uses anything other than 1.
        """
        self.state_name = state_name
        self.melt_form = melt_form
        self.parameter_scale = parameter_scale
        name = os.path.basename(subdir)
        super().__init__(test_case=test_case, name=name, subdir=subdir)

    def setup(self):
        """
        Set up this step of the test case
        """
        config = self.config
        section = config['ismip7_calibration']
        self.ntasks = section.getint('ntasks')
        self.min_tasks = self.ntasks
        base_path_mali = section.get('base_path_mali')
        mali_mesh_file = section.get('mali_mesh_file')
        timestep = section.get('timestep')

        _check_namelist_options(config, self.melt_form)
        _validate_slope_config(config, self.melt_form)

        self.add_input_file(filename='mesh.nc',
                            target=os.path.join(base_path_mali,
                                                mali_mesh_file))
        self.add_input_file(filename='masks.nc',
                            target='../remap_masks/'
                                   'ismip7_masks_on_mali.nc')
        self.add_input_file(filename='forcing.nc',
                            target=f'../remap_forcing/'
                                   f'forcing_{self.state_name}.nc')

        # MALI builds the partition filename from
        # config_block_decomp_file_prefix plus the task count; a symlink
        # called graph.info alone is silently not found
        self.add_input_file(
            filename=f'graph.info.part.{self.ntasks}',
            target=f'../make_graph/graph.info.part.{self.ntasks}')

        resource_location = 'compass.landice.tests.ismip7_calibration.ais'

        self.add_namelist_file(resource_location, 'namelist.landice',
                               out_name='namelist.landice')

        mali_method = mali_melt_method(self.melt_form)
        options = {'config_basal_mass_bal_float': f"'{mali_method}'",
                   'config_dt': f"'{timestep}'",
                   'config_run_duration': f"'{timestep}'"}
        options.update(_melt_namelist_options(config, self.melt_form,
                                              self.parameter_scale))
        self.add_namelist_options(options=options,
                                  out_name='namelist.landice')

        self.add_streams_file(
            resource_location, 'streams.landice.template',
            out_name='streams.landice',
            template_replacements={'mesh_file': 'mesh.nc',
                                   'masks_file': 'masks.nc',
                                   'parameter_file': 'melt_parameter.nc',
                                   'parameter_variable':
                                       PARAMETER_VARIABLE[self.melt_form],
                                   'forcing_file': 'forcing.nc',
                                   'output_interval': timestep})

        _write_parameter_file(config, self.melt_form, self.parameter_scale,
                              os.path.join(self.work_dir,
                                           'melt_parameter.nc'))

        self.add_model_as_input()
        self.add_output_file(filename='output_melt.nc')

    def runtime_setup(self):
        """
        Set the number of ocean layers from the forcing file

        MPAS takes dimension sizes from the input stream only, and the mesh
        file has no ``nISMIP6OceanLayers``.  Without this the 3-D forcing
        fields are allocated against a zero-length dimension and the run dies
        trying to allocate hundreds of GB.  It is read from the forcing
        rather than hard-coded, so it stays right if ISMIP7 changes the
        vertical grid.
        """
        super().runtime_setup()
        with xr.open_dataset('forcing.nc') as ds:
            n_layers = ds.sizes['nISMIP6OceanLayers']
        self.update_namelist_at_runtime(
            options={'config_nISMIP6OceanLayers': f'{n_layers}'},
            out_name='namelist.landice')

    def run(self):
        """
        Run this step of the test case
        """
        # the partition file is supplied ready-made alongside the mesh, so
        # there is no plain graph.info for gpmetis to partition
        run_model(self, partition_graph=False)


def _melt_namelist_options(config, melt_form, parameter_scale=1.0):
    """The namelist options specific to one melt form."""
    section = config['ismip7_calibration_melt']
    if is_ismip7(melt_form):
        # Base ISMIP7 options (all forms)
        options = {
            'config_ismip7_melt_sin_slope':
                repr(section.getfloat('sin_slope')),
            'config_ismip7_melt_coriolis':
                repr(section.getfloat('coriolis')),
            'config_ismip7_melt_salinity_source': "'constant'",
            'config_ismip7_melt_salinity':
                repr(section.getfloat('salinity')),
        }
        # Slope options (ismip7_slope only)
        if uses_spatial_slope(melt_form):
            options.update({
                'config_ismip7_melt_spatially_variable_slope': '.true.',
                'config_ismip7_melt_max_slope':
                    repr(section.getfloat('max_slope')),
                'config_ismip7_melt_slope_smoothing_iterations':
                    repr(section.getint('slope_smoothing_iterations')),
                'config_ismip7_melt_slope_method':
                    f"'{section.get('slope_method')}'",
                'config_ismip7_melt_slope_stencil_rings':
                    repr(section.getint('slope_stencil_rings'))})
        else:
            options['config_ismip7_melt_spatially_variable_slope'] = '.false.'
        return options
    return {}


def reference_parameter(config, melt_form):
    """
    The reference value of a melt form's parameter, from config.

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for the test case

    melt_form : {'ismip7_const', 'ismip7_slope', 'ismip6'}
        The melt form

    Returns
    -------
    value : float
        ``reference_k`` for ISMIP7 forms or ``reference_gamma0`` for ISMIP6
    """
    section = config['ismip7_calibration_melt']
    if is_ismip7(melt_form):
        return section.getfloat('reference_k')
    return section.getfloat('reference_gamma0')


def _write_parameter_file(config, melt_form, parameter_scale, filename):
    """
    Write the file MALI reads the melt parameter from.

    Both melt forms read their parameter from an input stream -- gamma0 for
    the ISMIP6 method, K for the ISMIP7 one -- so that a single parameter
    file carries a complete calibration.  The ensemble runs at a reference
    value that the aggregation divides out again; the linearity check runs
    at multiples of it.
    """
    value = reference_parameter(config, melt_form) * parameter_scale
    ds = xr.Dataset()
    ds[PARAMETER_VARIABLE[melt_form]] = value
    ds.attrs['note'] = (
        f'{PARAMETER_VARIABLE[melt_form]} at {parameter_scale:g} times the '
        f'reference value; melt is proportional to it and the calibration '
        f'scales this away')
    write_netcdf(ds, filename)


def _check_namelist_options(config, melt_form):
    """
    Check that MALI's default namelist has the options this melt form needs.

    Raises
    ------
    ValueError
        If any required option is missing, which means the MALI build does
        not support this melt form
    """
    defaults = config.get('namelists', 'forward')
    if not os.path.exists(defaults):
        raise FileNotFoundError(
            f'MALI default namelist not found at {defaults}.  Build MALI, or '
            f'set [paths] mpas_model (or [namelists] forward) in your config '
            f'file.')

    with open(defaults) as handle:
        text = handle.read()

    missing = [option for option in REQUIRED_OPTIONS[melt_form]
               if option not in text]
    if missing:
        mali_method = mali_melt_method(melt_form)
        raise ValueError(
            f"MALI's default namelist at\n  {defaults}\ndoes not contain "
            f"{', '.join(missing)}, so this build does not support the "
            f"'{melt_form}' form "
            f"(config_basal_mass_bal_float = '{mali_method}').\n"
            f"Compass only warns about namelist options it cannot find, so "
            f"without this check the run would silently use MALI's defaults "
            f"and produce a plausible but wrong calibration.\n"
            f"Build MALI from a branch that has the ISMIP7 melt "
            f"parameterization and point [paths] mpas_model at it.")


def _validate_slope_config(config, melt_form):
    """
    Validate the slope configuration options for the ismip7_slope form.

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for the test case

    melt_form : str
        The melt form

    Raises
    ------
    ValueError
        If any slope config option is invalid
    """
    if not uses_spatial_slope(melt_form):
        return  # Only validate for the ismip7_slope form
    section = config['ismip7_calibration_melt']

    slope_method = section.get('slope_method')
    if slope_method not in {'local', 'polyfit'}:
        raise ValueError(
            f"config slope_method must be 'local' or 'polyfit', but is "
            f"'{slope_method}'")

    max_slope = section.getfloat('max_slope')
    if max_slope <= 0.0:
        raise ValueError(
            f"config max_slope must be positive, but is {max_slope}")

    slope_smoothing_iterations = section.getint('slope_smoothing_iterations')
    if slope_smoothing_iterations < 0:
        raise ValueError(
            f"config slope_smoothing_iterations must be non-negative, but is "
            f"{slope_smoothing_iterations}")

    if slope_method == 'polyfit':
        slope_stencil_rings = section.getint('slope_stencil_rings')
        if slope_stencil_rings < 1:
            raise ValueError(
                f"config slope_stencil_rings must be >= 1 for polyfit method, "
                f"but is {slope_stencil_rings}")

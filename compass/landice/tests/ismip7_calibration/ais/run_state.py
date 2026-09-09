"""
A single-timestep MALI melt diagnostic for one ocean state and melt form.
"""

import os

import xarray as xr

from compass.model import run_model
from compass.step import Step

#: namelist options that must exist in MALI's defaults for each melt form.
#: Compass only *warns* when an option is missing from the defaults, so
#: without an explicit check a MALI build without the ISMIP7 melt method
#: would silently drop these and produce a plausible but wrong calibration.
REQUIRED_OPTIONS = {
    'ismip7': ['config_ismip7_melt_K', 'config_ismip7_melt_sin_slope',
               'config_ismip7_melt_coriolis', 'config_ismip7_melt_salinity',
               'config_ismip7_melt_salinity_source'],
    'ismip6': ['config_basal_mass_bal_float'],
}


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

    def __init__(self, test_case, state_name, melt_form, subdir):
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
        """
        self.state_name = state_name
        self.melt_form = melt_form
        name = f'{melt_form}_{state_name}'
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
        graph_file_prefix = section.get('graph_file_prefix')
        timestep = section.get('timestep')

        _check_namelist_options(config, self.melt_form)

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
            target=os.path.join(base_path_mali,
                                f'{graph_file_prefix}{self.ntasks}'))

        resource_location = 'compass.landice.tests.ismip7_calibration.ais'

        self.add_namelist_file(resource_location, 'namelist.landice',
                               out_name='namelist.landice')

        options = {'config_basal_mass_bal_float': f"'{self.melt_form}'",
                   'config_dt': f"'{timestep}'",
                   'config_run_duration': f"'{timestep}'"}
        options.update(_melt_namelist_options(config, self.melt_form))
        self.add_namelist_options(options=options,
                                  out_name='namelist.landice')

        self.add_streams_file(
            resource_location, 'streams.landice.template',
            out_name='streams.landice',
            template_replacements={'mesh_file': 'mesh.nc',
                                   'masks_file': 'masks.nc',
                                   'forcing_file': 'forcing.nc',
                                   'output_interval': timestep})

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


def _melt_namelist_options(config, melt_form):
    """The namelist options specific to one melt form."""
    section = config['ismip7_calibration_melt']
    if melt_form == 'ismip7':
        # the run is done at a reference parameter value; melt is exactly
        # proportional to it, so the ensemble is formed by scaling afterwards
        return {
            'config_ismip7_melt_K': repr(section.getfloat('reference_k')),
            'config_ismip7_melt_sin_slope':
                repr(section.getfloat('sin_slope')),
            'config_ismip7_melt_coriolis':
                repr(section.getfloat('coriolis')),
            'config_ismip7_melt_salinity_source': "'constant'",
            'config_ismip7_melt_salinity':
                repr(section.getfloat('salinity'))}
    # the ISMIP6 non-local form reads gamma0 from its input stream, so the
    # reference value is written into the masks file rather than set here
    return {}


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
        raise ValueError(
            f"MALI's default namelist at\n  {defaults}\ndoes not contain "
            f"{', '.join(missing)}, so this build does not support "
            f"config_basal_mass_bal_float = '{melt_form}'.\n"
            f"Compass only warns about namelist options it cannot find, so "
            f"without this check the run would silently use MALI's defaults "
            f"and produce a plausible but wrong calibration.\n"
            f"Build MALI from a branch that has the ISMIP7 melt "
            f"parameterization and point [paths] mpas_model at it.")

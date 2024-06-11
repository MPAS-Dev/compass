import os

import numpy as np
import xarray as xr
from mpas_tools.cime.constants import constants
from mpas_tools.io import write_netcdf
from pyremap import LatLonGridDescriptor, MpasCellMeshDescriptor, Remapper

from compass.step import Step


class RemapTopography(Step):
    """
    A step for remapping bathymetry and ice-shelf topography from a
    latitude-longitude grid to a global MPAS-Ocean mesh

    Attributes
    ----------
    base_mesh_step : compass.mesh.spherical.SphericalBaseStep
        The base mesh step containing input files to this step

    mesh_name : str
        The name of the MPAS mesh to include in the mapping file
    """

    def __init__(self, test_case, base_mesh_step, name='remap_topography',
                 subdir=None, mesh_name='MPAS_mesh'):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case this step belongs to

        base_mesh_step : compass.mesh.spherical.SphericalBaseStep
            The base mesh step containing input files to this step

        name : str, optional
            the name of the step

        subdir : str, optional
            the subdirectory for the step.  The default is ``name``

        mesh_name : str, optional
            The name of the MPAS mesh to include in the mapping file
        """
        super().__init__(test_case, name=name, subdir=subdir,
                         ntasks=None, min_tasks=None)
        self.base_mesh_step = base_mesh_step
        self.mesh_name = mesh_name

        self.add_output_file(filename='topography_remapped.nc')

    def setup(self):
        """
        Set up the step in the work directory, including downloading any
        dependencies.
        """
        super().setup()
        config = self.config
        topo_filename = config.get('remap_topography', 'topo_filename')
        self.add_input_file(
            filename='topography.nc',
            target=topo_filename,
            database='bathymetry_database')

        base_path = self.base_mesh_step.path
        base_filename = self.base_mesh_step.config.get(
            'spherical_mesh', 'mpas_mesh_filename')
        target = os.path.join(base_path, base_filename)
        self.add_input_file(filename='base_mesh.nc', work_dir_target=target)

        self.ntasks = config.getint('remap_topography', 'ntasks')
        self.min_tasks = config.getint('remap_topography', 'min_tasks')

    def constrain_resources(self, available_resources):
        """
        Constrain ``cpus_per_task`` and ``ntasks`` based on the number of
        cores available to this step

        Parameters
        ----------
        available_resources : dict
            The total number of cores available to the step
        """
        config = self.config
        self.ntasks = config.getint('remap_topography', 'ntasks')
        self.min_tasks = config.getint('remap_topography', 'min_tasks')
        super().constrain_resources(available_resources)

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        logger = self.logger
        parallel_executable = config.get('parallel', 'parallel_executable')

        lon_var = config.get('remap_topography', 'lon_var')
        lat_var = config.get('remap_topography', 'lat_var')
        method = config.get('remap_topography', 'method')
        renorm_threshold = config.getfloat('remap_topography',
                                           'renorm_threshold')
        ice_density = config.getfloat('remap_topography', 'ice_density')
        ocean_density = constants['SHR_CONST_RHOSW']
        g = constants['SHR_CONST_G']

        in_descriptor = LatLonGridDescriptor.read(fileName='topography.nc',
                                                  lonVarName=lon_var,
                                                  latVarName=lat_var)

        in_mesh_name = in_descriptor.meshName

        out_mesh_name = self.mesh_name
        out_descriptor = MpasCellMeshDescriptor(fileName='base_mesh.nc',
                                                meshName=self.mesh_name)

        mapping_file_name = \
            f'map_{in_mesh_name}_to_{out_mesh_name}_{method}.nc'
        remapper = Remapper(in_descriptor, out_descriptor, mapping_file_name)

        remapper.build_mapping_file(method=method, mpiTasks=self.ntasks,
                                    tempdir='.', logger=logger,
                                    esmf_parallel_exec=parallel_executable)

        remapper.remap_file(inFileName='topography.nc',
                            outFileName='topography_ncremap.nc',
                            logger=logger)

        ds_in = xr.open_dataset('topography_ncremap.nc')
        ds_in = ds_in.rename({'ncol': 'nCells'})
        ds_out = xr.Dataset()
        rename = {'bathymetry_var': 'bed_elevation',
                  'ice_thickness_var': 'landIceThkObserved',
                  'ice_frac_var': 'landIceFracObserved',
                  'grounded_ice_frac_var': 'landIceGroundedFracObserved',
                  'ocean_frac_var': 'oceanFracObserved',
                  'bathy_frac_var': 'bathyFracObserved'}

        for option, out_var in rename.items():
            in_var = config.get('remap_topography', option)
            ds_out[out_var] = ds_in[in_var]

        ds_out['landIceFloatingFracObserved'] = \
            ds_out.landIceFracObserved - ds_out.landIceGroundedFracObserved

        # make sure fractions don't exceed 1
        for var in ['landIceFracObserved', 'landIceGroundedFracObserved',
                    'landIceFloatingFracObserved', 'oceanFracObserved',
                    'bathyFracObserved']:
            ds_out[var] = np.minimum(ds_out[var], 1.)

        # renormalize elevation variables
        norm = ds_out.bathyFracObserved
        valid = norm > renorm_threshold
        for var in ['bed_elevation', 'landIceThkObserved']:
            ds_out[var] = xr.where(valid, ds_out[var] / norm, 0.)

        thickness = ds_out.landIceThkObserved
        ds_out['landIcePressureObserved'] = ice_density * g * thickness

        # compute the ice draft to be consistent with the land ice pressure
        # and using E3SM's density of seawater
        draft = - (ice_density / ocean_density) * thickness
        bed = ds_out.bed_elevation

        # can't be deeper than the bed
        draft = xr.where(draft >= bed, draft, bed)

        ds_out['landIceDraftObserved'] = draft

        ds_out['ssh'] = draft

        write_netcdf(ds_out, 'topography_remapped.nc')

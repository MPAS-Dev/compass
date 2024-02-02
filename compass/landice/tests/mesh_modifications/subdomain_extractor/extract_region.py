import shutil
import sys

import numpy as np
import xarray
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call
from mpas_tools.mesh.conversion import convert, cull
from pyremap import MpasCellMeshDescriptor, Remapper

from compass.landice.mesh import mpas_flood_fill
from compass.model import make_graph_file
from compass.step import Step


class ExtractRegion(Step):
    """
    A step for extracting a regional domain from a larger domain

    Attributes
    ----------
    """
    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='extract_region')

    def setup(self):
        self.ntasks = 128
        self.mintasks = 1

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger

        # Get info from config file
        config = self.config
        section = config['subdomain']
        source_file_path = section.get('source_file')
        source_file_name = source_file_path.split('/')[-1]
        region_mask_file = section.get('region_mask_file')
        region_number = section.getint('region_number')
        dest_file_name = section.get('dest_file_name')
        mesh_projection = section.get('mesh_projection')
        extend_mesh = section.getboolean('extend_ocean_buffer')
        grow_iters = section.getint('grow_iters')
        interp_method = section.get('interp_method')

        # get needed dims from source mesh
        ds_src = xarray.open_dataset(source_file_path)
        nCells = ds_src.sizes['nCells']
        levels = ds_src.sizes['nVertLevels']

        # create cull mask
        logger.info('creating cull mask file')
        dsMask = xarray.open_dataset(region_mask_file)
        regionCellMasks = dsMask['regionCellMasks'][:].values
        # get region mask for the requested region
        keepMask = regionCellMasks[:, region_number - 1]
        if extend_mesh:
            # Grow the mask into the ocean, because the standard regions
            # may end at the ice terminus.
            thickness = ds_src['thickness'][:].values
            bed = ds_src['bedTopography'][:].values
            oceanMask = np.squeeze((thickness[0, :] == 0.0) * (bed[0, :] <=
                                                               0.0))
            floatMask = np.squeeze(((thickness[0, :] * 910.0 / 1028.0 +
                                     bed[0, :]) < 0.0) *
                                   (thickness[0, :] > 0))
            conc = ds_src['cellsOnCell'][:].values
            neonc = ds_src['nEdgesOnCell'][:].values

            # First grow forward to capture any adjacent ice shelf
            print('Starting floating ice fill')
            keepMask = mpas_flood_fill(keepMask, floatMask, conc, neonc)

            # Don't grow into other regions.
            # The area to grow into is region adjacent to the domain that
            # either has no region assigned to it OR is open ocean.
            # We also want to fill into any *adjacent* floating ice, due to
            # some funky region boundaries near ice-shelf fronts.
            print('Starting ocean grow fill')
            noRegionMask = (np.squeeze(regionCellMasks.sum(axis=1)) == 0)
            growMask = np.logical_or(noRegionMask, oceanMask)
            print(f'sum norregion={growMask.sum()}, {growMask.shape}')
            keepMask = mpas_flood_fill(keepMask, growMask, conc, neonc,
                                       grow_iters=grow_iters)

        # To call 'cull' with an inverse mask, we need a dataset with the
        # mask saved to the field regionCellMasks
        outdata = {'regionCellMasks': (('nCells', 'nRegions'),
                                       keepMask.reshape(nCells, 1))}
        dsMaskOut = xarray.Dataset(data_vars=outdata)
        # For troubleshooting, one may want to inspect the mask, so write out
        # (otherwise not necessary to save to disk)
        write_netcdf(dsMaskOut, 'cull_mask.nc')

        # cull the mesh
        logger.info('culling and converting mesh')
        ds_out = cull(ds_src, dsInverse=dsMaskOut, logger=logger)

        # convert mesh
        ds_out = convert(ds_out, logger=logger)
        write_netcdf(ds_out, f'{source_file_name}_culled.nc')

        # mark horns for culling
        logger.info('Marking horns for culling')
        args = ['mark_horns_for_culling.py', '-f',
                f'{source_file_name}_culled.nc']
        check_call(args, logger=logger)

        # cull again
        logger.info('culling and converting mesh')
        ds_out = xarray.open_dataset(f'{source_file_name}_culled.nc')
        ds_out = cull(ds_out, logger=logger)
        ds_out = convert(ds_out, logger=logger)
        dest_mesh_only_name = 'dest_mesh_only.nc'
        write_netcdf(ds_out, dest_mesh_only_name)

        # set lat/lon
        logger.info('calling set_lat_lon_fields_in_planar_grid.py')
        args = ['set_lat_lon_fields_in_planar_grid.py',
                '-f', dest_mesh_only_name, '-p', mesh_projection]
        check_call(args, logger=logger)

        if interp_method == 'ncremap':
            # remap data from the original file to the culled mesh using
            # pyremap with nearest neighbor interpolation
            in_descriptor = MpasCellMeshDescriptor(source_file_path,
                                                   'src_mesh')
            out_descriptor = MpasCellMeshDescriptor(dest_mesh_only_name,
                                                    'dst_mesh')

            mapping_filename = 'map_src_to_dst_nstd.nc'

            logger.info(f'Creating the mapping file {mapping_filename}...')
            remapper = Remapper(in_descriptor, out_descriptor,
                                mapping_filename)

            parallel_executable = config.get('parallel',
                                             'parallel_executable')
            remapper.build_mapping_file(method='neareststod',
                                        mpiTasks=self.ntasks,
                                        tempdir=self.work_dir, logger=logger,
                                        esmf_parallel_exec=parallel_executable)  # noqa
            logger.info('done.')

            logger.info('Remapping mesh file...')
            # ncremap requires the spatial dimension to be the last one,
            # which MALI does not exclusively follow.  So we have to
            # permute dimensions before calling ncremap, and then permute back
            args = ['ncpdq', '-O', '-a',
                    'Time,nVertInterfaces,nVertLevels,nRegions,nISMIP6OceanLayers,nEdges,nCells',  # noqa
                    source_file_path, f'{source_file_name}_permuted.nc']
            check_call(args, logger=logger)
            args = ['ncremap', '-m', mapping_filename,
                    f'{source_file_name}_permuted.nc',
                    f'{dest_file_name}_permuted.nc']
            check_call(args, logger=logger)
            args = ['ncpdq', '-O', '-a',
                    'Time,nCells,nEdges,nVertInterfaces,nVertLevels,nRegions,nISMIP6OceanLayers',  # noqa
                    f'{dest_file_name}_permuted.nc',
                    f'{dest_file_name}_extra_var.nc']
            check_call(args, logger=logger)
            # drop some extra vars that ncremap adds
            args = ['ncks', '-O', '-C', '-x', '-v',
                    'lat,lon,lat_vertices,lon_vertices,area',
                    f'{dest_file_name}_extra_var.nc',
                    f'{dest_file_name}_vars_only.nc']
            check_call(args, logger=logger)

            # now combine the remapped variables with the mesh fields
            # that don't get remapped
            shutil.copyfile(dest_mesh_only_name, dest_file_name)
            args = ['ncks', '-A', f'{dest_file_name}_vars_only.nc',
                    dest_file_name]
            check_call(args, logger=logger)

            logger.info('done.')

        elif interp_method == 'mali_interp':
            # create landice mesh
            logger.info('calling create_landice_grid_from_generic_MPAS_grid.py')  # noqa
            args = ['create_landice_grid_from_generic_MPAS_grid.py',
                    '-i', f'{source_file_name}_culled_dehorned.nc',
                    '-o', dest_file_name,
                    '-l', f'{levels}', '-v', 'glimmer',
                    '--beta', '--thermal', '--obs', '--diri']
            check_call(args, logger=logger)

            # interpolate to new mesh using nearest neighbor to ensure we get
            # identical values
            logger.info('calling interpolate_to_mpasli_grid.py')
            args = ['interpolate_to_mpasli_grid.py',
                    '-s', source_file_path,
                    '-d', dest_file_name, '-m', 'n']
            check_call(args, logger=logger)
        else:
            sys.exit(f"Unknown interp_method of {interp_method}")

        # mark Dirichlet boundaries
        logger.info('Marking domain boundaries dirichlet')
        args = ['mark_domain_boundaries_dirichlet.py',
                '-f', dest_file_name]
        check_call(args, logger=logger)

        logger.info('creating graph.info')
        make_graph_file(mesh_filename=dest_file_name,
                        graph_filename='graph.info')

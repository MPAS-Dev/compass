import os
import shutil
import sys

import mpas_tools
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
        source_file_name = os.path.basename(source_file_path)
        source_file_rootname = source_file_name.rsplit('.nc', 1)[0]
        region_definition = section.get('region_definition')
        geojson_file = section.get('geojson_file')
        region_mask_file = section.get('region_mask_file')
        region_number = section.getint('region_number')
        dest_file_name = section.get('dest_file_name')
        dest_file_rootname = dest_file_name.rsplit('.nc', 1)[0]
        mesh_projection = section.get('mesh_projection')
        extend_mesh = section.getboolean('extend_ocean_buffer')
        grow_iters = section.getint('grow_iters')
        interp_method = section.get('interp_method')
        extra_file1 = section.get('extra_file1')
        extra_file2 = section.get('extra_file2')
        extra_file3 = section.get('extra_file3')
        extra_file4 = section.get('extra_file4')
        extra_file5 = section.get('extra_file5')

        # create a tmp dir for intermediate files
        tmpdir = os.path.join(self.work_dir, 'tmp')
        os.makedirs(tmpdir, exist_ok=True)

        # get needed dims from source mesh
        ds_src = xarray.open_dataset(source_file_path)
        nCells = ds_src.sizes['nCells']
        levels = ds_src.sizes['nVertLevels']

        # create cull mask
        if region_definition == 'geojson':
            args = ['compute_mpas_region_masks',
                    '-m', source_file_path,
                    '-o', 'cull_mask.nc',
                    '-g', geojson_file,
                    '--process_count', f'{self.ntasks}',
                    '--format', mpas_tools.io.default_format,
                    '--engine', mpas_tools.io.default_engine]
            check_call(args, logger=logger)
            dsMaskOut = xarray.open_dataset('cull_mask.nc')

        elif region_definition == 'region_mask_file':
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
                logger.info('Starting floating ice fill')
                keepMask = mpas_flood_fill(keepMask, floatMask, conc, neonc)

                # Don't grow into other regions.
                # The area to grow into is region adjacent to the domain that
                # either has no region assigned to it OR is open ocean.
                # We also want to fill into any *adjacent* floating ice, due to
                # some funky region boundaries near ice-shelf fronts.
                logger.info('Starting ocean grow fill')
                noRegionMask = (np.squeeze(regionCellMasks.sum(axis=1)) == 0)
                growMask = np.logical_or(noRegionMask, oceanMask)
                keepMask = mpas_flood_fill(keepMask, growMask, conc, neonc,
                                           grow_iters=grow_iters)

            # To call 'cull' with an inverse mask, we need a dataset with the
            # mask saved to the field regionCellMasks
            outdata = {'regionCellMasks': (('nCells', 'nRegions'),
                                           keepMask.reshape(nCells, 1))}
            dsMaskOut = xarray.Dataset(data_vars=outdata)
            # For troubleshooting, one may want to inspect the mask,
            # so write out (otherwise not necessary to save to disk)
            write_netcdf(dsMaskOut, os.path.join(tmpdir, 'cull_mask.nc'))
        else:
            sys.exit('ERROR: unknown value for region_definition='
                     f'{region_definition}')

        # cull the mesh
        logger.info('culling and converting mesh')
        ds_out = cull(ds_src, dsInverse=dsMaskOut, logger=logger)

        # convert mesh
        ds_out = convert(ds_out, logger=logger)
        write_netcdf(ds_out,
                     os.path.join(tmpdir,
                                  f'{source_file_rootname}_culled.nc'))

        # mark horns for culling
        logger.info('Marking horns for culling')
        args = ['mark_horns_for_culling.py', '-f',
                os.path.join(tmpdir, f'{source_file_rootname}_culled.nc')]
        check_call(args, logger=logger)

        # cull again
        logger.info('culling and converting mesh')
        ds_out = xarray.open_dataset(
            os.path.join(tmpdir, f'{source_file_rootname}_culled.nc'))
        ds_out = cull(ds_out, logger=logger)
        ds_out = convert(ds_out, logger=logger)
        dest_mesh_only_name = os.path.join(tmpdir, 'dest_mesh_only.nc')
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

            mapping_filename = os.path.join(tmpdir, 'map_src_to_dst_nstd.nc')

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

            _remap_with_ncremap(
                source_file_path,
                os.path.join(tmpdir, f'{dest_file_rootname}_vars_only.nc'),
                mapping_filename, logger, tmpdir)

            # now combine the remapped variables with the mesh fields
            # that don't get remapped
            shutil.copyfile(dest_mesh_only_name, dest_file_name)
            args = [
                'ncks', '-A',
                os.path.join(tmpdir, f'{dest_file_rootname}_vars_only.nc'),
                dest_file_name]
            check_call(args, logger=logger)

            logger.info('done.')
            logger.info(f'Created {dest_file_name}')

        elif interp_method == 'mali_interp':
            # create landice mesh
            logger.info('calling create_landice_grid_from_generic_MPAS_grid.py')  # noqa
            args = ['create_landice_grid_from_generic_MPAS_grid.py',
                    '-i', os.path.join(tmpdir, 'dest_mesh_only.nc'),
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
            sys.exit(f"Error: Unknown interp_method of {interp_method}")

        # mark Dirichlet boundaries
        logger.info('Marking domain boundaries dirichlet')
        args = ['mark_domain_boundaries_dirichlet.py',
                '-f', dest_file_name]
        check_call(args, logger=logger)

        logger.info('creating graph.info')
        make_graph_file(mesh_filename=dest_file_name,
                        graph_filename='graph.info')

        for extra_file in [extra_file1, extra_file2, extra_file3,
                           extra_file4, extra_file5]:
            if extra_file != "None":
                if interp_method != "ncremap":
                    sys.exit("Error: interpolating ancillary files is only "
                             "supported when interp_method=ncremap")
                dst_file = \
                    f'{dest_file_rootname}_{os.path.basename(extra_file)}'
                _remap_with_ncremap(extra_file,
                                    dst_file,
                                    mapping_filename,
                                    logger, tmpdir)
                logger.info(f'Created {dst_file}')


def _remap_with_ncremap(src_path, dst_file, mapping_filename, logger,
                        tmpdir='.'):
    """
    Remaps a file using ncremap

    Parameters
    ----------
    src_path : str
        path to source file

    dst_file : str
        name of the destination file that should be created

    mapping_filename : str
        name of already generated mapping file

    logger
        logger object

    tmpdir : str
        temp dir to write intermediate files, optional

    Returns
    -------
    """

    src_file_rootname = os.path.basename(src_path).rsplit('.nc', 1)[0]

    # ncremap requires the spatial dimension to be the last one,
    # which MALI does not exclusively follow.  So we have to
    # permute dimensions before calling ncremap, and then permute back
    args = ['ncpdq', '-O', '-a',
            'Time,nVertInterfaces,nVertLevels,nRegions,nISMIP6OceanLayers,nEdges,nCells',  # noqa
            src_path, os.path.join(tmpdir, f'{src_file_rootname}_permuted.nc')]
    check_call(args, logger=logger)
    args = ['ncremap', '-m', mapping_filename,
            os.path.join(tmpdir, f'{src_file_rootname}_permuted.nc'),
            os.path.join(tmpdir, f'{dst_file}_permuted.nc')]
    check_call(args, logger=logger)
    args = ['ncpdq', '-O', '-a',
            'Time,nCells,nEdges,nVertInterfaces,nVertLevels,nRegions,nISMIP6OceanLayers',  # noqa
            os.path.join(tmpdir, f'{dst_file}_permuted.nc'),
            os.path.join(tmpdir, f'{dst_file}_extra_var.nc')]
    check_call(args, logger=logger)
    # drop some extra vars that ncremap adds
    ds_out = xarray.open_dataset(os.path.join(tmpdir,
                                              f'{dst_file}_extra_var.nc'))
    ds_out = ds_out.drop_vars(['lat', 'lon', 'lat_vertices',
                               'lon_vertices', 'area'])
    # drop variables on vertices or edges, which will not have been
    # remapped properly
    drop_list = []
    for varname, da in ds_out.data_vars.items():
        if 'nVertices' in da.dims or 'nEdges' in da.dims:
            drop_list.append(varname)
    ds_out = ds_out.drop_vars(drop_list)
    write_netcdf(ds_out, dst_file)

import numpy as np
import xarray
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call
from mpas_tools.mesh.conversion import convert, cull

from compass.model import make_graph_file
from compass.step import Step


class ExtractRegion(Step):
    """
    A step for creating a mesh and initial condition for koge_bugt_s
    test cases

    Attributes
    ----------
    mesh_type : str
        The resolution or mesh type of the test case
    """
    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        mesh_type : str
            The resolution or mesh type of the test case
        """
        super().__init__(test_case=test_case, name='extract_region')

    # no setup() method is needed

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        config = self.config
        section = config['subdomain']

        source_file_path = section.get('source_file')
        source_file_name = source_file_path.split('/')[-1]
        region_mask_file = section.get('region_mask_file')
        region_number = section.getint('region_number')
        dest_file_name = section.get('dest_file_name')
        levels = section.getint('levels')
        grow_iters = section.getint('grow_iters')

        # create cull mask
        logger.info('creating cull mask file')
        dsMesh = xarray.open_dataset(source_file_path)
        dsMask = xarray.open_dataset(region_mask_file)
        regionCellMasks = dsMask['regionCellMasks'][:].values
        # get region mask for the requested region
        keepMask = regionCellMasks[:, region_number - 1]
        print(f'sum keepMask={keepMask.sum()}')
        # Now grow the mask into the ocean, because the standard regions
        # end at the ice terminus.  Don't grow into other regions.
        # The region to grow into is region adjacent to the domain that
        # either has no region assigned to it OR is open ocean.
        # We also want to fill into any *adjacent* floating ice, due to some
        # funky region boundaries near ice-shelf fronts.
        noRegionMask = (np.squeeze(regionCellMasks.sum(axis=1)) == 0)
        thickness = dsMesh['thickness'][:].values
        bed = dsMesh['bedTopography'][:].values
        oceanMask = np.squeeze((thickness[0, :] == 0.0) * (bed[0, :] <= 0.0))
        floatMask = np.squeeze(((thickness[0, :] * 910.0 / 1028.0 +
                                 bed[0, :]) < 0.0) *
                               (thickness[0, :] > 0))
        growMask = np.logical_or(np.logical_or(noRegionMask, oceanMask),
                                 floatMask)
        print(f'sum norregion={growMask.sum()}, {growMask.shape}')
        conc = dsMesh['cellsOnCell'][:].values
        neonc = dsMesh['nEdgesOnCell'][:].values
        nCells = dsMesh.dims['nCells']
        # find cells at current ocean bdy
        for iter in range(grow_iters):
            maskInd = np.nonzero(keepMask == 1)[0]
            print(f'iter={iter}, keepMask size={keepMask.sum()}')
            newKeepMask = keepMask.copy()
            for iCell in maskInd:
                neighs = conc[iCell, :neonc[iCell]] - 1
                neighs = neighs[neighs >= 0]  # drop garbage cell
                for jCell in neighs:
                    if growMask[jCell] == 1:
                        newKeepMask[jCell] = 1
            keepMask = newKeepMask.copy()
        # To call 'cull' with an inverse mask, we need a dataset with the
        # mask saved to the field regionCellMasks
        outdata = {'regionCellMasks': (('nCells', 'nRegions'),
                                       keepMask.reshape(nCells, 1)),
                   'growMask': (('nCells',),
                                growMask.astype(int))}
        dsMaskOut = xarray.Dataset(data_vars=outdata)
        # For troubleshooting, one may want to inspect the mask, so write out
        write_netcdf(dsMaskOut, 'cull_mask.nc')

        # cull the mesh
        logger.info('culling and converting mesh')
        dsMesh = cull(dsMesh, dsInverse=dsMaskOut, logger=logger)

        # convert mesh
        dsMesh = convert(dsMesh, logger=logger)
        write_netcdf(dsMesh, f'{source_file_name}_culled.nc')

        # mark horns for culling
        logger.info('Marking horns for culling')
        args = ['mark_horns_for_culling.py', '-f',
                f'{source_file_name}_culled.nc']
        check_call(args, logger=logger)

        # cull again
        logger.info('culling and converting mesh')
        dsMesh = xarray.open_dataset(f'{source_file_name}_culled.nc')
        dsMesh = cull(dsMesh, logger=logger)
        dsMesh = convert(dsMesh, logger=logger)
        write_netcdf(dsMesh, f'{source_file_name}_culled_dehorned.nc')

        # set lat/lon
#        logger.info('calling set_lat_lon_fields_in_planar_grid.py')
#        args = ['set_lat_lon_fields_in_planar_grid.py', '-f',
#                f'{source_file_name}_culled_dehorned.nc', '-p', 'gis-gimp']
#        check_call(args, logger=logger)

        # create landice mesh
        logger.info('calling create_landice_grid_from_generic_MPAS_grid.py')
        args = ['create_landice_grid_from_generic_MPAS_grid.py',
                '-i', f'{source_file_name}_culled_dehorned.nc',
                '-o', dest_file_name,
                '-l', f'{levels}', '-v', 'glimmer',
                '--beta', '--thermal', '--obs', '--diri']
        check_call(args, logger=logger)

        # interpolate to new mesh using NN
        logger.info('calling interpolate_to_mpasli_grid.py')
        args = ['interpolate_to_mpasli_grid.py',
                '-s', source_file_path,
                '-d', dest_file_name, '-m', 'n']
        check_call(args, logger=logger)

        # mark Dirichlet boundaries
        logger.info('Marking domain boundaries dirichlet')
        args = ['mark_domain_boundaries_dirichlet.py',
                '-f', dest_file_name]
        check_call(args, logger=logger)

        logger.info('creating graph.info')
        make_graph_file(mesh_filename=dest_file_name,
                        graph_filename='graph.info')

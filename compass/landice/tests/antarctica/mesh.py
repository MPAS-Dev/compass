import numpy as np
import netCDF4
import xarray
from matplotlib import pyplot as plt
from shutil import copyfile
import time
from scipy.interpolate import NearestNDInterpolator

from mpas_tools.mesh.creation import build_planar_mesh
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.planar_hex import make_planar_hex_mesh
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call
from mpas_tools.scrip.from_mpas import scrip_from_mpas

from compass.step import Step
from compass.model import make_graph_file
from compass.landice.mesh import gridded_flood_fill, \
    set_rectangular_geom_points_and_edges, \
    set_cell_width, get_dist_to_edge_and_GL


class Mesh(Step):
    """
    A step for creating a mesh and initial condition for Antarctica test cases

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
        super().__init__(test_case=test_case, name='mesh')

        self.add_output_file(filename='graph.info')
        self.add_output_file(filename='Antarctica.nc')
        self.add_input_file(
            filename='antarctica_8km_2020_10_20.nc',
            target='antarctica_8km_2020_10_20.nc',
            database='')

    # no setup() method is needed

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        config = self.config
        section = config['antarctica']
        data_path = '/usr/projects/climate/trhille/data/'
        nProcs = section.get('nProcs')
        logger.info('calling build_cell_width')
        cell_width, x1, y1, geom_points, geom_edges, floodFillMask = \
            self.build_cell_width()
        logger.info('calling build_planar_mesh')
        build_planar_mesh(cell_width, x1, y1, geom_points,
                          geom_edges, logger=logger)
        dsMesh = xarray.open_dataset('base_mesh.nc')
        logger.info('culling mesh')
        dsMesh = cull(dsMesh, logger=logger)
        logger.info('converting to MPAS mesh')
        dsMesh = convert(dsMesh, logger=logger)
        logger.info('writing grid_converted.nc')
        write_netcdf(dsMesh, 'grid_converted.nc')

        levels = section.get('levels')
        logger.info('calling create_landice_grid_from_generic_MPAS_grid.py')
        args = ['create_landice_grid_from_generic_MPAS_grid.py',
                '-i', 'grid_converted.nc',
                '-o', 'ais_8km_preCull.nc',
                '-l', levels, '-v', 'glimmer']
        check_call(args, logger=logger)

        # Apply floodFillMask to thickness field to help with culling
        copyfile('antarctica_8km_2020_10_20.nc',
                 'antarctica_8km_2020_10_20_floodFillMask.nc')
        gg = netCDF4.Dataset('antarctica_8km_2020_10_20_floodFillMask.nc',
                             'r+')
        gg.variables['thk'][0, :, :] *= floodFillMask
        gg.variables['vx'][0, :, :] *= floodFillMask
        gg.variables['vy'][0, :, :] *= floodFillMask
        gg.close()

       # Now deal with the peculiarities of the AIS dataset. This section
       # could be separated into its own function
        copyfile('antarctica_8km_2020_10_20_floodFillMask.nc',
                'antarctica_8km_2020_10_20_floodFillMask_filledFields.nc')
        data = netCDF4.Dataset(
                'antarctica_8km_2020_10_20_floodFillMask_filledFields.nc',
                'r+')
        data.set_auto_mask(False)
        x1 = data.variables["x1"][:]
        y1 = data.variables["y1"][:]
        cellsWithIce = data.variables["thk"][:].ravel() > 0.
        data.createVariable('iceMask', 'f', ('time', 'y1', 'x1'))
        data.variables['iceMask'][:] = data.variables["thk"][:] > 0.

        # Note: dhdt is only reported over grounded ice, so we will have to
        # either update the dataset to include ice shelves or give them values of
        # 0 with reasonably large uncertainties.
        dHdt = data.variables["dhdt"][:]
        dHdtErr = 0.05 * dHdt #assign arbitrary uncertainty of 5%
        dHdtErr[dHdt > 1.e30] = 1.  # Where dHdt data are missing, set large uncertainty

        xGrid, yGrid = np.meshgrid(x1,y1)
        xx = xGrid.ravel()
        yy = yGrid.ravel()
        for field in ['thk', 'bheatflx', 'vx', 'vy',
                      'ex', 'ey', 'thkerr', 'dhdt']:
            tic = time.perf_counter()
            logger.info('Beginning building interpolator for {}'.format(field))
            if field in ['thk', 'thkerr']:
                mask = cellsWithIce.ravel()
            elif field == 'bheatflx':
                mask = np.logical_and(
                        data.variables[field][:].ravel() < 1.0e9,
                        data.variables[field][:].ravel() != 0.0)
            elif field in ['vx', 'vy', 'ex', 'ey', 'dhdt']:
                mask = np.logical_and(
                         data.variables[field][:].ravel() < 1.0e9,
                         cellsWithIce.ravel() > 0)
            else:
                mask = cellsWithIce
            interp = NearestNDInterpolator(
                                   list(zip(xx[mask], yy[mask])),
                                   data.variables[field][:].ravel()[mask])
            toc = time.perf_counter()
            logger.info('Finished building interpolator in {} seconds'.format(
                            toc - tic))

            tic = time.perf_counter()
            logger.info('Beginning interpolation for {}'.format(field))
            data.variables[field][0, :] = interp(xGrid,yGrid)
            toc = time.perf_counter()
            logger.info('Interpolation completed in {} seconds'.format(
                            toc - tic))

        bigToc = time.perf_counter()
        logger.info('All interpolations completed in {} seconds'.format(
                        toc - tic))

        data.createVariable('dHdtErr', 'f', ('time', 'y1', 'x1'))
        data.variables['dHdtErr'][:] = dHdtErr

        data.createVariable('vErr', 'f', ('time', 'y1', 'x1'))
        data.variables['vErr'][:] = np.sqrt(data.variables['ex'][:]**2
                                            + data.variables['ey'][:]**2)

        data.variables['bheatflx'][:] *= -1.e-3  # correct units and sign
        data.variables['bheatflx'].units = 'W m-2'

        data.variables['subm'][:] *= -1.0  # correct basal melting sign
        data.variables['subm_ss'][:] *= -1.0 

        data.renameVariable('dhdt', 'dHdt')
        data.renameVariable('thkerr', 'topgerr')

        data.createVariable('x', 'f', ('x1'))
        data.createVariable('y', 'f', ('y1'))
        data.variables['x'][:] = x1
        data.variables['y'][:] = y1

        data.close()

        logger.info('calling interpolate_to_mpasli_grid.py')
        args = ['interpolate_to_mpasli_grid.py', '-s',
                'antarctica_8km_2020_10_20_floodFillMask.nc', '-d',
                'ais_8km_preCull.nc', '-m', 'b', '-t']

        check_call(args, logger=logger)

       # Cull a certain distance from the ice margin
        cullCells = section.get('cull_cells')
        logger.info('calling define_cullMask.py')
        args = ['define_cullMask.py', '-f',
                'ais_8km_preCull.nc', '-m',
                'numCells', '-n', cullCells]

        check_call(args, logger=logger)

        dsMesh = xarray.open_dataset('ais_8km_preCull.nc')
        dsMesh = cull(dsMesh, logger=logger)
        write_netcdf(dsMesh, 'antarctica_culled.nc')

        logger.info('Marking horns for culling')
        args = ['mark_horns_for_culling.py', '-f', 'antarctica_culled.nc']
        check_call(args, logger=logger)

        logger.info('culling and converting')
        dsMesh = xarray.open_dataset('antarctica_culled.nc')
        dsMesh = cull(dsMesh, logger=logger)
        dsMesh = convert(dsMesh, logger=logger)
        write_netcdf(dsMesh, 'antarctica_dehorned.nc')

        logger.info('calling create_landice_grid_from_generic_MPAS_grid.py')
        args = ['create_landice_grid_from_generic_MPAS_grid.py', '-i',
                'antarctica_dehorned.nc', '-o',
                'Antarctica.nc', '-l', levels, '-v', 'glimmer',
                '--beta', '--thermal', '--obs', '--diri']

        check_call(args, logger=logger)

        # Add iceMask for later trimming. Could be added to
        # calling create_landice_grid_from_generic_MPAS_grid.py
        # if we want to make this a typical feature.
        data = netCDF4.Dataset('Antarctica.nc', 'r+')
        data.createVariable('iceMask', 'f', ('Time', 'nCells'))
        data.variables['iceMask'][:] = 0.
        data.close()

        logger.info('creating scrip file for BedMachine dataset')
        args = ['create_SCRIP_file_from_planar_rectangular_grid.py',
                '-i', data_path+'BedMachineAntarctica_2020-07-15_v02.nc',
                '-s', 'BedMachineAntarctica_2020-07-15_v02.scrip.nc',
                '-p', 'ais-bedmap2', '-r', '2']
        check_call(args, logger=logger)

        logger.info('creating scrip file for velocity dataset')
        args = ['create_SCRIP_file_from_planar_rectangular_grid.py',
                '-i', data_path+'antarctica_ice_velocity_450m_v2_edits.nc',
                '-s', 'antarctica_ice_velocity_450m_v2.scrip.nc',
                '-p', 'ais-bedmap2', '-r', '2']
        check_call(args, logger=logger)

        logger.info('calling set_lat_lon_fields_in_planar_grid.py')
        args = ['set_lat_lon_fields_in_planar_grid.py', '-f',
                'Antarctica.nc', '-p', 'ais-bedmap2']
        check_call(args, logger=logger)

        logger.info('creating scrip file for destination mesh')
        scrip_from_mpas('Antarctica.nc', 'Antarctica.scrip.nc')
        args = ['create_SCRIP_file_from_MPAS_mesh.py',
                '-m', 'Antarctica.nc',
                '-s', 'Antarctica.scrip.nc']
        check_call(args, logger=logger)
        # Testing shows 10 badger/grizzly nodes works well. 2 nodes is too few.
        # I have tested anything in betwee.
        logger.info('generating gridded dataset -> MPAS weights')
        args = ['srun', '-n', nProcs, 'ESMF_RegridWeightGen', '--source',
                'BedMachineAntarctica_2020-07-15_v02.scrip.nc',
                '--destination',
                'Antarctica.scrip.nc',
                '--weight', 'BedMachine_to_MPAS_weights.nc',
                '--method', 'conserve',
                "-i", "-64bit_offset",
                "--dst_regional", "--src_regional", '--netcdf4']
        check_call(args, logger=logger)

        logger.info('generating gridded dataset -> MPAS weights')
        args = ['srun', '-n', nProcs, 'ESMF_RegridWeightGen', '--source',
                'antarctica_ice_velocity_450m_v2.scrip.nc',
                '--destination',
                'Antarctica.scrip.nc',
                '--weight', 'measures_to_MPAS_weights.nc',
                '--method', 'conserve',
                "-i", "-64bit_offset", '--netcdf4',
                "--dst_regional", "--src_regional", '--ignore_unmapped']
        check_call(args, logger=logger)
        # Must add iceMask to interpolation script.
        # interpolate fields from composite dataset
        logger.info('calling interpolate_to_mpasli_grid.py')
        args = ['interpolate_to_mpasli_grid.py', '-s',
                'antarctica_8km_2020_10_20_floodFillMask_filledFields.nc',
                '-d', 'Antarctica.nc', '-m', 'd', '-v',
                'floatingBasalMassBal', 'basalHeatFlux', 'sfcMassBal',
                'surfaceAirTemperature', 'observedThicknessTendency',
                 'observedThicknessTendencyUncertainty']
        check_call(args, logger=logger)

        # interpoalte fields from BedMachine and Measures
        # Using conservative remapping
        logger.info('calling interpolate_to_mpasli_grid.py')
        args = ['interpolate_to_mpasli_grid.py', '-s',
                data_path+'BedMachineAntarctica_2020-07-15_v02_edits.nc',
                '-d', 'Antarctica.nc', '-m', 'e',
                '-w', 'BedMachine_to_MPAS_weights.nc']
        check_call(args, logger=logger)

        logger.info('calling interpolate_to_mpasli_grid.py')
        args = ['interpolate_to_mpasli_grid.py', '-s',
                data_path+'antarctica_ice_velocity_450m_v2_edits.nc',
                '-d', 'Antarctica.nc', '-m', 'e',
                '-w', 'measures_to_MPAS_weights.nc',
                '-v', 'observedSurfaceVelocityX',
                'observedSurfaceVelocityY',
                'observedSurfaceVelocityUncertainty']
        check_call(args, logger=logger)

        logger.info('Marking domain boundaries dirichlet')
        args = ['mark_domain_boundaries_dirichlet.py',
                '-f', 'Antarctica.nc']
        check_call(args, logger=logger)

        logger.info('creating graph.info')
        make_graph_file(mesh_filename='Antarctica.nc',
                        graph_filename='graph.info')

        # Clean up: trim to iceMask
        data = netCDF4.Dataset('Antarctica.nc', 'r+')
        data.set_auto_mask(False)
        #data.variables['thickness'][:] *= (data.variables['iceMask'][:] > 0.5)
        
        mask = np.logical_or(
                np.isnan(
                    data.variables['observedSurfaceVelocityUncertainty'][:]),
                data.variables['thickness'][:] < 1.0)
        data.variables['observedSurfaceVelocityUncertainty'][0,mask[0,:]] = 1.0
        data.close()

    def build_cell_width(self):
        """
        Determine MPAS mesh cell size based on user-defined density function.

        This includes hard-coded definition of the extent of the regional
        mesh and user-defined mesh density functions based on observed flow
        speed and distance to the ice margin.
        """
        # get needed fields from AIS dataset
        f = netCDF4.Dataset('antarctica_8km_2020_10_20.nc', 'r')
        f.set_auto_mask(False)  # disable masked arrays

        x1 = f.variables['x1'][:]
        y1 = f.variables['y1'][:]
        thk = f.variables['thk'][0, :, :]
        topg = f.variables['topg'][0, :, :]
        vx = f.variables['vx'][0, :, :]
        vy = f.variables['vy'][0, :, :]

        # Define extent of region to mesh.
        # These coords are specific to the Antarctica mesh.
        xx0 = -3333500
        xx1 = 3330500
        yy0 = -3333500
        yy1 = 3330500
        geom_points, geom_edges = set_rectangular_geom_points_and_edges(
            xx0, xx1, yy0, yy1)

        # Remove ice not connected to the ice sheet.
        floodMask = gridded_flood_fill(thk)
        thk[floodMask == 0] = 0.0
        vx[floodMask == 0] = 0.0
        vy[floodMask == 0] = 0.0

        # Calculate distance from each grid point to ice edge
        # and grounding line, for use in cell spacing functions.
        distToEdge, distToGL = get_dist_to_edge_and_GL(self, thk, topg, x1,
                                                       y1, window_size=1.e5)
        # optional - plot distance calculation
        # plt.pcolor(distToEdge/1000.0); plt.colorbar(); plt.show()

        # Set cell widths based on mesh parameters set in config file
        cell_width = set_cell_width(self, section='antarctica', thk=thk,
                                    vx=vx, vy=vy, dist_to_edge=distToEdge,
                                    dist_to_grounding_line=distToGL)
        # plt.pcolor(cell_width); plt.colorbar(); plt.show()

        return (cell_width.astype('float64'), x1.astype('float64'),
                y1.astype('float64'), geom_points, geom_edges, floodMask)

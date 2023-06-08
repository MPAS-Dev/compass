import time

import numpy as np
import xarray
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.planar_hex import make_planar_hex_mesh

from compass.ocean.vertical import init_vertical_coord
from compass.step import Step


class InitialState(Step):
    """
    A step for creating a mesh and initial condition for the
    lock-exchange test case
    """
    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
        """
        super().__init__(test_case=test_case, name='initial_state')

        for file in ['base_mesh.nc', 'culled_mesh.nc', 'culled_graph.info',
                     'initial_state.nc']:
            self.add_output_file(file)

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        logger = self.logger

        timeStart = time.time()

        section = config['horizontal_grid']
        nx = section.getint('nx')
        ny = section.getint('ny')
        dc = section.getfloat('dc')

        dsMesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc, nonperiodic_x=True,
                                      nonperiodic_y=False)
        write_netcdf(dsMesh, 'base_mesh.nc')

        dsMesh = cull(dsMesh, logger=logger)
        dsMesh = convert(dsMesh, graphInfoFileName='culled_graph.info',
                         logger=logger)
        write_netcdf(dsMesh, 'culled_mesh.nc')

        section = config['vertical_grid']
        maxDepth = section.getfloat('bottom_depth')
        nVertLevels = section.getint('vert_levels')

        section = config['lock_exchange']
        config_eos_linear_alpha = section.getfloat('eos_linear_alpha')
        config_eos_linear_Tref = section.getfloat('eos_linear_Tref')
        config_eos_linear_Sref = section.getfloat('eos_linear_Sref')
        config_eos_linear_densityref = section.getfloat(
            'eos_linear_densityref')
        middle_domain = section.getfloat('middle')
        lower_density = section.getfloat('lower_density')
        higher_density = section.getfloat('higher_density')

        # comment('obtain dimensions and mesh variables')
        # vertical_coordinate = 'uniform'

        ds = dsMesh.copy()
        nCells = ds.nCells.size
        nEdges = ds.nEdges.size
        nVertices = ds.nVertices.size

        xCell = ds.xCell

        # comment('create and initialize variables')
        time1 = time.time()

        ds['bottomDepth'] = maxDepth * xarray.ones_like(xCell)
        ds['ssh'] = xarray.zeros_like(xCell)

        init_vertical_coord(config, ds)

        # initial salinity, density, temperature
        ds['salinity'] = (config_eos_linear_Sref *
                          xarray.ones_like(ds.zMid)).where(ds.cellMask)
        ds['density'] = xarray.ones_like(ds.zMid).where(ds.cellMask)
        density = ds['density']
        for iCell in range(0, nCells):
            if (xCell[iCell] <= middle_domain):
                density[0, iCell, :] = lower_density
            else:
                density[0, iCell, :] = higher_density
        # T = Tref - (rho - rhoRef)/alpha
        ds['temperature'] = \
            (config_eos_linear_Tref -
             (ds.density - config_eos_linear_densityref) /
             config_eos_linear_alpha)

        # initial velocity on edges
        ds['normalVelocity'] = (('Time', 'nEdges', 'nVertLevels',),
                                np.zeros([1, nEdges, nVertLevels]))

        # Coriolis parameter
        ds['fCell'] = (('nCells', 'nVertLevels',),
                       np.zeros([nCells, nVertLevels]))
        ds['fEdge'] = (('nEdges', 'nVertLevels',),
                       np.zeros([nEdges, nVertLevels]))
        ds['fVertex'] = (('nVertices', 'nVertLevels',),
                         np.zeros([nVertices, nVertLevels]))

        print(f'   time: {time.time() - time1}')

        # comment('finalize and write file')
        time1 = time.time()

        # If you prefer not to have NaN as the fill value, you should consider
        # using mpas_tools.io.write_netcdf() instead
        write_netcdf(ds, 'initial_state.nc')
        print(f'   time: {time.time() - time1}')
        print(f'Total time: {time.time() - timeStart}')

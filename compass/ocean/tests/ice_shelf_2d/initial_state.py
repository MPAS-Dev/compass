import numpy as np
import xarray
from mpas_tools.cime.constants import constants
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.planar_hex import make_planar_hex_mesh

from compass.ocean.iceshelf import compute_land_ice_pressure_and_draft
from compass.ocean.vertical import init_vertical_coord
from compass.step import Step


class InitialState(Step):
    """
    A step for creating a mesh and initial condition for ice-shelf 2D test
    cases

    Attributes
    ----------
    resolution : str
        The resolution of the test case
    """
    def __init__(self, test_case, resolution):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolution : str
            The resolution of the test case
        """
        super().__init__(test_case=test_case, name='initial_state')
        self.resolution = resolution

        for file in ['base_mesh.nc', 'culled_mesh.nc', 'culled_graph.info',
                     'initial_state.nc']:
            self.add_output_file(file)

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        logger = self.logger

        section = config['ice_shelf_2d']
        nx = section.getint('nx')
        ny = section.getint('ny')
        dc = section.getfloat('dc')

        dsMesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc, nonperiodic_x=False,
                                      nonperiodic_y=True)
        write_netcdf(dsMesh, 'base_mesh.nc')

        dsMesh = cull(dsMesh, logger=logger)
        dsMesh = convert(dsMesh, graphInfoFileName='culled_graph.info',
                         logger=logger)
        write_netcdf(dsMesh, 'culled_mesh.nc')

        bottom_depth = config.getfloat('vertical_grid', 'bottom_depth')

        section = config['ice_shelf_2d']
        temperature = section.getfloat('temperature')
        surface_salinity = section.getfloat('surface_salinity')
        bottom_salinity = section.getfloat('bottom_salinity')

        # points 1 and 2 are where angles on ice shelf are located.
        # point 3 is at the surface.
        # d variables are total water-column thickness below ice shelf
        y1 = section.getfloat('y1')
        y2 = section.getfloat('y2')
        y3 = y2 + section.getfloat('edge_width')
        d1 = section.getfloat('cavity_thickness')
        d2 = d1 + section.getfloat('slope_height')
        d3 = bottom_depth

        ds = dsMesh.copy()

        ds['bottomDepth'] = bottom_depth * xarray.ones_like(ds.xCell)

        yCell = ds.yCell

        column_thickness = xarray.where(
            yCell < y1, d1, d1 + (d2 - d1) * (yCell - y1) / (y2 - y1))
        column_thickness = xarray.where(
            yCell < y2, column_thickness,
            d2 + (d3 - d2) * (yCell - y2) / (y3 - y2))
        column_thickness = xarray.where(yCell < y3, column_thickness, d3)

        ds['ssh'] = -bottom_depth + column_thickness

        # set up the vertical coordinate
        init_vertical_coord(config, ds)

        modify_mask = xarray.where(yCell < y3, 1, 0).expand_dims(
            dim='Time', axis=0)
        landIceFraction = modify_mask.astype(float)
        landIceMask = modify_mask.copy()
        landIceFloatingFraction = landIceFraction.copy()
        landIceFloatingMask = landIceMask.copy()

        ref_density = constants['SHR_CONST_RHOSW']
        landIcePressure, landIceDraft = compute_land_ice_pressure_and_draft(
            ssh=ds.ssh, modify_mask=modify_mask, ref_density=ref_density)

        salinity = surface_salinity + ((bottom_salinity - surface_salinity) *
                                       (ds.zMid / (-bottom_depth)))
        salinity, _ = xarray.broadcast(salinity, ds.layerThickness)
        salinity = salinity.transpose('Time', 'nCells', 'nVertLevels')

        normalVelocity = xarray.zeros_like(ds.xEdge)
        normalVelocity, _ = xarray.broadcast(normalVelocity, ds.refBottomDepth)
        normalVelocity = normalVelocity.transpose('nEdges', 'nVertLevels')
        normalVelocity = normalVelocity.expand_dims(dim='Time', axis=0)

        ds['temperature'] = temperature * xarray.ones_like(ds.layerThickness)
        ds['salinity'] = salinity
        ds['normalVelocity'] = normalVelocity
        ds['fCell'] = xarray.zeros_like(ds.xCell)
        ds['fEdge'] = xarray.zeros_like(ds.xEdge)
        ds['fVertex'] = xarray.zeros_like(ds.xVertex)
        ds['modifyLandIcePressureMask'] = modify_mask
        ds['landIceFraction'] = landIceFraction
        ds['landIceFloatingFraction'] = landIceFloatingFraction
        ds['landIceMask'] = landIceMask
        ds['landIceFloatingMask'] = landIceFloatingMask
        ds['landIcePressure'] = landIcePressure
        ds['landIceDraft'] = landIceDraft

        write_netcdf(ds, 'initial_state.nc')

        # Generate the tidal forcing dataset whether it is used or not
        ds_forcing = xarray.Dataset()
        y_max = np.max(ds.yCell.values)
        ds_forcing['tidalInputMask'] = xarray.where(
            ds.yCell > (y_max - 0.6 * 5.0e3), 1.0, 0.0)
        write_netcdf(ds_forcing, 'init_mode_forcing_data.nc')

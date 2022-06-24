import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import mpas_tools.mesh.creation.mesh_definition_tools as mdt
from mpas_tools.mesh.creation.signed_distance import \
    signed_distance_from_geojson, mask_from_geojson
from geometric_features import read_feature_collection
from mpas_tools.cime.constants import constants
from mpas_tools.viz.colormaps import register_sci_viz_colormaps

from compass.ocean.tests.global_ocean.mesh.mesh import MeshStep


class ARM10to60Mesh(MeshStep):
    """
    A step for creating SOwISC12to60 meshes
    """
    def __init__(self, test_case, mesh_name, with_ice_shelf_cavities):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.Mesh
            The test case this step belongs to

        mesh_name : str
            The name of the mesh

        with_ice_shelf_cavities : bool
            Whether the mesh includes ice-shelf cavities
        """

        super().__init__(test_case, mesh_name, with_ice_shelf_cavities,
                         package=self.__module__,
                         mesh_config_filename='arm10to60.cfg')

        inputs = ['Americas_land_mask.geojson',
                  'Atlantic_region.geojson',
                  'Europe_Africa_land_mask.geojson']
        for filename in inputs:
            self.add_input_file(filename=filename,
                                package=self.__module__)

    def build_cell_width_lat_lon(self):
        """
        Create cell width array for this mesh on a regular latitude-longitude
        grid

        Returns
        -------
        cellWidth : numpy.array
            m x n array of cell width in km

        lon : numpy.array
            longitude in degrees (length n and between -180 and 180)

        lat : numpy.array
            longitude in degrees (length m and between -90 and 90)
        """

        dlon = 0.1
        dlat = dlon
        earth_radius = constants['SHR_CONST_REARTH']
        print('\nCreating cellWidth on a lat-lon grid of: {0:.2f} x {0:.2f} '
              'degrees'.format(dlon, dlat))
        print('This can be set higher for faster test generation\n')
        nlon = int(360. / dlon) + 1
        nlat = int(180. / dlat) + 1
        lon = np.linspace(-180., 180., nlon)
        lat = np.linspace(-90., 90., nlat)
        km = 1.0e3

        print('plotting ...')
        plt.switch_backend('Agg')
        register_sci_viz_colormaps()
        fig = plt.figure()
        plt.clf()
        fig.set_size_inches(10.0, 10.0)
        register_sci_viz_colormaps()

        # Create cell width vs latitude for Atlantic and Pacific basins
        qu1 = np.ones(lat.size)
        ec30to60 = mdt.EC_CellWidthVsLat(lat)
        rrs10to30 = mdt.RRS_CellWidthVsLat(lat, 30, 10)
        atl_nh = rrs10to30
        atl_vs_lat = mdt.mergeCellWidthVsLat(lat, ec30to60, atl_nh, 0, 6)
        pac_nh = mdt.mergeCellWidthVsLat(lat, 30 * qu1, rrs10to30, 50, 10)
        pac_vs_lat = mdt.mergeCellWidthVsLat(lat, ec30to60, pac_nh, 0, 6)

        # Expand from 1D to 2D
        _, atl_grid = np.meshgrid(lon, atl_vs_lat)
        _, pac_grid = np.meshgrid(lon, pac_vs_lat)

        # Signed distance of Atlantic region
        fc = read_feature_collection('Atlantic_region.geojson')
        signed_distance = signed_distance_from_geojson(
            fc, lon, lat, earth_radius, max_length=0.25)

        # Merge Atlantic and Pacific distributions smoothly
        transition_width = 500.0 * km
        mask_smooth = 0.5 * (1 + np.tanh(signed_distance / transition_width))
        cell_width_smooth = \
            pac_grid * mask_smooth + atl_grid * (1 - mask_smooth)

        # Merge Atlantic and Pacific distributions with step function
        mask_sharp = 0.5 * (1 + np.sign(signed_distance))
        cell_width_sharp = pac_grid * mask_sharp + atl_grid * (1 - mask_sharp)

        # Create a land mask that is 1 over land
        fc = read_feature_collection('Americas_land_mask.geojson')
        americas_land_mask = mask_from_geojson(fc, lon, lat)
        fc = read_feature_collection('Europe_Africa_land_mask.geojson')
        europe_africa_land_mask = mask_from_geojson(fc, lon, lat)
        land_mask = np.fmax(americas_land_mask, europe_africa_land_mask)

        # Merge: step transition over land, smooth transition over water
        cell_width = \
            cell_width_sharp * land_mask + cell_width_smooth * (1 - land_mask)

        ax = plt.subplot(4, 2, 1)
        ax.plot(lat, atl_vs_lat, label='Atlantic')
        ax.plot(lat, pac_vs_lat, label='Pacific')
        ax.grid(True)
        plt.title('Grid cell size [km] versus latitude')
        plt.legend()

        var_names = [
            'signed_distance',
            'mask_smooth',
            'cell_width_smooth',
            'mask_sharp',
            'cell_width_sharp',
            'land_mask',
            'cell_width']
        j = 2
        for var_name in var_names:
            _plot_cartopy(j, var_name, vars()[var_name], '3Wbgy5')
            j += 1
        fig.canvas.draw()
        plt.tight_layout()

        plt.savefig('mesh_construction.png')

        return cell_width, lon, lat


def _plot_cartopy(plot_number, var_name, var, map_name):
    ax = plt.subplot(4, 2, plot_number, projection=ccrs.PlateCarree())
    ax.set_global()
    im = ax.imshow(var,
                   origin='lower',
                   transform=ccrs.PlateCarree(),
                   extent=[-180, 180, -90, 90], cmap=map_name,
                   zorder=0)
    ax.add_feature(cfeature.LAND, edgecolor='black', zorder=1)
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1,
        color='gray',
        alpha=0.5,
        linestyle='-', zorder=2)
    ax.coastlines()
    gl.top_labels = False
    gl.bottom_labels = False
    gl.right_labels = False
    gl.left_labels = False
    plt.colorbar(im, shrink=.9)
    plt.title(var_name)

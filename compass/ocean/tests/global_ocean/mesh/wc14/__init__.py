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

from compass.mesh import QuasiUniformSphericalMeshStep


class WC14BaseMesh(QuasiUniformSphericalMeshStep):
    """
    A step for creating WC14 mesh
    """
    def setup(self):
        """
        Add some input files
        """

        inputs = ['coastline_CUSP.geojson',
                  'land_mask_Kamchatka.geojson',
                  'land_mask_Mexico.geojson',
                  'namelist.split_explicit',
                  'region_Arctic_Ocean.geojson',
                  'region_Bering_Sea.geojson',
                  'region_Bering_Sea_reduced.geojson',
                  'region_Central_America.geojson',
                  'region_Gulf_of_Mexico.geojson',
                  'region_Gulf_Stream_extension.geojson']
        for filename in inputs:
            self.add_input_file(filename=filename,
                                package=self.__module__)

        super().setup()

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
        fig = plt.figure()
        plt.clf()
        fig.set_size_inches(10.0, 14.0)
        register_sci_viz_colormaps()

        # Create cell width vs latitude for Atlantic and Pacific basins
        EC60to30 = mdt.EC_CellWidthVsLat(lat)
        EC60to30Narrow = mdt.EC_CellWidthVsLat(lat, latPosEq=8.0,
                                               latWidthEq=3.0)

        # Expand from 1D to 2D
        _, cellWidth = np.meshgrid(lon, EC60to30Narrow)
        _plot_cartopy(2, 'narrow EC60to30', cellWidth, '3Wbgy5')
        plotFrame = 3

        # global settings for regionally refines mesh
        highRes = 14.0  # [km]

        fileName = 'region_Central_America'
        transitionWidth = 800.0 * km
        transitionOffset = 0.0
        fc = read_feature_collection('{}.geojson'.format(fileName))
        signedDistance = signed_distance_from_geojson(fc, lon, lat,
                                                      earth_radius,
                                                      max_length=0.25)
        mask = 0.5 * (1 + np.tanh((transitionOffset - signedDistance) /
                                  (transitionWidth / 2.)))
        cellWidth = 30.0 * mask + cellWidth * (1 - mask)

        fileName = 'coastline_CUSP'
        distanceToTransition = 600.0 * km
        # transitionWidth is distance from 0.07 to 0.03 of transition within
        # tanh
        transitionWidth = 600.0 * km
        transitionOffset = distanceToTransition + transitionWidth / 2.0
        fc = read_feature_collection('{}.geojson'.format(fileName))
        signedDistance = signed_distance_from_geojson(fc, lon, lat,
                                                      earth_radius,
                                                      max_length=0.25)
        mask = 0.5 * (1 + np.tanh((transitionOffset - signedDistance) /
                                  (transitionWidth / 2.)))
        cellWidth = highRes * mask + cellWidth * (1 - mask)
        _plot_cartopy(plotFrame, fileName + ' mask', mask, 'Blues')
        _plot_cartopy(plotFrame + 1, 'cellWidth ', cellWidth, '3Wbgy5')
        plotFrame += 2

        fileName = 'region_Gulf_of_Mexico'
        transitionOffset = 600.0 * km
        transitionWidth = 600.0 * km
        fc = read_feature_collection('{}.geojson'.format(fileName))
        signedDistance = signed_distance_from_geojson(fc, lon, lat,
                                                      earth_radius,
                                                      max_length=0.25)
        maskSmooth = 0.5 * (1 + np.tanh((transitionOffset - signedDistance) /
                                        (transitionWidth / 2.)))
        maskSharp = 0.5 * (1 + np.sign(-signedDistance))
        fc = read_feature_collection('land_mask_Mexico.geojson')
        signedDistance = signed_distance_from_geojson(fc, lon, lat,
                                                      earth_radius,
                                                      max_length=0.25)
        landMask = 0.5 * (1 + np.sign(-signedDistance))
        mask = maskSharp * landMask + maskSmooth * (1 - landMask)
        cellWidth = highRes * mask + cellWidth * (1 - mask)
        _plot_cartopy(plotFrame, fileName + ' mask', mask, 'Blues')
        _plot_cartopy(plotFrame + 1, 'cellWidth ', cellWidth, '3Wbgy5')
        plotFrame += 2

        fileName = 'region_Bering_Sea'
        transitionOffset = 0.0 * km
        transitionWidth = 600.0 * km
        fc = read_feature_collection('{}.geojson'.format(fileName))
        signedDistance = signed_distance_from_geojson(fc, lon, lat,
                                                      earth_radius,
                                                      max_length=0.25)
        maskSmoothEast = 0.5 * (
                    1 + np.tanh((transitionOffset - signedDistance) /
                                (transitionWidth / 2.)))

        fc = read_feature_collection('region_Bering_Sea_reduced.geojson')
        signedDistance = signed_distance_from_geojson(fc, lon, lat,
                                                      earth_radius,
                                                      max_length=0.25)
        maskSmoothWest = 0.5 * (
                    1 + np.tanh((transitionOffset - signedDistance) /
                                (transitionWidth / 2.)))

        fc = read_feature_collection('land_mask_Kamchatka.geojson')
        maskWest = mask_from_geojson(fc, lon, lat)
        mask = maskSmoothWest * maskWest + maskSmoothEast * (1 - maskWest)
        cellWidth = highRes * mask + cellWidth * (1 - mask)
        _plot_cartopy(plotFrame, fileName + ' mask', mask, 'Blues')
        _plot_cartopy(plotFrame + 1, 'cellWidth ', cellWidth, '3Wbgy5')
        plotFrame += 2

        fileName = 'region_Arctic_Ocean'
        transitionOffset = 0.0 * km
        transitionWidth = 600.0 * km
        fc = read_feature_collection('{}.geojson'.format(fileName))
        signedDistance = signed_distance_from_geojson(fc, lon, lat,
                                                      earth_radius,
                                                      max_length=0.25)
        mask = 0.5 * (1 + np.tanh((transitionOffset - signedDistance) /
                                  (transitionWidth / 2.)))
        cellWidth = highRes * mask + cellWidth * (1 - mask)
        _plot_cartopy(plotFrame, fileName + ' mask', mask, 'Blues')
        _plot_cartopy(plotFrame + 1, 'cellWidth ', cellWidth, '3Wbgy5')
        plotFrame += 2

        fileName = 'region_Gulf_Stream_extension'
        transitionOffset = 0.0 * km
        transitionWidth = 600.0 * km
        fc = read_feature_collection('{}.geojson'.format(fileName))
        signedDistance = signed_distance_from_geojson(fc, lon, lat,
                                                      earth_radius,
                                                      max_length=0.25)
        mask = 0.5 * (1 + np.tanh((transitionOffset - signedDistance) /
                                  (transitionWidth / 2.)))
        cellWidth = highRes * mask + cellWidth * (1 - mask)
        _plot_cartopy(plotFrame, fileName + ' mask', mask, 'Blues')
        _plot_cartopy(plotFrame + 1, 'cellWidth ', cellWidth, '3Wbgy5')
        plotFrame += 2

        ax = plt.subplot(6, 2, 1)
        ax.plot(lat, EC60to30, label='original EC60to30')
        ax.plot(lat, EC60to30Narrow, label='narrow EC60to30')
        ax.grid(True)
        plt.title('Grid cell size [km] versus latitude')
        plt.legend(loc="upper left")

        plt.savefig('mesh_construction.png', dpi=300)

        return cellWidth, lon, lat


def _plot_cartopy(nPlot, varName, var, map_name):
    ax = plt.subplot(6, 2, nPlot, projection=ccrs.PlateCarree())
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
    plt.title(varName)

import mpas_tools.ocean.coastal_tools as ct
import numpy as np

from compass.ocean.mesh.floodplain import FloodplainMeshStep
from compass.ocean.tests.tides.mesh.vr45to5 import VRTidesMesh


class DEVR45to5rr1BaseMesh(VRTidesMesh, FloodplainMeshStep):
    """
    A step for creating DEVR45to5rr1BaseMesh meshes
    """
    def region_multiplier(self, xgrid):
        """
        Create cell width multiplier array for this mesh on a
        regular latitude-longitude grid

        Returns
        -------
        cellWidth : numpy.array
            m x n array of cell width in km

        lon : numpy.array
            longitude in degrees (length n and between -180 and 180)

        lat : numpy.array
            longitude in degrees (length m and between -90 and 90)
        """
        km = 1000.0

        params = ct.default_params
        params['ddeg'] = xgrid[1] - xgrid[0]

        # Background
        params["mesh_type"] = "QU"
        params["dx_max_global"] = 1.0
        params["region_box"] = ct.Atlantic
        params["restrict_box"] = ct.Atlantic_restrict
        params["plot_box"] = ct.Western_Atlantic
        params["dx_min_coastal"] = .99
        params["trans_width"] = 5000.0 * km
        params["trans_start"] = 500.0 * km

        multiplier, lon, lat = ct.coastal_refined_mesh(params)

        # Northeast refinement
        params["region_box"] = ct.Delaware_Bay
        params["plot_box"] = ct.Western_Atlantic
        params["dx_min_coastal"] = 0.5
        params["trans_width"] = 600.0 * km
        params["trans_start"] = 400.0 * km

        multiplier, lon, lat = ct.coastal_refined_mesh(
            params, multiplier, lon, lat)

        # Delaware regional refinement (1.25 km)
        params["region_box"] = ct.Delaware_Region
        params["plot_box"] = ct.Delaware
        params["dx_min_coastal"] = 0.25
        params["trans_width"] = 175.0 * km
        params["trans_start"] = 75.0 * km

        multiplier, lon, lat = ct.coastal_refined_mesh(
            params, multiplier, lon, lat)

        return multiplier, lon, lat

    def build_cell_width_lat_lon(self):

        cell_width, xgrid, ygrid = super().build_cell_width_lat_lon()
        print(cell_width.shape)

        multiplier, lon, lat = self.region_multiplier(xgrid)
        print(multiplier.shape)

        cell_width = np.multiply(multiplier, cell_width)

        return cell_width, lon, lat

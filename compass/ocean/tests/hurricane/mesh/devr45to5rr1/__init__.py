import mpas_tools.ocean.coastal_tools as ct
import numpy as np

from compass.ocean.mesh.floodplain import FloodplainMeshStep
from compass.ocean.tests.tides.mesh.vr45to5 import VRTidesMesh


class DEVR45to5rr1BaseMesh(VRTidesMesh, FloodplainMeshStep):
    """
    A step for creating DEQU120at30cr10rr2 meshes
    """
    def region_multiplier(self, xgrid):
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
        km = 1000.0

        params = ct.default_params
        params['ddeg'] = xgrid[1] - xgrid[0]

        # QU 120 background mesh and enhanced Atlantic (30km)
        params["mesh_type"] = "QU"
        params["dx_max_global"] = 1.0
        params["region_box"] = ct.Atlantic
        params["restrict_box"] = ct.Atlantic_restrict
        params["plot_box"] = ct.Western_Atlantic
        params["dx_min_coastal"] = .99
        params["trans_width"] = 5000.0 * km
        params["trans_start"] = 500.0 * km

        cell_width, lon, lat = ct.coastal_refined_mesh(params)

        # Northeast refinement (10km)
        params["region_box"] = ct.Delaware_Bay
        params["plot_box"] = ct.Western_Atlantic
        params["dx_min_coastal"] = 0.5
        params["trans_width"] = 600.0 * km
        params["trans_start"] = 400.0 * km

        cell_width, lon, lat = ct.coastal_refined_mesh(
            params, cell_width, lon, lat)

        # Delaware regional refinement (6km)
        params["region_box"] = ct.Delaware_Region
        params["plot_box"] = ct.Delaware
        params["dx_min_coastal"] = 0.25
        params["trans_width"] = 175.0 * km
        params["trans_start"] = 75.0 * km

        cell_width, lon, lat = ct.coastal_refined_mesh(
            params, cell_width, lon, lat)

        # Delaware Bay high-resolution (2km)
        Delaware_restrict = {"include": [np.array([[-75.853, 39.732],
                                                   [-74.939, 36.678],
                                                   [-71.519, 40.156],
                                                   [-74.784, 40.296]]),
                                         np.array([[-76.024, 37.188],
                                                   [-75.214, 36.756],
                                                   [-74.512, 37.925],
                                                   [-75.274, 38.318]])],
                             "exclude": []}
        params["region_box"] = ct.Delaware_Bay
        params["plot_box"] = ct.Delaware
        params["restrict_box"] = Delaware_restrict
        params["dx_min_coastal"] = 0.125
        params["trans_width"] = 100.0 * km
        params["trans_start"] = 17.0 * km

        cell_width, lon, lat = ct.coastal_refined_mesh(
            params, cell_width, lon, lat)

        return cell_width, lon, lat

    def build_cell_width_lat_lon(self):

        cell_width, xgrid, ygrid = super().build_cell_width_lat_lon()
        print(cell_width.shape)

        multiplier, lon, lat = self.region_multiplier(xgrid)
        print(multiplier.shape)

        cell_width = np.multiply(multiplier, cell_width)

        return cell_width, lon, lat

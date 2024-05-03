import mpas_tools.ocean.coastal_tools as ct

from compass.ocean.mesh.floodplain import FloodplainMeshStep


class DEQU120at30cr10rr2BaseMesh(FloodplainMeshStep):
    """
    A step for creating DEQU120at30cr10rr2 meshes
    """
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
        km = 1000.0

        params = ct.default_params

        # QU 120 background mesh and enhanced Atlantic (30km)
        params["mesh_type"] = "QU"
        params["dx_max_global"] = 120.0 * km
        params["region_box"] = ct.Atlantic
        params["restrict_box"] = ct.Atlantic_restrict
        params["plot_box"] = ct.Western_Atlantic
        params["dx_min_coastal"] = 30.0 * km
        params["trans_width"] = 5000.0 * km
        params["trans_start"] = 500.0 * km

        cell_width, lon, lat = ct.coastal_refined_mesh(params)

        # Northeast refinement (10km)
        params["region_box"] = ct.Delaware_Bay
        params["plot_box"] = ct.Western_Atlantic
        params["dx_min_coastal"] = 10.0 * km
        params["trans_width"] = 600.0 * km
        params["trans_start"] = 400.0 * km

        cell_width, lon, lat = ct.coastal_refined_mesh(
            params, cell_width, lon, lat)

        # Delaware regional refinement (6km)
        params["region_box"] = ct.Delaware_Region
        params["plot_box"] = ct.Delaware
        params["dx_min_coastal"] = 5.0 * km
        params["trans_width"] = 175.0 * km
        params["trans_start"] = 75.0 * km

        cell_width, lon, lat = ct.coastal_refined_mesh(
            params, cell_width, lon, lat)

        # Delaware Bay high-resolution (2km)
        params["region_box"] = ct.Delaware_Bay
        params["plot_box"] = ct.Delaware
        params["restrict_box"] = ct.Delaware_restrict
        params["dx_min_coastal"] = 2.0 * km
        params["trans_width"] = 100.0 * km
        params["trans_start"] = 17.0 * km

        cell_width, lon, lat = ct.coastal_refined_mesh(
            params, cell_width, lon, lat)

        return cell_width / 1000, lon, lat

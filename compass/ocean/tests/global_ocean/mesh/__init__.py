from compass.ocean.tests.global_ocean.mesh import qu240


def build_cell_width_lat_lon(mesh_name):
    """
    Create cell width array for this mesh on a regular latitude-longitude grid

    Parameters
    ----------
    mesh_name : str
        The name of the mesh

    Returns
    -------
    cellWidth : numpy.array
        m x n array of cell width in km

    lon : numpy.array
        longitude in degrees (length n and between -180 and 180)

    lat : numpy.array
        longitude in degrees (length m and between -90 and 90)
    """

    if mesh_name == 'QU240':
        return qu240.build_cell_width_lat_lon()
    else:
        raise ValueError('Unknown mesh name {}'.format(mesh_name))

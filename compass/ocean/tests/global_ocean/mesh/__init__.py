import sys

# make sure to add all meshes here so they will be found in sys.modules below
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

    package = get_mesh_package(mesh_name)
    build_cell_width = getattr(package, 'build_cell_width_lat_lon')
    return build_cell_width()


def get_mesh_package(mesh_name):
    """
    Get the system module corresponding to the given mesh name

    Parameters
    ----------
    mesh_name : str
        The name of the mesh

    Returns
    -------
    module : Package
        The system module for the given mesh, one of the packages in
        ``compass.ocean.tests.global_ocean.mesh`` with the mesh name converted
        to lowercase

    Raises
    ------
    ValueError
        If the corresponding module for the given mesh does not exist

    """
    package = 'compass.ocean.tests.global_ocean.mesh.{}'.format(
        mesh_name.lower())
    if package in sys.modules:
        package = sys.modules[package]
        return package
    else:
        raise ValueError('Mesh {} missing corresponding package {}'.format(
            mesh_name, package))


def mesh_package_has_file(package, filename):
    """

    Parameters
    ----------
    package : str or Package
        The package in which to check for the file

    filename : str
        The file to check for

    Returns
    -------
    has_file : bool
        Whether the file is in the package
    """
    package_contents = contents(package)
    return filename in package_contents

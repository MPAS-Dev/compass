import os

import xarray as xr
from mpas_tools.io import write_netcdf

from compass.step import Step

# the horizontal-mesh variables of an MPAS file
MESH_VARIABLES = [
    'latCell', 'lonCell', 'xCell', 'yCell', 'zCell', 'indexToCellID',
    'latEdge', 'lonEdge', 'xEdge', 'yEdge', 'zEdge', 'indexToEdgeID',
    'latVertex', 'lonVertex', 'xVertex', 'yVertex', 'zVertex',
    'indexToVertexID', 'cellsOnEdge', 'nEdgesOnCell', 'nEdgesOnEdge',
    'edgesOnCell', 'edgesOnEdge', 'weightsOnEdge', 'dvEdge', 'dcEdge',
    'angleEdge', 'areaCell', 'areaTriangle', 'cellsOnCell',
    'verticesOnCell', 'verticesOnEdge', 'edgesOnVertex', 'cellsOnVertex',
    'kiteAreasOnVertex', 'meshDensity',
]

MESH_ATTRIBUTES = [
    'on_a_sphere', 'sphere_radius', 'is_periodic', 'x_period', 'y_period',
    'mesh_spec',
]


class ExistingMeshStep(Step):
    """
    A step for extracting an existing, culled MPAS-Ocean horizontal mesh
    (e.g. from an initial condition used in E3SM) so that topography and a
    new initial condition can be created on exactly the same mesh.  The
    mesh takes the place of the base mesh in the
    :py:class:`compass.ocean.tests.global_ocean.mesh.Mesh` test case.
    """

    def __init__(self, test_case, name='base_mesh', subdir=None):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case this step belongs to

        name : str, optional
            the name of the step

        subdir : str, optional
            the subdirectory for the step
        """
        super().__init__(test_case, name=name, subdir=subdir)

    def setup(self):
        """
        Add the existing mesh as an input file and the base mesh as an
        output
        """
        super().setup()
        config = self.config
        mesh_filename = config.get('existing_mesh', 'mesh_filename')
        self.add_input_file(filename='existing_mesh.nc',
                            target=os.path.abspath(mesh_filename))
        base_filename = config.get('spherical_mesh', 'mpas_mesh_filename')
        self.add_output_file(filename=base_filename)

    def run(self):
        """
        Extract the horizontal mesh from the existing file
        """
        base_filename = self.config.get('spherical_mesh',
                                        'mpas_mesh_filename')
        with xr.open_dataset('existing_mesh.nc',
                             decode_times=False) as ds_in:
            ds_mesh = xr.Dataset()
            for var in MESH_VARIABLES:
                if var in ds_in:
                    ds_mesh[var] = ds_in[var]
            for attr in MESH_ATTRIBUTES:
                if attr in ds_in.attrs:
                    ds_mesh.attrs[attr] = ds_in.attrs[attr]
            write_netcdf(ds_mesh, base_filename)

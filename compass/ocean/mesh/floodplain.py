from mpas_tools.ocean import inject_bathymetry, inject_preserve_floodplain

from compass.mesh.spherical import QuasiUniformSphericalMeshStep


class FloodplainMeshStep(QuasiUniformSphericalMeshStep):
    """
    A step for creating a global MPAS-Ocean mesh that includes variables
    needed for preserving a floodplain

    preserve_floodplain : bool
        Whether the mesh includes land cells
    """

    def __init__(self, test_case, name='base_mesh', subdir=None,
                 cell_width=None, preserve_floodplain=True):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.testcase.TestCase
            The test case this step belongs to

        name : str
            the name of the step

        subdir : {str, None}
            the subdirectory for the step.  The default is ``name``

        cell_width : float, optional
            The approximate cell width in km of the mesh if constant resolution

        preserve_floodplain : bool, optional
            Whether the mesh includes land cells
        """

        super().__init__(test_case=test_case, name=name, subdir=subdir,
                         cell_width=cell_width)

        self.preserve_floodplain = preserve_floodplain

        self.add_input_file(filename='earth_relief_15s.nc',
                            target='SRTM15_plus_earth_relief_15s.nc',
                            database='bathymetry_database')

    def run(self):
        """
        Run this step of the test case
        """
        super().run()

        config = self.config

        mesh_filename = config.get('spherical_mesh', 'mpas_mesh_filename')

        inject_bathymetry(mesh_file=mesh_filename)
        if self.preserve_floodplain:
            floodplain_elevation = config.getfloat('spherical_mesh',
                                                   'floodplain_elevation')
            min_depth_outside_floodplain = config.getfloat(
                'spherical_mesh',
                'min_depth_outside_floodplain')
            if config.has_option('spherical_mesh', 'floodplain_resolution'):
                floodplain_resolution = config.getfloat(
                    'spherical_mesh',
                    'floodplain_resolution')
            else:
                floodplain_resolution = 1e10

            inject_preserve_floodplain(
                mesh_file=mesh_filename,
                floodplain_elevation=floodplain_elevation,
                floodplain_resolution=floodplain_resolution,
                min_depth_outside_floodplain=min_depth_outside_floodplain)

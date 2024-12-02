from compass.mesh.spherical import IcosahedralMeshStep
from compass.ocean.mesh.cull import CullMeshStep
from compass.ocean.mesh.remap_topography import RemapTopography
from compass.ocean.tests.tides.configure import configure_tides
from compass.ocean.tests.tides.dem import CreatePixelFile
from compass.ocean.tests.tides.mesh.vr45to5 import VRTidesMesh
from compass.testcase import TestCase


class Mesh(TestCase):
    """
    A test case for creating a global MPAS-Ocean mesh
    """
    def __init__(self, test_group, mesh_name):
        """
        Create test case for creating a global MPAS-Ocean mesh

        Parameters
        ----------
        test_group : compass.ocean.tests.tides.Tides
            The test group that this test case belongs to

        mesh_name : str
            The name of the mesh
        """
        self.mesh_name = mesh_name
        name = 'mesh'
        subdir = '{}/{}'.format(mesh_name, name)
        super().__init__(test_group=test_group, name=name, subdir=subdir)

        pixel_step = CreatePixelFile(self)
        self.add_step(pixel_step)

        name = 'base_mesh'
        if mesh_name == 'Icos7':
            base_mesh_step = IcosahedralMeshStep(
                self, name=name, subdivisions=7)
            mesh_lower = 'icos7'
        elif mesh_name == 'VR45to5':
            base_mesh_step = VRTidesMesh(
                self, pixel_step,
                name='base_mesh', subdir=None,
                elev_file='RTopo_2_0_4_GEBCO_v2023_30sec_pixel.nc',
                spac_dhdx=0.125, spac_hmin=5, spac_hmax=45, spac_hbar=45,
                ncell_nwav=80, ncell_nslp=4,
                filt_sdev=0.5, filt_halo=50, filt_plev=0.325)
            mesh_lower = 'vr45to5'
        else:
            raise ValueError(f'Unexpected mesh name {mesh_name}')

        self.package = f'compass.ocean.tests.tides.mesh.{mesh_lower}'
        self.mesh_config_filename = f'{mesh_lower}.cfg'

        self.add_step(base_mesh_step)

        remap_step = RemapTopography(test_case=self,
                                     base_mesh_step=base_mesh_step,
                                     mesh_name=mesh_name)
        self.add_step(remap_step)

        self.add_step(CullMeshStep(
            test_case=self, base_mesh_step=base_mesh_step,
            with_ice_shelf_cavities=True, remap_topography=remap_step))

    def configure(self):
        """
        Modify the configuration options for this test case
        """
        configure_tides(test_case=self, mesh=self)

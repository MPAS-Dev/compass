from compass.mesh.spherical import (
    IcosahedralMeshStep,
    QuasiUniformSphericalMeshStep,
)


class IcosMeshFromConfigStep(IcosahedralMeshStep):
    """
    A step for creating quasi-uniform meshes at a resolution given by a config
    option
    """
    def setup(self):
        """
        Set up the test case in the work directory, including downloading any
        dependencies.
        """
        # get the these properties from the config options
        super().setup()
        config = self.config
        self.cell_width = config.getfloat('global_ocean', 'qu_resolution')


class QUMeshFromConfigStep(QuasiUniformSphericalMeshStep):
    """
    A step for creating quasi-uniform meshes at a resolution given by a config
    option
    """
    def setup(self):
        """
        Set up the test case in the work directory, including downloading any
        dependencies.
        """
        # get the these properties from the config options
        super().setup()
        config = self.config
        self.cell_width = config.getfloat('global_ocean', 'qu_resolution')

from compass.config import CompassConfigParser
from compass.landice.tests.mismipplus.run_model import RunModel
from compass.landice.tests.mismipplus.setup_mesh import SetupMesh
from compass.testcase import TestCase


class SpinUp(TestCase):
    """
    Test case for creating the MISMIP+ mesh, initial conditions,
    input files, and runs a short segment of the spin up experiments
    """
    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.test.mismipplus
            The test group that this test case belongs to

        """
        name = "spin_up"

        super().__init__(test_group=test_group, name=name)

        config = CompassConfigParser()
        module = 'compass.landice.tests.mismipplus.spin_up'
        # add from config
        config.add_from_package(module, 'spin_up.cfg')
        resolution = int(config.getfloat('mesh', 'resolution'))

        resolution_key = f'{resolution:d}m'

        # Mesh generation step
        step_name = 'setup_mesh'
        self.add_step(SetupMesh(test_case=self,
                                name=f'{resolution_key}_{step_name}',
                                subdir=f'{resolution_key}/{step_name}',
                                resolution=resolution))

        # Simulation step
        step_name = 'run_model'
        step = RunModel(test_case=self,
                        name=f'{resolution_key}_{step_name}',
                        subdir=f'{resolution_key}/{step_name}',
                        resolution=resolution)

        # add the mesh file from the previous step as dependency
        step.mesh_file = 'landice_grid.nc'
        step.add_input_file(filename='landice_grid.nc',
                            target='../setup_mesh/landice_grid.nc')

        package = "compass.landice.tests.mismipplus.spin_up"
        # modify the namelist options and streams file
        step.add_streams_file(package, 'streams.spin_up')
        step.add_namelist_file(package, 'namelist.spin_up')
        # read the density value from config file and update the namelist
        ice_density = config['mesh'].getfloat('ice_density')
        step.add_namelist_options({'config_ice_density': f'{ice_density}'})

        self.add_step(step)

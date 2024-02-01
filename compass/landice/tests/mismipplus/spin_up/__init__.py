import os

from compass.config import CompassConfigParser
from compass.landice.tests import mismipplus
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

        # Mesh generation step
        step_name = 'setup_mesh'
        self.add_step(SetupMesh(test_case=self, name=step_name))

        # Simulation step
        step_name = 'run_model'
        step = RunModel(test_case=self, name=step_name)

        # add the mesh file from the previous step as dependency
        step.mesh_file = 'landice_grid.nc'
        step.add_input_file(filename='landice_grid.nc',
                            target='../setup_mesh/landice_grid.nc')

        package = "compass.landice.tests.mismipplus.spin_up"
        # modify the namelist options and streams file
        step.add_streams_file(package, 'streams.spin_up')
        step.add_namelist_file(package, 'namelist.spin_up')

        self.add_step(step)

    def configure(self):
        """
        Set up the directory structure, based on the requested resolution.
        """
        # get the config options from the TestCase, which
        config = self.config

        # get the resolution from the parsed config file(s)
        resolution = config.getfloat('mesh', 'resolution')

        # loop over the steps of the `TestCase` and create a consistent
        # directory structure based on the value of `resolution` at the time
        # of compass setup.
        for step_name, step in self.steps.items():

            # format resolution for creating subdirectory structure
            resolution_key = f'{resolution:4.0f}m'
            step.subdir = f'{resolution_key}/{step.name}'

            # set the path attribute, based on the subdir attribute set above.
            step.path = os.path.join(step.mpas_core.name,
                                     step.test_group.name,
                                     step.test_case.subdir,
                                     step.subdir)

            # NOTE: we do not set the `step.work_dir` attribute, since it
            # will be set by `compass setup`` by joining the work dir
            # provided through the command line interface and the
            # `step.path` set above.

            # store the resolution (at the time of `compass setup`) as an
            # attribute. This is needed to prevent the changing of resolution
            # between `compass setup` and `compas run`, which could result
            # in a mesh having a different resolution than the dir it sits in.
            step.resolution = resolution

            # read the density value from config file and update the namelist
            if step_name == "run_model":
                ice_density = config['mesh'].getfloat('ice_density')
                step.add_namelist_options(
                    {'config_ice_density': f'{ice_density}'})

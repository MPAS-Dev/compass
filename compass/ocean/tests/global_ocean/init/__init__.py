import os

from compass.ocean.tests.global_ocean.init.initial_state import InitialState
from compass.ocean.tests.global_ocean.init.remap_ice_shelf_melt import (
    RemapIceShelfMelt,
)
from compass.ocean.tests.global_ocean.init.ssh_adjustment import SshAdjustment
from compass.ocean.tests.global_ocean.init.ssh_from_surface_density import (
    SshFromSurfaceDensity,
)
from compass.testcase import TestCase
from compass.validate import compare_variables


class Init(TestCase):
    """
    A test case for creating initial conditions on a global MPAS-Ocean mesh

    Attributes
    ----------
    mesh : compass.ocean.tests.global_ocean.mesh.Mesh
        The test case that creates the mesh used by this test case

    initial_condition : {'WOA23', 'PHC', 'EN4_1900'}
        The initial condition dataset to use

    init_subdir : str
        The subdirectory within the test group for all test cases with this
        initial condition
    """
    def __init__(self, test_group, mesh, initial_condition):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.global_ocean.GlobalOcean
            The global ocean test group that this test case belongs to

        mesh : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case that creates the mesh used by this test case

        initial_condition : {'WOA23', 'PHC', 'EN4_1900'}
            The initial condition dataset to use
        """
        name = 'init'
        ic_dir = initial_condition
        self.init_subdir = os.path.join(mesh.mesh_subdir, ic_dir)
        subdir = os.path.join(self.init_subdir, name)
        super().__init__(test_group=test_group, name=name, subdir=subdir)

        self.mesh = mesh
        self.initial_condition = initial_condition

    def configure(self, config=None):
        """
        Modify the configuration options for this test case

        config : compass.config.CompassConfigParser, optional
            Configuration options to update if not those for this test case
        """
        add_steps = config is None
        if config is None:
            config = self.config

        mesh = self.mesh

        # set mesh-relate config options
        mesh.configure(config=config)

        initial_condition = self.initial_condition
        descriptions = {'WOA23': 'World Ocean Atlas 2023 climatology '
                                 '1991-2020',
                        'PHC': 'Polar science center Hydrographic '
                               'Climatology (PHC)',
                        'EN4_1900': "Met Office Hadley Centre's EN4 dataset "
                                    "from 1900"}
        config.set('global_ocean', 'init_description',
                   descriptions[initial_condition])

        if add_steps:
            # add the steps for ssh adjustment
            if mesh.with_ice_shelf_cavities:
                step_index = 1
                name = \
                    f'{step_index:02d}_init_with_draft_from_constant_density'
                subdir = f'adjust_ssh/{name}'
                init_const_rho = InitialState(
                    test_case=self, mesh=mesh,
                    initial_condition=initial_condition,
                    name=name, subdir=subdir,
                    adjustment_fraction=0.)
                self.add_step(init_const_rho)

                # Recompute ssh using surface density
                step_index += 1
                name = f'{step_index:02d}_ssh_from_surface_density'
                subdir = f'adjust_ssh/{name}'
                ssh_from_surf_rho = SshFromSurfaceDensity(
                    test_case=self, init_path=init_const_rho.path,
                    name=name, subdir=subdir)
                self.add_step(ssh_from_surf_rho)

                culled_topo_path = ssh_from_surf_rho.path

                iteration_count = config.getint('ssh_adjustment', 'iterations')
                for iter_index in range(iteration_count):
                    fraction = iter_index / iteration_count

                    step_index += 1
                    name = f'{step_index:02d}_init'
                    subdir = f'adjust_ssh/{name}'
                    init_step = InitialState(
                        test_case=self, mesh=mesh,
                        initial_condition=initial_condition,
                        culled_topo_path=culled_topo_path,
                        name=name, subdir=subdir,
                        adjustment_fraction=fraction)
                    self.add_step(init_step)

                    step_index += 1
                    name = f'{step_index:02d}_adjust_ssh'
                    subdir = f'adjust_ssh/{name}'
                    adjust_ssh = SshAdjustment(
                        test_case=self, init_path=init_step.path,
                        name=name, subdir=subdir)
                    self.add_step(adjust_ssh)
                    culled_topo_path = adjust_ssh.path

                name = 'initial_state'
                subdir = 'initial_state'
                init_step = InitialState(
                    test_case=self, mesh=mesh,
                    initial_condition=initial_condition,
                    culled_topo_path=culled_topo_path,
                    name=name, subdir=subdir,
                    adjustment_fraction=1.0)
                self.add_step(init_step)

                self.add_step(RemapIceShelfMelt(test_case=self, mesh=mesh))
            else:
                self.add_step(
                    InitialState(
                        test_case=self, mesh=mesh,
                        initial_condition=initial_condition))

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        variables = ['temperature', 'salinity', 'layerThickness']
        compare_variables(test_case=self, variables=variables,
                          filename1='initial_state/initial_state.nc')

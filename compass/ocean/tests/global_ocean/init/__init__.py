import os

from compass.ocean.tests.global_ocean.init.initial_state import InitialState
from compass.ocean.tests.global_ocean.init.remap_ice_shelf_melt import (
    RemapIceShelfMelt,
)
from compass.ocean.tests.global_ocean.init.ssh_adjustment import SshAdjustment
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

    inactive_top_comp_subdir : str
        If ``with_inactive_top_cells == True``, the subdirectory equivalent to
        ``init_subdir`` for test cases without inactive top cells for use for
        validation between tests with and without inactive top cells
    """
    def __init__(self, test_group, mesh, initial_condition,
                 with_inactive_top_cells=False):
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
        mesh_name = mesh.mesh_name
        ic_dir = initial_condition
        init_subdir = os.path.join(mesh_name, ic_dir)
        if with_inactive_top_cells:
            # Add the name of the subdir without inactive top cells whether or
            # not is has or will be run
            self.inactive_top_comp_subdir = init_subdir
            init_subdir = os.path.join(init_subdir, inactive_top)
        self.init_subdir = init_subdir
        subdir = os.path.join(self.init_subdir, name)
        super().__init__(test_group=test_group, name=name, subdir=subdir)

        self.mesh = mesh
        self.initial_condition = initial_condition
        self.with_inactive_top_cells = with_inactive_top_cells

        self.add_step(
            InitialState(
                test_case=self, mesh=mesh,
                initial_condition=initial_condition, 
                with_inactive_top_cells=with_inactive_top_cells))

        if mesh.with_ice_shelf_cavities:
            self.add_step(RemapIceShelfMelt(test_case=self, mesh=mesh))

            self.add_step(
                SshAdjustment(test_case=self))

    def configure(self, config=None):
        """
        Modify the configuration options for this test case

        config : compass.config.CompassConfigParser, optional
            Configuration options to update if not those for this test case
        """
        if config is None:
            config = self.config

        # set mesh-relate config options
        self.mesh.configure(config=config)

        initial_condition = self.initial_condition
        descriptions = {'WOA23': 'World Ocean Atlas 2023 climatology '
                                 '1991-2020',
                        'PHC': 'Polar science center Hydrographic '
                               'Climatology (PHC)',
                        'EN4_1900': "Met Office Hadley Centre's EN4 dataset "
                                    "from 1900"}
        config.set('global_ocean', 'init_description',
                   descriptions[initial_condition])

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        variables = ['temperature', 'salinity', 'layerThickness']
        compare_variables(test_case=self, variables=variables,
                          filename1='initial_state/initial_state.nc')

        if self.with_inactive_top_cells:
            # construct the work directory for the other test
            filename2 = os.path.join(self.base_work_dir, self.mpas_core.name,
                                     self.test_group.name,
                                     self.inactive_top_comp_subdir,
                                     'init/initial_state/initial_state.nc')
            if os.path.exists(filename2):
                variables = ['temperature', 'salinity', 'layerThickness',
                             'normalVelocity']
                compare_variables(test_case=self, variables=variables,
                                  filename1='initial_state/initial_state_crop.nc'
                                  filename2=filename2,
                                  quiet=False, check_outputs=False,
                                  skip_if_step_not_run=False)

            else:
                self.logger.warn('The version of "init" without inactive top '
                                 'cells was not run.  Skipping\n'
                                 'validation.')

        if self.mesh.with_ice_shelf_cavities:
            variables = ['ssh', 'landIcePressure']
            compare_variables(test_case=self, variables=variables,
                              filename1='ssh_adjustment/adjusted_init.nc')

import os

from compass.testcase import TestCase
from compass.ocean.tests.global_ocean.init.initial_state import InitialState
from compass.ocean.tests.global_ocean.init.ssh_adjustment import SshAdjustment
from compass.validate import compare_variables


class Init(TestCase):
    """
    A test case for creating initial conditions on a global MPAS-Ocean mesh

    Attributes
    ----------
    mesh : compass.ocean.tests.global_ocean.mesh.Mesh
        The test case that creates the mesh used by this test case

    initial_condition : {'PHC', 'EN4_1900'}
        The initial condition dataset to use

    with_bgc : bool
        Whether to include biogeochemistry (BGC) in the initial condition

    init_subdir : str
        The subdirectory within the test group for all test cases with this
        initial condition
    """
    def __init__(self, test_group, mesh, initial_condition, with_bgc):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.global_ocean.GlobalOcean
            The global ocean test group that this test case belongs to

        mesh : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case that creates the mesh used by this test case

        initial_condition : {'PHC', 'EN4_1900'}
            The initial condition dataset to use

        with_bgc : bool
            Whether to include biogeochemistry (BGC) in the initial condition
        """
        name = 'init'
        mesh_name = mesh.mesh_name
        if with_bgc:
            ic_dir = '{}_BGC'.format(initial_condition)
        else:
            ic_dir = initial_condition
        self.init_subdir = os.path.join(mesh_name, ic_dir)
        subdir = os.path.join(self.init_subdir, name)
        super().__init__(test_group=test_group, name=name, subdir=subdir)

        self.mesh = mesh
        self.initial_condition = initial_condition
        self.with_bgc = with_bgc

        self.add_step(
            InitialState(
                test_case=self, mesh=mesh,
                initial_condition=initial_condition, with_bgc=with_bgc))

        if mesh.with_ice_shelf_cavities:
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
        descriptions = {'PHC': 'Polar science center Hydrographic '
                               'Climatology (PHC)',
                        'EN4_1900':
                            "Met Office Hadley Centre's EN4 dataset from 1900"}
        config.set('global_ocean', 'init_description',
                   descriptions[initial_condition])

        if self.with_bgc:
            # todo: this needs to be filled in!
            config.set('global_ocean', 'bgc_description',
                       '<<<Missing>>>')

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        variables = ['temperature', 'salinity', 'layerThickness']
        compare_variables(test_case=self, variables=variables,
                          filename1='initial_state/initial_state.nc')

        if self.with_bgc:
            variables = [
                'temperature', 'salinity', 'layerThickness', 'PO4', 'NO3',
                'SiO3', 'NH4', 'Fe', 'O2', 'DIC', 'DIC_ALT_CO2', 'ALK',
                'DOC', 'DON', 'DOFe', 'DOP', 'DOPr', 'DONr', 'zooC',
                'spChl', 'spC', 'spFe', 'spCaCO3', 'diatChl', 'diatC',
                'diatFe', 'diatSi', 'diazChl', 'diazC', 'diazFe',
                'phaeoChl', 'phaeoC', 'phaeoFe', 'DMS', 'DMSP', 'PROT',
                'POLY', 'LIP']
            compare_variables(test_case=self, variables=variables,
                              filename1='initial_state/initial_state.nc')

        if self.mesh.with_ice_shelf_cavities:
            variables = ['ssh', 'landIcePressure']
            compare_variables(test_case=self, variables=variables,
                              filename1='ssh_adjustment/adjusted_init.nc')
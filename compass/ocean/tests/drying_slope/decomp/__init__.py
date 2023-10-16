from compass.ocean.tests import drying_slope
from compass.ocean.tests.drying_slope.forward import Forward
from compass.ocean.tests.drying_slope.initial_state import InitialState
from compass.testcase import TestCase
from compass.validate import compare_variables


class Decomp(TestCase):
    """
    A decomposition test case for the baroclinic channel test group, which
    makes sure the model produces identical results on 1 and 12 cores.

    Attributes
    ----------
    resolution : str
        The resolution of the test case
    """

    def __init__(self, test_group, resolution, coord_type):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.baroclinic_channel.BaroclinicChannel
            The test group that this test case belongs to

        resolution : str
            The resolution of the test case
        """
        name = 'decomp'
        self.resolution = resolution
        self.coord_type = coord_type
        if resolution < 1.:
            res_name = f'{int(resolution*1e3)}m'
        else:
            res_name = f'{int(resolution)}km'
        subdir = f'{res_name}/{coord_type}/{name}'
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        self.add_step(InitialState(test_case=self, coord_type=coord_type))
        if coord_type == 'single_layer':
            damping_coeff = None
        else:
            damping_coeff = 0.01
        for procs in [1, 12]:
            name = '{}proc'.format(procs)
            forward_step = Forward(test_case=self, name=name, subdir=name,
                                   resolution=resolution,
                                   ntasks=procs, openmp_threads=1,
                                   damping_coeff=damping_coeff,
                                   coord_type=coord_type)
            self.add_step(forward_step)

    def configure(self):
        """
        Modify the configuration options for this test case.
        """

        resolution = self.resolution
        config = self.config
        ny = round(28 / resolution)
        if resolution < 1.:
            ny += 2
        dc = 1e3 * resolution

        config.set('drying_slope', 'ny', f'{ny}', comment='the number of '
                   'mesh cells in the y direction')
        config.set('drying_slope', 'dc', f'{dc}', comment='the distance '
                   'between adjacent cell centers')

    # no run() method is needed

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        variables = ['temperature', 'salinity', 'layerThickness',
                     'normalVelocity']
        compare_variables(test_case=self, variables=variables,
                          filename1='1proc/output.nc',
                          filename2='12proc/output.nc')

from compass.testcase import TestCase

from compass.ocean.tests.spherical_harmonic_transform.qu_convergence.mesh \
     import Mesh
from compass.ocean.tests.spherical_harmonic_transform.qu_convergence.init \
     import Init
from compass.ocean.tests.spherical_harmonic_transform.qu_convergence.analysis \
     import Analysis


class QuConvergence(TestCase):
    """
    A test case for creating a global MPAS-Ocean mesh
    Attributes
    ----------
    resolutions : list of int
    """
    def __init__(self, test_group):
        """
        Create test case for creating a global MPAS-Ocean mesh
        Parameters
        ----------
        test_group : compass.ocean.tests.spherical_harmonic_transform.SphericalHarmonicTransform
            The global ocean test group that this test case belongs to
        """
        super().__init__(test_group=test_group, name='qu_convergence')
        self.resolutions = None

    def configure(self):
        """
        Set config options for the test case
        """
        config = self.config
        resolutions = config.get('qu_convergence', 'resolutions')
        resolutions = [int(resolution) for resolution in
                       resolutions.replace(',', ' ').split()]

        self.resolutions = resolutions

        serial_nLat = config.get('qu_convergence', 'serial_nLat')
        serial_nLat = [int(nLat) for nLat in
                       serial_nLat.replace(',', ' ').split()]

        parallel_N = config.get('qu_convergence', 'parallel_N')
        parallel_N = [int(N) for N in
                      parallel_N.replace(',', ' ').split()]

        for resolution in resolutions:
            self.add_step(Mesh(test_case=self,
                               resolution=resolution,
                               serial_nLat=serial_nLat))

            for N in parallel_N:

                self.add_step(Init(test_case=self,
                                   resolution=resolution,
                                   algorithm='parallel',
                                   order=N))

            for nLat in serial_nLat:

                self.add_step(Init(test_case=self,
                                   resolution=resolution,
                                   algorithm='serial',
                                   order=nLat))

        self.add_step(Analysis(test_case=self,
                               resolutions=resolutions,
                               parallel_N=parallel_N,
                               serial_nLat=serial_nLat))

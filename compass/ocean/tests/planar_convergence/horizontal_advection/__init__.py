from compass.ocean.tests.planar_convergence.conv_test_case import ConvTestCase
from compass.ocean.tests.planar_convergence.horizontal_advection.init import \
    Init

from compass.ocean.tests.planar_convergence.horizontal_advection.analysis \
    import Analysis


class HorizontalAdvection(ConvTestCase):
    """
    A test case for testing horizontal advection in MPAS-Ocean with planar,
    doubly periodic meshes
    """
    def __init__(self, test_group):
        """
        Create test case for creating a global MPAS-Ocean mesh

        Parameters
        ----------
        test_group : compass.ocean.tests.cosine_bell.GlobalOcean
            The global ocean test group that this test case belongs to
        """
        super().__init__(test_group=test_group, name='horizontal_advection')

        self.add_step(Analysis(test_case=self, resolutions=self.resolutions))

    def create_init(self, resolution):
        """

        Child class must override this to return an instance of a
        ConvInit step

        Parameters
        ----------
        resolution : int
            The resolution of the test case

        Returns
        -------
        init : compass.ocean.tests.planar_convergence.conv_init.ConvInit
            The init step object
        """

        return Init(test_case=self, resolution=resolution)

    def create_analysis(self, resolutions):
        """

        Child class must override this to return an instance of a
        ConvergenceInit step

        Parameters
        ----------
        resolutions : list of int
            The resolutions of the other steps in the test case

        Returns
        -------
        analysis : compass.ocean.tests.planar_convergence.conv_analysis.ConvAnalysis
            The init step object
        """

        return Analysis(test_case=self, resolutions=resolutions)

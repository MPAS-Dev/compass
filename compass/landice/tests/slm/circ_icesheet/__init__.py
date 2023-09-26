from compass.landice.tests.slm.circ_icesheet.run_model import RunModel
from compass.landice.tests.slm.circ_icesheet.setup_mesh import SetupMesh
from compass.landice.tests.slm.circ_icesheet.visualize import Visualize
from compass.testcase import TestCase


class CircIcesheetTest(TestCase):
    """
    This test generates an idealized, circular ice sheet that has a
    prescribed thickness evolution for testing coupling between MALI
    and the Sea-Level Model.

    Attributes
    ----------
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.dome.Dome
            The test group that this test case belongs to
            The resolution or type of mesh of the test case
        """
        name = 'circular_icesheet_test'
        subdir = name
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        self.add_step(SetupMesh(test_case=self))

        self.add_step(RunModel(test_case=self, ntasks=4, openmp_threads=1,
                      name='run_step'))

        step = Visualize(test_case=self)
        self.add_step(step, run_by_default=True)

    # no configure() method is needed because we will use the default
    # config options

    # no run() method is needed because we're doing the default: running all
    # steps

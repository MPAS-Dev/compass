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

        name : str, optional
            the name of the test case
        """
        name = 'circular_icesheet_test'
        subdir = name
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

    def configure(self):
        """
        Set up the desired mesh-resolution tests

        Read the list of resolutions from the config
        """
        config = self.config
        section = config['circ_icesheet']
        mali_res = section.get('mali_res').split(',')

        section = config['slm']
        slm_nglv = section.get('slm_nglv').split(',')
        print(f'list of MALI-mesh resolution is {mali_res} km.')
        print(f'list of SLM Gauss-Legendre latitudinal points is {slm_nglv}.')

        for res in mali_res:
            for nglv in slm_nglv:
                self.add_step(SetupMesh(test_case=self,
                              name=f'mali{res}km_slm{nglv}/setup_mesh'))
                self.add_step(RunModel(test_case=self, ntasks=1,
                              openmp_threads=1,
                              name=f'mali{res}km_slm{nglv}/run_model'))
        step = Visualize(test_case=self)
        self.add_step(step, run_by_default=True)

    # no run() method is needed because we're doing the default: running all
    # steps

from compass.landice.tests.slm_circ_icesheet.run_model import RunModel
from compass.landice.tests.slm_circ_icesheet.setup_mesh import SetupMesh
from compass.landice.tests.slm_circ_icesheet.visualize import Visualize
from compass.testcase import TestCase


class MeshConvergenceTest(TestCase):
    """
    This test uses the circular ice sheet test case to test the convergence
    of the coupled MALI-SLM system with respect to mesh resolution of both
    MALI and the SLM.
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.slm_circ_icesheet.SlmCircIcesheet
            The test group that this test case belongs to
            The resolution or type of mesh of the test case
        """
        name = 'mesh_convergence'
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
                                        name=f'mali{res}km_slm{nglv}/'
                                        'setup_mesh', res=res, nglv=nglv))
                if (int(res) <= 16 and int(res) > 2):
                    ntasks = 256
                elif (int(res) <= 2):
                    ntasks = 512
                else:
                    ntasks = 128
                min_tasks = 10
                self.add_step(RunModel(test_case=self, res=res, nglv=nglv,
                                       ntasks=ntasks, min_tasks=min_tasks,
                                       openmp_threads=1,
                                       name=f'mali{res}km_slm{nglv}'
                                       '/run_model'))
        step = Visualize(test_case=self)
        self.add_step(step, run_by_default=True)

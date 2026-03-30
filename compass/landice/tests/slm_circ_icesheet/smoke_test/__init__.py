from compass.landice.tests.slm_circ_icesheet.run_model import RunModel
from compass.landice.tests.slm_circ_icesheet.setup_mesh import SetupMesh
from compass.landice.tests.slm_circ_icesheet.visualize import Visualize
from compass.testcase import TestCase


class SmokeTest(TestCase):
    """
    A lightweight smoke test using the baseline circ_icesheet configuration.
    """
    def __init__(self, test_group):
        name = 'smoke_test'
        subdir = name
        super().__init__(test_group=test_group, name=name, subdir=subdir)

    def configure(self):
        config = self.config
        section = config['circ_icesheet']
        mali_res = section.getint('mali_res')

        section = config['slm']
        slm_nglv = section.getint('slm_nglv')

        res = mali_res
        nglv = slm_nglv

        # setup mesh and run model using resources from circ_icesheet package
        self.add_step(
            SetupMesh(
                test_case=self,
                name='setup_mesh',
                res=res,
                nglv=nglv,
            )
        )

        if (int(res) <= 16 and int(res) > 2):
            ntasks = 256
        elif (int(res) <= 2):
            ntasks = 512
        else:
            ntasks = 128
        min_tasks = 10

        resource_package = (
            'compass.landice.tests.slm_circ_icesheet.smoke_test'
        )
        self.add_step(
            RunModel(
                test_case=self,
                res=res,
                nglv=nglv,
                resource_package=resource_package,
                ntasks=ntasks,
                min_tasks=min_tasks,
                openmp_threads=1,
                name='run_model',
            )
        )

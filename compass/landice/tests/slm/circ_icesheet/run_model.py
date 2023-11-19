import os
from importlib import resources

from jinja2 import Template

from compass.model import run_model
from compass.step import Step


class RunModel(Step):
    """
    A step for performing forward MALI runs as part of dome test cases.

    Attributes
    ----------
    """
    def __init__(self, test_case, res, nglv, ntasks, name='run_model',
                 subdir=None, min_tasks=None, openmp_threads=1):
        """
        Create a new test case

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        res : str
            Resolution of the MALI domain

        nglv : str
            Number of Gauss-Legendre nodes in latitude in the SLM grid

        ntasks : int
            the number of tasks the step would ideally use.  If fewer tasks
            are available on the system, the step will run on all available
            tasks as long as this is not below ``min_tasks``

        name : str, optional
            the name of the test case

        subdir : str, optional
            the subdirectory for the step.  The default is ``name``

        min_tasks : int, optional
            the number of tasks the step requires.  If the system has fewer
            than this number of tasks, the step will fail

        openmp_threads : int, optional
            the number of OpenMP threads the step will use
        """
        self.res = res
        self.nglv = nglv

        if min_tasks is None:
            min_tasks = ntasks
        super().__init__(test_case=test_case, name=name, subdir=subdir,
                         ntasks=ntasks, min_tasks=min_tasks,
                         openmp_threads=openmp_threads)

        self.add_namelist_file(
            'compass.landice.tests.slm.circ_icesheet', 'namelist.landice',
            out_name='namelist.landice')

        self.add_streams_file(
            'compass.landice.tests.slm.circ_icesheet', 'streams.landice',
            out_name='streams.landice')

        self.add_input_file(filename='landice_grid.nc',
                            target='../setup_mesh/landice_grid.nc')
        self.add_input_file(filename='graph.info',
                            target='../setup_mesh/graph.info')
        self.add_input_file(filename='smb_forcing.nc',
                            target='../setup_mesh/smb_forcing.nc')
        self.add_input_file(filename='mapping_file_mali_to_slm.nc',
                            target='../setup_mesh/'
                                   'mapping_file_mali_to_slm.nc')
        self.add_input_file(filename='mapping_file_slm_to_mali.nc',
                            target='../setup_mesh/'
                                   'mapping_file_slm_to_mali.nc')
        self.add_model_as_input()

        self.add_output_file(filename='output.nc')

    def setup(self):
        os.makedirs(os.path.join(self.work_dir, 'OUTPUT_SLM/'),
                    exist_ok='True')
        os.makedirs(os.path.join(self.work_dir, 'ICELOAD_SLM/'),
                    exist_ok='True')

        # change the sealevel namelist
        template = Template(resources.read_text
                            ('compass.landice.tests.slm',
                             'namelist.sealevel.template'))
        text = template.render(nglv=self.nglv)

        # write out the namelise.sealevel file
        file_slm_nl = os.path.join(self.work_dir, 'namelist.sealevel')
        with open(file_slm_nl, 'w') as handle:
            handle.write(text)

    def run(self):
        """
        Run this step of the test case
        """
        run_model(step=self)

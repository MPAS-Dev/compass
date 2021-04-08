import os
import stat
from jinja2 import Template
from importlib import resources


class Step:
    """
    The base class for a step of a test cases, such as setting up a mesh,
    creating an initial condition, or running the MPAS core forward in time.
    The step is the smallest unit of work in compass that can be run on its
    own by a user, though users will typically run full test cases or test
    suites.

    Attributes
    ----------
    name : str
        the name of the test case
    """

    def __init__(self, name, subdir=None):
        """
        Create a new test case

        Parameters
        ----------
        name : str
            the name of the test case

        subdir : str, optional
            the subdirectory for the step.  The default is ``name``
        """
        self.name = name
        self.mpas_core = None
        self.test_group = None
        self.test_case = None

        self.cores = 1
        self.min_cores = 1
        self.max_memory = 1000
        self.max_disk = 1000

        self.inputs = list()
        self.outputs = list()

        self.config = None
        self.config_filename = None

        self.logger = None
        self.log_filename = None

        if subdir is not None:
            self.subdir = subdir
        else:
            self.subdir = name

        self.path = None
        self.work_dir = None
        self.base_work_dir = None

    def setup(self):
        """
        Set up the test case in the work directory, including downloading any
        dependencies.  The step should override this function to perform setup
        operations such as generating namelist and streams files, adding inputs
        and outputs.
        """
        pass

    def run(self):
        """
        Run the step.  The step should override this function to perform the
        main work.
        """
        pass

    def generate(self):
        """
        Generate a ``run.py`` script for the test case or step.
        """

        template = Template(
            resources.read_text('compass.step', 'step.template'))
        test_case = {'name': self.test_case.name}
        step = {'name': self.name,
                'config_filename': self.config_filename}
        work_dir = self.work_dir
        script = template.render(test_case=test_case, step=step)

        run_filename = os.path.join(work_dir, 'run.py')
        with open(run_filename, 'w') as handle:
            handle.write(script)

        # make sure it has execute permission
        st = os.stat(run_filename)
        os.chmod(run_filename, st.st_mode | stat.S_IEXEC)

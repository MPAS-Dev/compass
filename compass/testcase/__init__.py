import os
import stat
from jinja2 import Template
from importlib import resources

from mpas_tools.logging import LoggingContext
from compass.parallel import get_available_cores_and_nodes


class TestCase:
    """
    The base class for test cases---such as a decomposition, threading or
    restart test---that are made up of one or more steps

    Attributes
    ----------
    name : str
        the name of the test case

    test_group : compass.TestGroup
        The test group the test case belongs to

    mpas_core : compass.MpasCore
        The MPAS core the test group belongs to

    steps : dict
        A dictionary of steps in the test case with step names as keys

    steps_to_run : list
        A list of the steps to run when ``run()`` gets called.  This list
        includes all steps by default but can be replaced with a list of only
        those tests that should run by default if some steps are optional and
        should be run manually by the user.

    subdir : str
        the subdirectory for the test case

    path : str
        the path within the base work directory of the test case, made up of
        ``mpas_core``, ``test_group``, and the test case's ``subdir``

    config : configparser.ConfigParser
        Configuration options for this test case, a combination of the defaults
        for the machine, core and configuration

    config_filename : str
        The local name of the config file that ``config`` has been written to
        during setup and read from during run

    work_dir : str
        The test case's work directory, defined during setup as the combination
        of ``base_work_dir`` and ``path``

    base_work_dir : str
        The base work directory

    logger : logging.Logger
        A logger for output from the test case

    log_filename : str
        At run time, the name of a log file where output/errors from the test
        case are being logged, or ``None`` if output is to stdout/stderr

    new_step_log_file : bool
        Whether to create a new log file for each step or to log output to a
        common log file for the whole test case.  The latter is used when
        running the test case as part of a test suite
    """

    def __init__(self, test_group, name, subdir=None):
        """
        Create a new test case

        Parameters
        ----------
        test_group : compass.TestGroup
            the test group that this test case belongs to

        name : str
            the name of the test case

        subdir : str, optional
            the subdirectory for the test case.  The default is ``name``
        """
        self.name = name
        self.mpas_core = test_group.mpas_core
        self.test_group = test_group
        if subdir is not None:
            self.subdir = subdir
        else:
            self.subdir = name
        test_group.add_test_case(self)

        self.path = os.path.join(self.mpas_core.name, test_group.name,
                                 self.subdir)

        # steps will be added by calling add_step()
        self.steps = dict()
        self.steps_to_run = list()

        # these will be set during setup
        self.config = None
        self.config_filename = None
        self.work_dir = None
        self.base_work_dir = None

        # these will be set when running the test case
        self.new_step_log_file = True
        self.logger = None
        self.log_filename = None

    def configure(self):
        """
        Modify the configuration options for this test case. Test cases should
        override this method if they want to add config options specific to
        the test case, e.g. from a config file stored in the test case's python
        package
        """
        pass

    def run(self):
        """
        Run each step of the test case.  Test cases can override this method
        to perform additional operations in addition to running the test case's
        steps

        """
        logger = self.logger
        cwd = os.getcwd()
        for step_name in self.steps_to_run:
            step = self.steps[step_name]
            step.config = self.config
            new_log_file = self.new_step_log_file
            if self.log_filename is not None:
                step.log_filename = self.log_filename
                do_local_logging = True
            else:
                # We only want to do local log output if the step output is
                # being redirected to a file.  Otherwise, we assume we're
                # probably just running one step and the local logging is
                # redundant and unnecessary
                do_local_logging = new_log_file

            if do_local_logging:
                logger.info(' * Running {}'.format(step_name))
            try:
                self._run_step(step, new_log_file)
            except BaseException:
                if do_local_logging:
                    logger.info('     Failed')
                raise

            if do_local_logging:
                logger.info('     Complete')

            os.chdir(cwd)

    def add_step(self, step, run_by_default=True):
        """
        Add a step to the test case

        Parameters
        ----------
        step : compass.Step
            The step to add

        run_by_default : bool, optional
            Whether to add this step to the list of steps to run when the
            ``run()`` method gets called.  If ``run_by_default=False``, users
            would need to run this step manually.
        """
        self.steps[step.name] = step
        if run_by_default:
            self.steps_to_run.append(step.name)

    def generate(self):
        """
        Generate a ``run.py`` script for the test case or step.
        """

        template = Template(
            resources.read_text('compass.testcase', 'testcase.template'))
        test_case = {'name': self.name,
                     'config_filename': self.config_filename}
        work_dir = self.work_dir
        script = template.render(test_case=test_case)

        run_filename = os.path.join(work_dir, 'run.py')
        with open(run_filename, 'w') as handle:
            handle.write(script)

        # make sure it has execute permission
        st = os.stat(run_filename)
        os.chmod(run_filename, st.st_mode | stat.S_IEXEC)

    def _run_step(self, step, new_log_file):
        """
        Run the requested step

        Parameters
        ----------
        step : compass.Step
            The step to run

        new_log_file : bool
            Whether to log to a new log file
        """
        logger = self.logger
        config = self.config
        cwd = os.getcwd()
        available_cores, _ = get_available_cores_and_nodes(config)
        step.cores = min(step.cores, available_cores)
        if step.min_cores is not None:
            if step.cores < step.min_cores:
                raise ValueError(
                    'Available cores for {} is below the minimum of {}'
                    ''.format(step.cores, step.min_cores))

        missing_files = list()
        for input_file in step.inputs:
            if not os.path.exists(input_file):
                missing_files.append(input_file)

        if len(missing_files) > 0:
            raise OSError(
                'input file(s) missing in step {} of {}/{}/{}: {}'.format(
                    step.name, step.mpas_core.name, step.test_group.name,
                    step.test_case.subdir, missing_files))

        test_name = step.path.replace('/', '_')
        if new_log_file:
            log_filename = '{}/{}.log'.format(cwd, step.name)
            step.log_filename = log_filename
            step_logger = None
        else:
            step_logger = logger
            log_filename = None
        with LoggingContext(name=test_name, logger=step_logger,
                            log_filename=log_filename) as step_logger:
            step.logger = step_logger
            os.chdir(step.work_dir)
            step.run()

        missing_files = list()
        for output_file in step.outputs:
            if not os.path.exists(output_file):
                missing_files.append(output_file)

        if len(missing_files) > 0:
            raise OSError(
                'output file(s) missing in step {} of {}/{}/{}: {}'.format(
                    step.name, step.mpas_core.name, step.test_group.name,
                    step.test_case.subdir, missing_files))

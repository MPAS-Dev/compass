import os

from mpas_tools.logging import LoggingContext, check_call
from compass.parallel import get_available_cores_and_nodes
from compass.logging import log_method_call


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

    config : compass.config.CompassConfigParser
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

    baseline_dir : str, optional
        Location of the same test case within the baseline work directory,
        for use in comparing variables and timers

    stdout_logger : logging.Logger
        A logger for output from the test case that goes to stdout regardless
        of whether ``logger`` is a log file or stdout

    logger : logging.Logger
        A logger for output from the test case

    log_filename : str
        At run time, the name of a log file where output/errors from the test
        case are being logged, or ``None`` if output is to stdout/stderr

    new_step_log_file : bool
        Whether to create a new log file for each step or to log output to a
        common log file for the whole test case.  The latter is used when
        running the test case as part of a test suite

    validation : dict
        A dictionary with the status of internal and baseline comparisons, used
        by the ``compass`` framework to determine whether the test case passed
        or failed internal and baseline validation.
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
        # may be set during setup if there is a baseline for comparison
        self.baseline_dir = None

        # these will be set when running the test case
        self.new_step_log_file = True
        self.stdout_logger = None
        self.logger = None
        self.log_filename = None
        self.validation = None

    def configure(self):
        """
        Modify the configuration options for this test case. Test cases should
        override this method if they want to add config options specific to
        the test case, e.g. from a config file stored in the test case's python
        package.  If a test case overrides this method, it should assume that
        the ``<self.name>.cfg`` file in its package has already been added
        to the config options prior to calling ``configure()``.
        """
        pass

    def run(self):
        """
        Run each step of the test case.  Test cases can override this method
        to perform additional operations in addition to running the test case's
        steps

        Developers need to make sure they call ``super().run()`` at some point
        in the overridden ``run()`` method to actually call the steps of the
        run.  The developer will need to decide where in the overridden method
        to make the call to ``super().run()``, after any updates to steps
        based on config options, typically at the end of the new method.
        """
        logger = self.logger
        cwd = os.getcwd()
        for step_name in self.steps_to_run:
            step = self.steps[step_name]
            if step.cached:
                logger.info('  * Cached step: {}'.format(step_name))
                continue
            step.config = self.config
            if self.log_filename is not None:
                step.log_filename = self.log_filename

            self._print_to_stdout('  * step: {}'.format(step_name))

            try:
                if step.run_as_subprocess:
                    self._run_step_as_subprocess(step, self.new_step_log_file)
                else:
                    self._run_step(step, self.new_step_log_file)
            except BaseException:
                self._print_to_stdout('      Failed')
                raise
            os.chdir(cwd)

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        pass

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

    def check_validation(self):
        """
        Check the test case's "validation" dictionary to see if validation
        failed.
        """
        validation = self.validation
        logger = self.logger
        if validation is not None:
            internal_pass = validation['internal_pass']
            baseline_pass = validation['baseline_pass']

            both_pass = True
            if internal_pass is not None and not internal_pass:
                logger.error('Comparison failed between files within the test '
                             'case.')
                both_pass = False

            if baseline_pass is not None and not baseline_pass:
                logger.error('Comparison failed between the test case and the '
                             'baseline.')
                both_pass = False

            if both_pass:
                raise ValueError('Comparison failed, see above.')

    def _print_to_stdout(self, message):
        """
        write out a message to stdout if we're not running a single step on its
        own
        """
        if self.stdout_logger is not None:
            self.stdout_logger.info(message)
            if self.logger != self.stdout_logger:
                # also write it to the log file
                self.logger.info(message)

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
        step.constrain_resources(available_cores)

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
            step_logger.info('')
            log_method_call(method=step.run, logger=step_logger)
            step_logger.info('')
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

    def _run_step_as_subprocess(self, step, new_log_file):
        """
        Run the requested step as a subprocess

        Parameters
        ----------
        step : compass.Step
            The step to run

        new_log_file : bool
            Whether to log to a new log file
        """
        logger = self.logger
        cwd = os.getcwd()
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

            os.chdir(step.work_dir)
            step_args = ['compass', 'run', '--step_is_subprocess']
            check_call(step_args, step_logger)

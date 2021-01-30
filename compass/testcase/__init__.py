import os
import sys
import stat
from jinja2 import Template
from importlib import resources

from mpas_tools.logging import LoggingContext
from compass.parallel import get_available_cores_and_nodes


def add_testcase(testcases, module, **kwargs):
    """
    Add a test case to a list of test cases in a configuration.  This should
    be called for each test case in a configuration's ``collect()`` function.

    Any additional keyword arguments passed to this function will be added
    as keys and values to the ``testcase`` dictionary

    Parameters
    ----------
    testcases : `list`
        A list of test-case dictionaries to which the one for this test case
        should be appended

    module : `module`
        The python module containing the test case

    **kwargs
        Additional keyword arguments.

    Returns
    -------
    testcase : `dict`
        A dictionary of information about the test case
    """

    testcase = _get_testcase_default(module.__name__)
    testcase.update(kwargs)
    module.collect(testcase)
    if testcase['description'] is None:
        raise ValueError('No description was added for {}'.format(
            testcase['path']))
    testcases.append(testcase)

    return testcase


def set_testcase_subdir(testcase, subdir):
    """
    Replaces the default subdirectory and relative path in the test case using
    the given subdirectory.  This should be called in the test case's
    ``collect()`` function if the default subdirectory (the name of the test
    case) is not used.

    Parameters
    ----------
    testcase : `dict`
        A dictionary of information about the test case

    subdir : `str`
        A subdirectory for the test case within the configuration
    """
    testcase['subdir'] = subdir
    testcase['path'] = os.path.join(testcase['core'],
                                    testcase['configuration'], subdir)


def add_step(testcase, module, **kwargs):
    """
    Add a step to the dictionary of steps in the test case.  This should
    be called for each step in a test case's ``collect()`` function.

    Any additional keyword arguments passed to this function will be added
    as keys and values to the ``step`` dictionary

    Parameters
    ----------
    testcase : `dict`
        A dictionary of information about the test case

    module : `module`
        The python module containing the step

    **kwargs
        Additional keyword arguments.

    Returns
    -------
    step : `dict`
        A dictionary of information about the step
    """

    step = _get_step_default(module.__name__, testcase)
    step.update(kwargs)
    module.collect(testcase, step)
    testcase['steps'][step['name']] = step
    testcase['steps_to_run'].append(step['name'])
    return step


def run_steps(testcase, test_suite, config, logger):
    """
    Run the requested steps of a test case

    Parameters
    ----------
    testcase : dict
        The dictionary describing the test case with info from
        :py:func:`compass.testcase.get_testcase_default()` and any additional
        information added when collecting and setting up the test case.

    test_suite : dict
        A dictionary of properties of the test suite

    config : configparser.ConfigParser
        Configuration options for this test case

    logger : logging.Logger
        A logger for output from the test case
    """
    cwd = os.getcwd()
    for step_name in testcase['steps_to_run']:
        step = testcase['steps'][step_name]
        new_log_file = testcase['new_step_log_file']
        if 'log_filename' in testcase:
            step['log_filename'] = testcase['log_filename']
            do_local_logging = True
        else:
            # We only want to do local log output if the step output is being
            # redirected to a file.  Otherwise, we assume we're probably just
            # running one step and the local logging is redundant and
            # unnecessary
            do_local_logging = new_log_file

        if do_local_logging:
            logger.info(' * Running {}'.format(step_name))
        try:
            run_step(step, test_suite, config, logger, new_log_file)
        except BaseException:
            if do_local_logging:
                logger.info('     Failed')
            raise

        if do_local_logging:
            logger.info('     Complete')

        os.chdir(cwd)


def run_step(step, test_suite, config, logger, new_log_file):
    """
    Run the requested step of a test case

    Parameters
    ----------
    step : dict
        The dictionary describing the step with info from
        :py:func:`compass.testcase.get_step_default()` and any additional
        information added when collecting and setting up the step.

    test_suite : dict
        A dictionary of properties of the test suite

    config : configparser.ConfigParser
        Configuration options for this test case

    logger : logging.Logger
        A logger for output from the test case

    new_log_file : bool
        Whether to log to a new log file
    """
    cwd = os.getcwd()
    step_name = step['name']
    if 'cores' in step:
        available_cores, _ = get_available_cores_and_nodes(config)
        step['cores'] = min(step['cores'], available_cores)
    else:
        logger.warning('Core count not specified for step {}. Default is '
                       '1 core.'.format(step_name))
        step['cores'] = 1
    if 'min_cores' in step:
        if step['cores'] < step['min_cores']:
            raise ValueError(
                'Available cores for {} is below the minimum of {}'
                ''.format(step['cores'], step['min_cores']))

    missing_files = list()
    for input_file in step['inputs']:
        if not os.path.exists(input_file):
            missing_files.append(input_file)

    if len(missing_files) > 0:
        raise OSError(
            'input file(s) missing in step {} of {}/{}/{}: {}'.format(
                step_name, step['core'], step['configuration'],
                step['testcase_subdir'], missing_files))

    test_name = step['path'].replace('/', '_')
    if new_log_file:
        log_filename = '{}/{}.log'.format(cwd, step_name)
        step['log_filename'] = log_filename
        step_logger = None
    else:
        step_logger = logger
        log_filename = None
    with LoggingContext(name=test_name, logger=step_logger,
                        log_filename=log_filename) as step_logger:
        run = getattr(sys.modules[step['module']], step['run'])
        os.chdir(step['work_dir'])

        run(step, test_suite, config, step_logger)

    missing_files = list()
    for output_file in step['outputs']:
        if not os.path.exists(output_file):
            missing_files.append(output_file)

    if len(missing_files) > 0:
        raise OSError(
            'output file(s) missing in step {} of {}/{}/{}: {}'.format(
                step_name, step['core'], step['configuration'],
                step['testcase_subdir'], missing_files))


def generate_run(template_name, testcase, step=None):
    """
    Generate a ``run.py`` script for the given test case or step.

    Parameters
    ----------
    template_name : str
        The name of the template file to use to create the run script

    testcase : dict
        The dictionary of information about the test case, used to fill in the
        script template

    step : dict, optional
        The dictionary of information about the step, used to fill in the
        script template
    """

    template = Template(resources.read_text('compass.testcase', template_name))
    kwargs = {'testcase': testcase}
    if step is None:
        work_dir = testcase['work_dir']
    else:
        work_dir = step['work_dir']
        kwargs['step'] = step
    script = template.render(**kwargs)

    run_filename = os.path.join(work_dir, 'run.py')
    with open(run_filename, 'w') as handle:
        handle.write(script)

    # make sure it has execute permission
    st = os.stat(run_filename)
    os.chmod(run_filename, st.st_mode | stat.S_IEXEC)


def _get_step_default(module, testcase):
    """
    Set up a default dictionary describing the step in the given module.  The
    dictionary contains the full name of the python module for the step, the
    name of the step (the name of the python file without the ``.py``
    extension), the subdirectory for the step (the same as the ``name``),
    the names of the ``setup()`` and ``run()`` functions within the module,
    and empty lists of ``inputs`` and ``outputs``, to be filled with the
    files required to run the step or produced by the step, respectively.

    Parameters
    ----------
    module : str
        The full name of the python module for the step, usually supplied from
        ``__name__``

    testcase : dict
        A dictionary with information about the test case that this step
        belongs to

    Returns
    -------
    step : dict
        A dictionary with the default information about the step, most of which
        can be modified as appropriate

    """
    name = module.split('.')[-1]
    # some steps may not need a setup function
    if hasattr(sys.modules[module], 'setup'):
        setup = 'setup'
    else:
        setup = None
    step = {'module': module,
            'name': name,
            'core': testcase['core'],
            'configuration': testcase['configuration'],
            'testcase': testcase['name'],
            'testcase_subdir': testcase['subdir'],
            'subdir': name,
            'setup': setup,
            'run': 'run',
            'inputs': [],
            'outputs': []}
    return step


def _get_testcase_default(module):
    """
    Set up a default dictionary describing the test case in the given module.
    The dictionary contains the full name of the python module for the
    test case, the name of the test case (the final part of the full module
    name), the default subdirectory for the test case (the same as the
    ``name``), the names of the ``configure()`` and ``run()`` functions within
    the module, and the ``core`` and ``configuration`` of the test case (the
    parsed from the full module name).

    Parameters
    ----------
    module : str
        The full name of the python module for the test case, usually supplied
        from ``__name__``

    Returns
    -------
    testcase : dict
        A dictionary with the default information about the test case, most of
        which can be modified as appropriate
    """
    name = module.split('.')[-1]
    core = module.split('.')[1]
    configuration = module.split('.')[3]
    subdir = name
    path = os.path.join(core, configuration, subdir)
    # some test cases may not need a configure function
    if hasattr(sys.modules[module], 'configure'):
        configure = 'configure'
    else:
        configure = None
    testcase = {'module': module,
                'name': name,
                'path': path,
                'core': core,
                'configuration': configuration,
                'subdir': subdir,
                'description': None,
                'steps': dict(),
                'configure': configure,
                'run': 'run',
                'new_step_log_file': True,
                'steps_to_run': list()}
    return testcase

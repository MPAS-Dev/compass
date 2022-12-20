import argparse
import sys
import os
import pickle
import time
import glob
import inspect

from mpas_tools.logging import LoggingContext, check_call
import mpas_tools.io
from compass.parallel import check_parallel_system, set_cores_per_node, \
    get_available_cores_and_nodes
from compass.logging import log_method_call, log_function_call
from compass.config import CompassConfigParser


def run_tests(suite_name, quiet=False, is_test_case=False, steps_to_run=None,
              steps_not_to_run=None):
    """
    Run the given test suite or test case

    Parameters
    ----------
    suite_name : str
        The name of the test suite

    quiet : bool, optional
        Whether step names are not included in the output as the test suite
        progresses

    is_test_case : bool
        Whether this is a test case instead of a full test suite

    steps_to_run : list of str, optional
        A list of the steps to run if this is a test case, not a full suite.
        The default behavior is to run the default steps unless they are in
        ``steps_not_to_run``

    steps_not_to_run : list of str, optional
        A list of steps not to run if this is a test case, not a full suite.
        Typically, these are steps to remove from the defaults
    """
    # ANSI fail text: https://stackoverflow.com/a/287944/7728169
    start_fail = '\033[91m'
    start_pass = '\033[92m'
    start_time_color = '\033[94m'
    end = '\033[0m'
    pass_str = f'{start_pass}PASS{end}'
    success_str = f'{start_pass}SUCCESS{end}'
    fail_str = f'{start_fail}FAIL{end}'
    error_str = f'{start_fail}ERROR{end}'

    # Allow a suite name to either include or not the .pickle suffix
    if suite_name.endswith('.pickle'):
        # code below assumes no suffix, so remove it
        suite_name = suite_name[:-len('.pickle')]
    # Now open the the suite's pickle file
    if not os.path.exists(f'{suite_name}.pickle'):
        raise ValueError(f'The suite "{suite_name}" does not appear to have '
                         f'been set up here.')
    with open(f'{suite_name}.pickle', 'rb') as handle:
        test_suite = pickle.load(handle)

    # get the config file for the first test case in the suite
    test_case = next(iter(test_suite['test_cases'].values()))
    config_filename = os.path.join(test_case.work_dir,
                                   test_case.config_filename)
    config = CompassConfigParser()
    config.add_from_file(config_filename)
    check_parallel_system(config)

    # start logging to stdout/stderr
    with LoggingContext(suite_name) as logger:

        os.environ['PYTHONUNBUFFERED'] = '1'

        if not is_test_case:
            try:
                os.makedirs('case_outputs')
            except OSError:
                pass

        failures = 0
        cwd = os.getcwd()
        suite_start = time.time()
        test_times = dict()
        success = dict()
        for test_name in test_suite['test_cases']:
            test_case = test_suite['test_cases'][test_name]

            logger.info(f'{test_name}')

            test_name = test_case.path.replace('/', '_')
            if is_test_case:
                log_filename = None
                test_logger = logger
            else:
                log_filename = f'{cwd}/case_outputs/{test_name}.log'
                test_logger = None
            with LoggingContext(test_name, logger=test_logger,
                                log_filename=log_filename) as test_logger:
                if quiet:
                    # just log the step names and any failure messages to the
                    # log file
                    test_case.stdout_logger = test_logger
                else:
                    # log steps to stdout
                    test_case.stdout_logger = logger
                test_case.logger = test_logger
                test_case.log_filename = log_filename
                test_case.new_step_log_file = is_test_case

                os.chdir(test_case.work_dir)

                config = CompassConfigParser()
                config.add_from_file(test_case.config_filename)
                test_case.config = config
                set_cores_per_node(test_case.config)

                mpas_tools.io.default_format = config.get('io', 'format')
                mpas_tools.io.default_engine = config.get('io', 'engine')

                test_case.steps_to_run = _update_steps_to_run(
                    steps_to_run, steps_not_to_run, config, test_case.steps)

                test_start = time.time()
                log_method_call(method=test_case.run, logger=test_logger)
                test_logger.info('')
                try:
                    _test_case_run_deprecated(test_case)
                    run_status = success_str
                    test_pass = True
                except BaseException:
                    run_status = error_str
                    test_pass = False
                    test_logger.exception('Exception raised in the test '
                                          'case\'s run() method')

                if test_pass:
                    log_function_call(function=_run_test, logger=test_logger)
                    test_logger.info('')
                    test_list = ', '.join(test_case.steps_to_run)
                    test_logger.info(f'Running steps: {test_list}')
                    try:
                        _run_test(test_case)
                        run_status = success_str
                        test_pass = True
                    except BaseException:
                        run_status = error_str
                        test_pass = False
                        test_logger.exception('Exception raised while running '
                                              'the steps of the test case')

                if test_pass:
                    test_logger.info('')
                    log_method_call(method=test_case.validate,
                                    logger=test_logger)
                    test_logger.info('')
                    try:
                        test_case.validate()
                    except BaseException:
                        run_status = error_str
                        test_pass = False
                        test_logger.exception('Exception raised in the test '
                                              'case\'s validate() method')

                baseline_status = None
                internal_status = None
                if test_case.validation is not None:
                    internal_pass = test_case.validation['internal_pass']
                    baseline_pass = test_case.validation['baseline_pass']

                    if internal_pass is not None:
                        if internal_pass:
                            internal_status = pass_str
                        else:
                            internal_status = fail_str
                            test_logger.error(
                                'Internal test case validation failed')
                            test_pass = False

                    if baseline_pass is not None:
                        if baseline_pass:
                            baseline_status = pass_str
                        else:
                            baseline_status = fail_str
                            test_logger.error('Baseline validation failed')
                            test_pass = False

                status = f'  test execution:      {run_status}'
                if internal_status is not None:
                    status = f'{status}\n' \
                             f'  test validation:     {internal_status}'
                if baseline_status is not None:
                    status = f'{status}\n' \
                             f'  baseline comparison: {baseline_status}'

                if test_pass:
                    logger.info(status)
                    success[test_name] = pass_str
                else:
                    logger.error(status)
                    if not is_test_case:
                        logger.error(f'  see: case_outputs/{test_name}.log')
                    success[test_name] = fail_str
                    failures += 1

                test_times[test_name] = time.time() - test_start

                secs = round(test_times[test_name])
                mins = secs // 60
                secs -= 60 * mins
                logger.info(f'  test runtime:        '
                            f'{start_time_color}{mins:02d}:{secs:02d}{end}')

        suite_time = time.time() - suite_start

        os.chdir(cwd)

        logger.info('Test Runtimes:')
        for test_name, test_time in test_times.items():
            secs = round(test_time)
            mins = secs // 60
            secs -= 60 * mins
            logger.info(f'{mins:02d}:{secs:02d} {success[test_name]} '
                        f'{test_name}')
        secs = round(suite_time)
        mins = secs // 60
        secs -= 60 * mins
        logger.info(f'Total runtime {mins:02d}:{secs:02d}')

        if failures == 0:
            logger.info('PASS: All passed successfully!')
        else:
            if failures == 1:
                message = '1 test'
            else:
                message = f'{failures} tests'
            logger.error(f'FAIL: {message} failed, see above.')
            sys.exit(1)


def run_single_step(step_is_subprocess=False):
    """
    Used by the framework to run a step when ``compass run`` gets called in the
    step's work directory

    Parameters
    ----------
    step_is_subprocess : bool, optional
        Whether the step is being run as a subprocess of a test case or suite
    """
    with open('step.pickle', 'rb') as handle:
        test_case, step = pickle.load(handle)
    test_case.steps_to_run = [step.name]
    test_case.new_step_log_file = False

    if step_is_subprocess:
        step.run_as_subprocess = False

    config = CompassConfigParser()
    config.add_from_file(step.config_filename)

    check_parallel_system(config)

    test_case.config = config
    set_cores_per_node(test_case.config)

    mpas_tools.io.default_format = config.get('io', 'format')
    mpas_tools.io.default_engine = config.get('io', 'engine')

    # start logging to stdout/stderr
    test_name = step.path.replace('/', '_')
    with LoggingContext(name=test_name) as logger:
        test_case.logger = logger
        test_case.stdout_logger = None
        log_function_call(function=_run_test, logger=logger)
        logger.info('')
        _run_test(test_case)

        if not step_is_subprocess:
            # only perform validation if the step is being run by a user on its
            # own
            logger.info('')
            log_method_call(method=test_case.validate, logger=logger)
            logger.info('')
            test_case.validate()


def main():
    parser = argparse.ArgumentParser(
        description='Run a test suite, test case or step',
        prog='compass run')
    parser.add_argument("suite", nargs='?',
                        help="The name of a test suite to run. Can exclude "
                        "or include the .pickle filename suffix.")
    parser.add_argument("--steps", dest="steps", nargs='+',
                        help="The steps of a test case to run")
    parser.add_argument("--no-steps", dest="no_steps", nargs='+',
                        help="The steps of a test case not to run, see "
                             "steps_to_run in the config file for defaults.")
    parser.add_argument("-q", "--quiet", dest="quiet", action="store_true",
                        help="If set, step names are not included in the "
                             "output as the test suite progresses.  Has no "
                             "effect when running test cases or steps on "
                             "their own.")
    parser.add_argument("--step_is_subprocess", dest="step_is_subprocess",
                        action="store_true",
                        help="Used internally by compass to indicate that"
                             "a step is being run as a subprocess.")
    args = parser.parse_args(sys.argv[2:])
    if args.suite is not None:
        run_tests(args.suite, quiet=args.quiet)
    elif os.path.exists('test_case.pickle'):
        run_tests(suite_name='test_case', quiet=args.quiet, is_test_case=True,
                  steps_to_run=args.steps, steps_not_to_run=args.no_steps)
    elif os.path.exists('step.pickle'):
        run_single_step(args.step_is_subprocess)
    else:
        pickles = glob.glob('*.pickle')
        if len(pickles) == 1:
            suite = os.path.splitext(os.path.basename(pickles[0]))[0]
            run_tests(suite, quiet=args.quiet)
        elif len(pickles) == 0:
            raise OSError('No pickle files were found. Are you sure this is '
                          'a compass suite, test-case or step work directory?')
        else:
            raise ValueError('More than one suite was found. Please specify '
                             'which to run: compass run <suite>')


def _update_steps_to_run(steps_to_run, steps_not_to_run, config, steps):
    """
    Update the steps to run
    """
    if steps_to_run is None:
        steps_to_run = config.get('test_case',
                                  'steps_to_run').replace(',', ' ').split()

    for step in steps_to_run:
        if step not in steps:
            raise ValueError(
                f'A step "{step}" was requested but is not one of the steps '
                f'in this test case:'
                f'\n{list(steps)}')

    if steps_not_to_run is not None:
        for step in steps_not_to_run:
            if step not in steps:
                raise ValueError(
                    f'A step "{step}" was flagged not to run but is not one '
                    f'of the steps in this test case:'
                    f'\n{list(steps)}')

        steps_to_run = [step for step in steps_to_run if step not in
                        steps_not_to_run]

    return steps_to_run


def _print_to_stdout(test_case, message):
    """
    Write out a message to stdout if we're not running a single step
    """
    if test_case.stdout_logger is not None:
        test_case.stdout_logger.info(message)
        if test_case.logger != test_case.stdout_logger:
            # also write it to the log file
            test_case.logger.info(message)


def _run_test(test_case):
    """
    Run each step of the test case
    """
    logger = test_case.logger
    cwd = os.getcwd()
    for step_name in test_case.steps_to_run:
        step = test_case.steps[step_name]
        if step.cached:
            logger.info(f'  * Cached step: {step_name}')
            continue
        step.config = test_case.config
        if test_case.log_filename is not None:
            step.log_filename = test_case.log_filename

        _print_to_stdout(test_case, f'  * step: {step_name}')

        try:
            if step.run_as_subprocess:
                _run_step_as_subprocess(
                    test_case, step, test_case.new_step_log_file)
            else:
                _run_step(test_case, step, test_case.new_step_log_file)
        except BaseException:
            _print_to_stdout(test_case, '      Failed')
            raise
        os.chdir(cwd)


def _run_step(test_case, step, new_log_file):
    """
    Run the requested step
    """
    logger = test_case.logger
    config = test_case.config
    cwd = os.getcwd()
    available_cores, _, _ = get_available_cores_and_nodes(config)
    step.constrain_resources(available_cores)

    missing_files = list()
    for input_file in step.inputs:
        if not os.path.exists(input_file):
            missing_files.append(input_file)

    if len(missing_files) > 0:
        raise OSError(
            f'input file(s) missing in step {step.name} of '
            f'{step.mpas_core.name}/{step.test_group.name}/'
            f'{step.test_case.subdir}: {missing_files}')

    test_name = step.path.replace('/', '_')
    if new_log_file:
        log_filename = f'{cwd}/{step.name}.log'
        step.log_filename = log_filename
        step_logger = None
    else:
        step_logger = logger
        log_filename = None
    with LoggingContext(name=test_name, logger=step_logger,
                        log_filename=log_filename) as step_logger:
        step.logger = step_logger
        os.chdir(step.work_dir)

        # runtime_setup() will perform small tasks that require knowing the
        # resources of the task before the step runs (such as creating
        # graph partitions)
        step_logger.info('')
        log_method_call(method=step.runtime_setup, logger=step_logger)
        step_logger.info('')
        step.runtime_setup()

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
            f'output file(s) missing in step {step.name} of '
            f'{step.mpas_core.name}/{step.test_group.name}/'
            f'{step.test_case.subdir}: {missing_files}')


def _run_step_as_subprocess(test_case, step, new_log_file):
    """
    Run the requested step as a subprocess
    """
    logger = test_case.logger
    cwd = os.getcwd()
    test_name = step.path.replace('/', '_')
    if new_log_file:
        log_filename = f'{cwd}/{step.name}.log'
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


def _test_case_run_deprecated(test_case):
    method = test_case.run

    # get the "child" class and its location (import sequence) from the method
    child_class = method.__self__.__class__

    # iterate over the classes that the child class descends from to find the
    # first one that actually implements the given method.
    actual_class = None
    # inspect.getmro() returns a list of classes the child class descends from,
    # starting with the child class itself and going "back" to the "object"
    # class that all python classes descend from.
    for cls in inspect.getmro(child_class):
        if method.__name__ in cls.__dict__:
            actual_class = cls
            break

    if actual_class is None:
        raise ValueError('We could not find test_case.run(). Something is '
                         'buggy!')

    if actual_class.__name__ != 'TestCase':
        # the run() method has been overridden.  We need to give the user a
        # deprecation warning.
        actual_location = f'{actual_class.__module__}.{actual_class.__name__}'
        test_case.logger.warn(
            f'\nWARNING: Overriding the TestCase.run() method is deprecated.\n'
            f'  Please move the contents of\n'
            f'  {actual_location}.run() \n'
            f'  to the runtime_setup() or constrain_resources() methods of '
            f'its steps.\n')

    test_case.run()

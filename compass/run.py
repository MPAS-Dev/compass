import argparse
import sys
import os
import pickle
import configparser
import time
import glob

from mpas_tools.logging import LoggingContext
import mpas_tools.io
from compass.parallel import check_parallel_system, set_cores_per_node


def run_suite(suite_name, quiet=False):
    """
    Run the given test suite

    Parameters
    ----------
    suite_name : str
        The name of the test suite

    quiet : bool, optional
        Whether step names are not included in the output as the test suite
        progresses

    """
    # ANSI fail text: https://stackoverflow.com/a/287944/7728169
    start_fail = '\033[91m'
    start_pass = '\033[92m'
    end = '\033[0m'
    pass_str = '{}PASS{}'.format(start_pass, end)
    success_str = '{}SUCCESS{}'.format(start_pass, end)
    fail_str = '{}FAIL{}'.format(start_fail, end)
    error_str = '{}ERROR{}'.format(start_fail, end)

    # Allow a suite name to either include or not the .pickle suffix
    if suite_name.endswith('.pickle'):
        # code below assumes no suffix, so remove it
        suite_name = suite_name[:-len('.pickle')]
    # Now open the the suite's pickle file
    if not os.path.exists('{}.pickle'.format(suite_name)):
        raise ValueError('The suite "{}" doesn\'t appear to have been set up '
                         'here.'.format(suite_name))
    with open('{}.pickle'.format(suite_name), 'rb') as handle:
        test_suite = pickle.load(handle)

    # get the config file for the first test case in the suite
    test_case = next(iter(test_suite['test_cases'].values()))
    config_filename = os.path.join(test_case.work_dir,
                                   test_case.config_filename)
    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation())
    config.read(config_filename)
    check_parallel_system(config)

    # start logging to stdout/stderr
    with LoggingContext(suite_name) as logger:

        os.environ['PYTHONUNBUFFERED'] = '1'

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

            logger.info('{}'.format(test_name))

            test_name = test_case.path.replace('/', '_')
            log_filename = '{}/case_outputs/{}.log'.format(cwd, test_name)
            with LoggingContext(test_name, log_filename=log_filename) as \
                    test_logger:
                if quiet:
                    # just log the step names and any failure messages to the
                    # log file
                    test_case.stdout_logger = test_logger
                else:
                    # log steps to stdout
                    test_case.stdout_logger = logger
                test_case.logger = test_logger
                test_case.log_filename = log_filename
                test_case.new_step_log_file = False

                os.chdir(test_case.work_dir)

                config = configparser.ConfigParser(
                    interpolation=configparser.ExtendedInterpolation())
                config.read(test_case.config_filename)
                test_case.config = config
                set_cores_per_node(test_case.config)

                mpas_tools.io.default_format = config.get('io', 'format')
                mpas_tools.io.default_engine = config.get('io', 'engine')

                test_case.steps_to_run = config.get(
                    'test_case', 'steps_to_run').replace(',', ' ').split()

                test_start = time.time()
                try:
                    test_case.run()
                    run_status = success_str
                    test_pass = True
                except BaseException:
                    run_status = error_str
                    test_pass = False
                    test_logger.exception('Exception raised in run()')

                if test_pass:
                    try:
                        test_case.validate()
                    except BaseException:
                        run_status = error_str
                        test_pass = False
                        test_logger.exception('Exception raised in validate()')

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
                            test_logger.exception(
                                'Internal test case validation failed')
                            test_pass = False

                    if baseline_pass is not None:
                        if baseline_pass:
                            baseline_status = pass_str
                        else:
                            baseline_status = fail_str
                            test_logger.exception('Baseline validation failed')
                            test_pass = False

                status = '  test execution:      {}'.format(run_status)
                if internal_status is not None:
                    status = '{}\n  test validation:     {}'.format(
                        status, internal_status)
                if baseline_status is not None:
                    status = '{}\n  baseline comparison: {}'.format(
                        status, baseline_status)

                if test_pass:
                    logger.info(status)
                    success[test_name] = pass_str
                else:
                    logger.error(status)
                    logger.error('  see: case_outputs/{}.log'.format(
                        test_name))
                    success[test_name] = fail_str
                    failures += 1

                test_times[test_name] = time.time() - test_start

        suite_time = time.time() - suite_start

        os.chdir(cwd)

        logger.info('Test Runtimes:')
        for test_name, test_time in test_times.items():
            secs = round(test_time)
            mins = secs // 60
            secs -= 60 * mins
            logger.info('{:02d}:{:02d} {} {}'.format(
                mins, secs, success[test_name], test_name))
        secs = round(suite_time)
        mins = secs // 60
        secs -= 60 * mins
        logger.info('Total runtime {:02d}:{:02d}'.format(mins, secs))

        if failures == 0:
            logger.info('PASS: All passed successfully!')
        else:
            if failures == 1:
                message = '1 test'
            else:
                message = '{} tests'.format(failures)
            logger.error('FAIL: {} failed, see above.'.format(message))
            sys.exit(1)


def run_test_case(steps_to_run=None, steps_not_to_run=None):
    """
    Used by the framework to run a test case when ``compass run`` gets called
    in the test case's work directory

    Parameters
    ----------
    steps_to_run : list of str, optional
        A list of the steps to run.  The default behavior is to run the default
        steps unless they are in ``steps_not_to_run``

    steps_not_to_run : list of str, optional
        A list of steps not to run.  Typically, these are steps to remove from
        the defaults
    """
    with open('test_case.pickle', 'rb') as handle:
        test_case = pickle.load(handle)

    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation())
    config.read(test_case.config_filename)

    check_parallel_system(config)

    test_case.config = config
    set_cores_per_node(test_case.config)

    mpas_tools.io.default_format = config.get('io', 'format')
    mpas_tools.io.default_engine = config.get('io', 'engine')

    if steps_to_run is None:
        steps_to_run = config.get('test_case',
                                  'steps_to_run').replace(',', ' ').split()

    for step in steps_to_run:
        if step not in test_case.steps:
            raise ValueError(
                'A step "{}" was requested but is not one of the steps in '
                'this test case:\n{}'.format(step, list(test_case.steps)))

    if steps_not_to_run is not None:
        for step in steps_not_to_run:
            if step not in test_case.steps:
                raise ValueError(
                    'A step "{}" was flagged not to run but is not one of the '
                    'steps in this test case:'
                    '\n{}'.format(step, list(test_case.steps)))

        steps_to_run = [step for step in steps_to_run if step not in
                        steps_not_to_run]

    test_case.steps_to_run = steps_to_run

    # start logging to stdout/stderr
    test_name = test_case.path.replace('/', '_')
    test_case.new_step_log_file = True
    with LoggingContext(name=test_name) as logger:
        test_case.logger = logger
        test_case.stdout_logger = logger
        logger.info('Running steps: {}'.format(', '.join(steps_to_run)))
        test_case.run()
        test_case.validate()


def run_step():
    """
    Used by the framework to run a step when ``compass run`` gets called in the
    step's work directory
    """
    with open('step.pickle', 'rb') as handle:
        test_case, step = pickle.load(handle)
    test_case.steps_to_run = [step.name]
    test_case.new_step_log_file = False

    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation())
    config.read(step.config_filename)

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
        test_case.run()
        test_case.validate()


def main():
    parser = argparse.ArgumentParser(
        description='Run a test suite, test case or step',
        prog='compass run')
    parser.add_argument("suite", nargs='?', default=None,
                        help="The name of a test suite to run. Can exclude "
                        "or include the .pickle filename suffix.")
    parser.add_argument("--steps", dest="steps", nargs='+', default=None,
                        help="The steps of a test case to run")
    parser.add_argument("--no-steps", dest="no_steps", nargs='+', default=None,
                        help="The steps of a test case not to run, see "
                             "steps_to_run in the config file for defaults.")
    parser.add_argument("-q", "--quiet", dest="quiet", action="store_true",
                        help="If set, step names are not included in the "
                             "output as the test suite progresses.  Has no "
                             "effect when running test cases or steps on "
                             "their own.")
    args = parser.parse_args(sys.argv[2:])
    if args.suite is not None:
        run_suite(args.suite, quiet=args.quiet)
    elif os.path.exists('test_case.pickle'):
        run_test_case(args.steps, args.no_steps)
    elif os.path.exists('step.pickle'):
        run_step()
    else:
        pickles = glob.glob('*.pickle')
        if len(pickles) == 1:
            suite = os.path.splitext(os.path.basename(pickles[0]))[0]
            run_suite(suite, quiet=args.quiet)
        elif len(pickles) == 0:
            raise OSError('No pickle files were found. Are you sure this is '
                          'a compass suite, test-case or step work directory?')
        else:
            raise ValueError('More than one suite was found. Please specify '
                             'which to run: compass run <suite>')

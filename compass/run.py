import argparse
import sys
import os
import pickle
import configparser
import time
import numpy

from mpas_tools.logging import LoggingContext


def run_suite(suite_name):
    """
    Run the given test suite

    Parameters
    ----------
    suite_name : str
        The name of the test suite
    """
    # ANSI fail text: https://stackoverflow.com/a/287944/7728169
    start_fail = '\033[91m'
    start_pass = '\033[92m'
    end = '\033[0m'
    pass_str = '{}PASS{}'.format(start_pass, end)
    success_str = '{}SUCCESS{}'.format(start_pass, end)
    fail_str = '{}FAIL{}'.format(start_fail, end)
    error_str = '{}ERROR{}'.format(start_fail, end)

    if not os.path.exists('{}.pickle'.format(suite_name)):
        raise ValueError('The suite "{}" doesn\'t appear to have been set up '
                         'here.'.format(suite_name))
    with open('{}.pickle'.format(suite_name), 'rb') as handle:
        test_suite = pickle.load(handle)

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
                test_case.logger = test_logger
                test_case.log_filename = log_filename
                test_case.new_step_log_file = False

                os.chdir(test_case.work_dir)

                config = configparser.ConfigParser(
                    interpolation=configparser.ExtendedInterpolation())
                config.read(test_case.config_filename)
                test_case.config = config

                test_start = time.time()
                try:
                    test_case.run()
                    run_status = success_str
                    test_pass = True
                except BaseException:
                    run_status = error_str
                    test_pass = False
                    test_logger.exception('Exception raised')

                internal_status = None
                if test_pass:
                    try:
                        test_case.validate()
                        internal_status = pass_str
                    except BaseException:
                        internal_status = fail_str
                        test_pass = False
                        test_logger.exception('Exception raised')

                baseline_status = None
                if test_case.validation is not None:
                    internal_pass = test_case.validation['internal_pass']
                    baseline_pass = test_case.validation['baseline_pass']

                    if internal_pass is not None and not internal_pass:
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

                if internal_status is None:
                    status = '  {}'.format(run_status)
                elif baseline_status is None:
                    status = '  test execution:      {}\n' \
                             '  test validation:     {}'.format(
                                  run_status, internal_status)
                else:
                    status = '  test execution:      {}\n' \
                             '  test validation:     {}\n' \
                             '  baseline comparison: {}'.format(
                                  run_status, internal_status, baseline_status)

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
            mins = int(numpy.floor(test_time / 60.0))
            secs = int(numpy.ceil(test_time - mins * 60))
            logger.info('{:02d}:{:02d} {} {}'.format(
                mins, secs, success[test_name], test_name))
        mins = int(numpy.floor(suite_time / 60.0))
        secs = int(numpy.ceil(suite_time - mins * 60))
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


def run_test_case():
    """
    Used by the framework to run a test case when ``compass run`` gets called
    in the test case's work directory
    """
    with open('test_case.pickle', 'rb') as handle:
        test_case = pickle.load(handle)

    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation())
    config.read(test_case.config_filename)
    test_case.config = config

    # start logging to stdout/stderr
    test_name = test_case.path.replace('/', '_')
    test_case.new_step_log_file = True
    with LoggingContext(name=test_name) as logger:
        test_case.logger = logger
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
    test_case.config = config

    # start logging to stdout/stderr
    test_name = step.path.replace('/', '_')
    with LoggingContext(name=test_name) as logger:
        test_case.logger = logger
        test_case.run()
        test_case.validate()


def main():
    parser = argparse.ArgumentParser(
        description='Run a test suite, test case or step',
        prog='compass run')
    parser.add_argument("suite", nargs='?', default=None,
                        help="The name of a test suite to run")
    args = parser.parse_args(sys.argv[2:])
    if args.suite is not None:
        run_suite(args.suite)
    elif os.path.exists('test_case.pickle'):
        run_test_case()
    elif os.path.exists('step.pickle'):
        run_step()
    else:
        raise OSError('A suite name was not given but the current directory '
                      'does not contain a test case or step.')

import argparse
import sys
import os
import pickle
import warnings

from mache import discover_machine

from compass.mpas_cores import get_mpas_cores
from compass.config import CompassConfigParser
from compass.io import symlink
from compass import provenance
from compass.job import write_job_script


def setup_cases(tests=None, numbers=None, config_file=None, machine=None,
                work_dir=None, baseline_dir=None, mpas_model_path=None,
                suite_name='custom', cached=None):
    """
    Set up one or more test cases

    Parameters
    ----------
    tests : list of str, optional
        Relative paths for a test cases to set up

    numbers : list of str, optional
        Case numbers to setup, as listed from ``compass list``, optionally with
        a suffix ``c`` to indicate that all steps in that test case should be
        cached

    config_file : str, optional
        Configuration file with custom options for setting up and running test
        cases

    machine : str, optional
        The name of one of the machines with defined config options, which can
        be listed with ``compass list --machines``

    work_dir : str, optional
        A directory that will serve as the base for creating case directories

    baseline_dir : str, optional
        Location of baselines that can be compared to

    mpas_model_path : str, optional
        The relative or absolute path to the root of a branch where the MPAS
        model has been built

    suite_name : str, optional
        The name of the test suite if tests are being set up through a test
        suite or ``'custom'`` if not

    cached : list of list of str, optional
        For each test in ``tests``, which steps (if any) should be cached,
        or a list with "_all" as the first entry if all steps in the test case
        should be cached

    Returns
    -------
    test_cases : dict of compass.TestCase
        A dictionary of test cases, with the relative path in the work
        directory as keys
    """
    if machine is None and 'COMPASS_MACHINE' in os.environ:
        machine = os.environ['COMPASS_MACHINE']

    if machine is None:
        machine = discover_machine()

    if config_file is None and machine is None:
        raise ValueError('At least one of config_file and machine is needed.')

    if config_file is not None and not os.path.exists(config_file):
        raise FileNotFoundError(
            f'The user config file wasn\'t found: {config_file}')

    if tests is None and numbers is None:
        raise ValueError('At least one of tests or numbers is needed.')

    if cached is not None:
        if tests is None:
            warnings.warn('Ignoring "cached" argument because "tests" was '
                          'not provided')
        elif len(cached) != len(tests):
            raise ValueError('A list of cached steps must be provided for '
                             'each test in "tests"')

    if work_dir is None:
        work_dir = os.getcwd()
    work_dir = os.path.abspath(work_dir)

    mpas_cores = get_mpas_cores()

    all_test_cases = dict()
    for mpas_core in mpas_cores:
        for test_group in mpas_core.test_groups.values():
            for test_case in test_group.test_cases.values():
                all_test_cases[test_case.path] = test_case

    test_cases = dict()
    cached_steps = dict()
    if numbers is not None:
        keys = list(all_test_cases)
        for number in numbers:
            cache_all = False
            if number.endswith('c'):
                cache_all = True
                number = int(number[:-1])
            else:
                number = int(number)

            if number >= len(keys):
                raise ValueError('test number {} is out of range.  There are '
                                 'only {} tests.'.format(number, len(keys)))
            path = keys[number]
            if cache_all:
                cached_steps[path] = ['_all']
            else:
                cached_steps[path] = list()
            test_cases[path] = all_test_cases[path]

    if tests is not None:
        for index, path in enumerate(tests):
            if path not in all_test_cases:
                raise ValueError('Test case with path {} is not in '
                                 'test_cases'.format(path))
            if cached is not None:
                cached_steps[path] = cached[index]
            else:
                cached_steps[path] = list()
            test_cases[path] = all_test_cases[path]

    # get the MPAS core of the first test case.  We'll assume all tests are
    # for this core
    first_path = next(iter(test_cases))
    mpas_core = test_cases[first_path].mpas_core.name

    basic_config = _get_basic_config(config_file, machine, mpas_model_path,
                                     mpas_core)

    provenance.write(work_dir, test_cases, config=basic_config)

    print('Setting up test cases:')
    for path, test_case in test_cases.items():
        setup_case(path, test_case, config_file, machine, work_dir,
                   baseline_dir, mpas_model_path,
                   cached_steps=cached_steps[path])

    test_suite = {'name': suite_name,
                  'test_cases': test_cases,
                  'work_dir': work_dir}

    # pickle the test or step dictionary for use at runtime
    pickle_file = os.path.join(test_suite['work_dir'],
                               '{}.pickle'.format(suite_name))
    with open(pickle_file, 'wb') as handle:
        pickle.dump(test_suite, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if 'LOAD_COMPASS_ENV' in os.environ:
        script_filename = os.environ['LOAD_COMPASS_ENV']
        # make a symlink to the script for loading the compass conda env.
        symlink(script_filename, os.path.join(work_dir, 'load_compass_env.sh'))

    max_cores, max_of_min_cores = _get_required_cores(test_cases)

    print(f'target cores: {max_cores}')
    print(f'minimum cores: {max_of_min_cores}')

    if machine is not None:
        write_job_script(basic_config, machine, max_cores, max_of_min_cores,
                         work_dir, suite=suite_name)

    return test_cases


def setup_case(path, test_case, config_file, machine, work_dir, baseline_dir,
               mpas_model_path, cached_steps):
    """
    Set up one or more test cases

    Parameters
    ----------
    path : str
        Relative path for a test cases to set up

    test_case : compass.TestCase
        A test case to set up

    config_file : str
        Configuration file with custom options for setting up and running test
        cases

    machine : str
        The name of one of the machines with defined config options, which can
        be listed with ``compass list --machines``

    work_dir : str
        A directory that will serve as the base for creating case directories

    baseline_dir : str
        Location of baselines that can be compared to

    mpas_model_path : str
        The relative or absolute path to the root of a branch where the MPAS
        model has been built

    cached_steps : list of str
        Which steps (if any) should be cached.  If all steps should be cached,
         the first entry is "_all"
    """

    print('  {}'.format(path))

    mpas_core = test_case.mpas_core.name
    config = _get_basic_config(config_file, machine, mpas_model_path,
                               mpas_core)

    # add the config options for the test group (if defined)
    test_group = test_case.test_group.name
    config.add_from_package(f'compass.{mpas_core}.tests.{test_group}',
                            f'{test_group}.cfg', exception=False)

    # add the config options for the test case (if defined)
    config.add_from_package(test_case.__module__,
                            f'{test_case.name}.cfg', exception=False)

    if 'COMPASS_BRANCH' in os.environ:
        compass_branch = os.environ['COMPASS_BRANCH']
        config.set('paths', 'compass_branch', compass_branch)
    else:
        config.set('paths', 'compass_branch', os.getcwd())

    test_case_dir = os.path.join(work_dir, path)
    try:
        os.makedirs(test_case_dir)
    except OSError:
        pass
    test_case.work_dir = test_case_dir
    test_case.base_work_dir = work_dir

    # add config options specific to the test case
    test_case.config = config
    test_case.configure()

    # add the baseline directory for this test case
    if baseline_dir is not None:
        test_case.baseline_dir = os.path.join(baseline_dir, path)

    # set the mpas_model path from the command line if provided
    if mpas_model_path is not None:
        mpas_model_path = os.path.abspath(mpas_model_path)
        config.set('paths', 'mpas_model', mpas_model_path, user=True)

    config.set('test_case', 'steps_to_run', ' '.join(test_case.steps_to_run))

    # write out the config file
    test_case_config = '{}.cfg'.format(test_case.name)
    test_case.config_filename = test_case_config
    with open(os.path.join(test_case_dir, test_case_config), 'w') as f:
        config.write(f)

    if len(cached_steps) > 0 and cached_steps[0] == '_all':
        cached_steps = list(test_case.steps.keys())
    if len(cached_steps) > 0:
        print_steps = ' '.join(cached_steps)
        print(f'    steps with cached outputs: {print_steps}')
    for step_name in cached_steps:
        test_case.steps[step_name].cached = True

    # iterate over steps
    for step in test_case.steps.values():
        # make the step directory if it doesn't exist
        step_dir = os.path.join(work_dir, step.path)
        try:
            os.makedirs(step_dir)
        except OSError:
            pass

        symlink(os.path.join(test_case_dir, test_case_config),
                os.path.join(step_dir, test_case_config))

        step.work_dir = step_dir
        step.base_work_dir = work_dir
        step.config_filename = test_case_config
        step.config = config

        # set up the step
        step.setup()

        # process input, output, namelist and streams files
        step.process_inputs_and_outputs()

    # wait until we've set up all the steps before pickling because steps may
    # need other steps to be set up
    for step in test_case.steps.values():

        # pickle the test case and step for use at runtime
        pickle_filename = os.path.join(step.work_dir, 'step.pickle')
        with open(pickle_filename, 'wb') as handle:
            pickle.dump((test_case, step), handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    # pickle the test case and step for use at runtime
    pickle_filename = os.path.join(test_case.work_dir, 'test_case.pickle')
    with open(pickle_filename, 'wb') as handle:
        test_suite = {'name': 'test_case',
                      'test_cases': {test_case.path: test_case},
                      'work_dir': test_case.work_dir}
        pickle.dump(test_suite, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if 'LOAD_COMPASS_ENV' in os.environ:
        script_filename = os.environ['LOAD_COMPASS_ENV']
        # make a symlink to the script for loading the compass conda env.
        symlink(script_filename, os.path.join(test_case_dir,
                                              'load_compass_env.sh'))

    if machine is not None:
        max_cores, max_of_min_cores = _get_required_cores({path: test_case})
        write_job_script(config, machine, max_cores, max_of_min_cores,
                         test_case_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Set up one or more test cases', prog='compass setup')
    parser.add_argument("-t", "--test", dest="test",
                        help="Relative path for a test case to set up",
                        metavar="PATH")
    parser.add_argument("-n", "--case_number", nargs='+', dest="case_num",
                        type=str,
                        help="Case number(s) to setup, as listed from "
                             "'compass list'. Can be a space-separated"
                             "list of case numbers.  A suffix 'c' indicates"
                             "that all steps in the test should use cached"
                             "outputs.", metavar="NUM")
    parser.add_argument("-f", "--config_file", dest="config_file",
                        help="Configuration file for test case setup",
                        metavar="FILE")
    parser.add_argument("-m", "--machine", dest="machine",
                        help="The name of the machine for loading machine-"
                             "related config options", metavar="MACH")
    parser.add_argument("-w", "--work_dir", dest="work_dir",
                        help="If set, case directories are created in "
                             "work_dir rather than the current directory.",
                        metavar="PATH")
    parser.add_argument("-b", "--baseline_dir", dest="baseline_dir",
                        help="Location of baselines that can be compared to",
                        metavar="PATH")
    parser.add_argument("-p", "--mpas_model", dest="mpas_model",
                        help="The path to the build of the MPAS model for the "
                             "core.",
                        metavar="PATH")
    parser.add_argument("--suite_name", dest="suite_name", default="custom",
                        help="The name to use for the 'custom' test suite"
                             "containing all setup test cases.",
                        metavar="SUITE")
    parser.add_argument("--cached", dest="cached", nargs='+',
                        help="A list of steps in the test case supplied with"
                             "--test that should use cached outputs, or "
                             "'_all' if all steps should be cached",
                        metavar="STEP")

    args = parser.parse_args(sys.argv[2:])
    cached = None
    if args.test is None:
        tests = None
    else:
        tests = [args.test]
        if args.cached is not None:
            cached = [args.cached]
    setup_cases(tests=tests, numbers=args.case_num,
                config_file=args.config_file, machine=args.machine,
                work_dir=args.work_dir, baseline_dir=args.baseline_dir,
                mpas_model_path=args.mpas_model, suite_name=args.suite_name,
                cached=cached)


def _get_required_cores(test_cases):
    """ Get the maximum number of target cores and the max of min cores """

    max_cores = 0
    max_of_min_cores = 0
    for test_case in test_cases.values():
        for step_name in test_case.steps_to_run:
            step = test_case.steps[step_name]
            if step.ntasks is None:
                raise ValueError(
                    f'The number of tasks (ntasks) was never set for '
                    f'{test_case.path} step {step_name}')
            if step.cpus_per_task is None:
                raise ValueError(
                    f'The number of CPUs per task (cpus_per_task) was never '
                    f'set for {test_case.path} step {step_name}')
            cores = step.cpus_per_task*step.ntasks
            min_cores = step.min_cpus_per_task*step.min_tasks
            max_cores = max(max_cores, cores)
            max_of_min_cores = max(max_of_min_cores, min_cores)

    return max_cores, max_of_min_cores


def _get_basic_config(config_file, machine, mpas_model_path, mpas_core):
    """
    Get a base config parser for the machine and MPAS core but not a specific
    test
    """
    config = CompassConfigParser()

    if config_file is not None:
        config.add_user_config(config_file)

    # start with default compass config options
    config.add_from_package('compass', 'default.cfg')

    # add the E3SM config options from mache
    if machine is not None:
        config.add_from_package('mache.machines', f'{machine}.cfg')

    # add the compass machine config file
    if machine is None:
        machine = 'default'
    config.add_from_package('compass.machines', f'{machine}.cfg')

    if 'COMPASS_BRANCH' in os.environ:
        compass_branch = os.environ['COMPASS_BRANCH']
        config.set('paths', 'compass_branch', compass_branch)
    else:
        config.set('paths', 'compass_branch', os.getcwd())

    # add the config options for the MPAS core
    config.add_from_package(f'compass.{mpas_core}', f'{mpas_core}.cfg')

    # set the mpas_model path from the command line if provided
    if mpas_model_path is not None:
        mpas_model_path = os.path.abspath(mpas_model_path)
        config.set('paths', 'mpas_model', mpas_model_path, user=True)

    return config

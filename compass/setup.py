import argparse
import sys
import configparser
import os
import pickle

import compass.testcases
from compass.config import add_config, ensure_absolute_paths
from compass.io import symlink
from compass.testcase import generate_run
from compass import provenance


def setup_cases(tests=None, numbers=None, config_file=None, machine=None,
                work_dir=None, baseline_dir=None):
    """
    Set up one or more test cases

    Parameters
    ----------
    tests : list of str, optional
        Relative paths for a test cases to set up

    numbers : list of int, optional
        Case numbers to setup, as listed from ``compass list``

    config_file : str, optional
        Configuration file with custom options for setting up and running test
        cases

    machine : str, optional
        The name of one of the machines with defined config options, which can
        be listed with ``compass list --machines``

    work_dir : str, optional
        A directory that will serve as the base for creating case directories

    baseline_dir : str, optional
        Location of baseslines that can be compared to

    Returns
    -------
    testcases : dict
        A dictionary of information about each testcase, with the relative path
        in the work directory as keys
    """

    if config_file is None and machine is None:
        raise ValueError('At least one of config_file and machine is needed.')

    if tests is None and numbers is None:
        raise ValueError('At least one of tests or numbers is needed.')

    if work_dir is None:
        work_dir = os.getcwd()

    all_testcases = compass.testcases.collect()
    testcases = dict()
    if numbers is not None:
        keys = list(all_testcases)
        for number in numbers:
            if number >= len(keys):
                raise ValueError('test number {} is out of range.  There are '
                                 'only {} tests.'.format(number, len(keys)))
            path = keys[number]
            testcases[path] = all_testcases[path]

    if tests is not None:
        for path in tests:
            if path not in all_testcases:
                raise ValueError('Testcase with path {} is not in '
                                 'testcases'.format(path))
            testcases[path] = all_testcases[path]

    provenance.write(work_dir, testcases)

    print('Setting up testcases:')
    for path, testcase in testcases.items():
        setup_case(path, testcase, config_file, machine, work_dir,
                   baseline_dir)
    return testcases


def setup_case(path, testcase, config_file, machine, work_dir, baseline_dir):
    """
    Set up one or more test cases

    Parameters
    ----------
    path : str
        Relative path for a test cases to set up

    testcase : dict
        A dictionary describing the testcase

    config_file : str
        Configuration file with custom options for setting up and running test
        cases

    machine : str
        The name of one of the machines with defined config options, which can
        be listed with ``compass list --machines``

    work_dir : str
        A directory that will serve as the base for creating case directories

    baseline_dir : str
        Location of baseslines that can be compared to
    """

    print('  {}'.format(path))

    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation())

    # start with default compass config options
    add_config(config, 'compass', 'default.cfg')

    # add the machine config file
    if machine is None:
        machine = 'default'
    add_config(config, 'compass.machines', '{}.cfg'.format(machine))

    # add the config options for the core
    core = testcase['core']
    add_config(config, 'compass.{}'.format(core), '{}.cfg'.format(core))

    # add the config options for the configuration (if defined)
    configuration = testcase['configuration']
    add_config(config, 'compass.{}.tests.{}'.format(core, configuration),
               '{}.cfg'.format(configuration), exception=False)

    testcase_dir = os.path.join(work_dir, path)
    try:
        os.makedirs(testcase_dir)
    except OSError:
        pass
    testcase['work_dir'] = testcase_dir
    testcase['base_work_dir'] = work_dir

    # add config options specific to the testcase
    if testcase['configure'] is not None:
        configure = getattr(sys.modules[testcase['module']],
                            testcase['configure'])
        configure(testcase, config)

    # add the custom config file last, so these options are the defaults
    if config_file is not None:
        config.read(config_file)

    # add the baseline directory for this testcase
    if baseline_dir is not None:
        baseline_root = os.path.join(baseline_dir, path)
        config.set('paths', 'baseline_dir', baseline_root)

    # make sure all paths in the paths, namelists and streams sections are
    # absolute paths
    ensure_absolute_paths(config)

    # write out the config file
    testcase_config = '{}.cfg'.format(testcase['name'])
    testcase['config'] = testcase_config
    testcase_config = os.path.join(testcase_dir, testcase_config)
    with open(testcase_config, 'w') as f:
        config.write(f)

    # iterate over steps
    for step in testcase['steps'].values():
        # make the step directory if it doesn't exist
        step_dir = os.path.join(work_dir, step['path'])
        try:
            os.makedirs(step_dir)
        except OSError:
            pass

        testcase_config = '{}.cfg'.format(testcase['name'])
        symlink(os.path.join('..', testcase_config),
                os.path.join(step_dir, testcase_config))

        step['work_dir'] = step_dir
        step['base_work_dir'] = work_dir
        step['config'] = testcase_config

        # set up the step
        setup = getattr(sys.modules[step['module']], step['setup'])
        setup(step, config)

        # write a run script for each step
        _write_run(step, 'step.template')

    # write a run script for each testcase
    _write_run(testcase, 'testcase.template')


def main():
    parser = argparse.ArgumentParser(
        description='Set up one or more test cases')
    parser.add_argument("-t", "--test", dest="test",
                        help="Relative path for a test case to set up",
                        metavar="PATH")
    parser.add_argument("-n", "--case_number", nargs='+', dest="case_num",
                        type=int,
                        help="Case number(s) to setup, as listed from "
                             "'compass list'. Can be a space-separated"
                             "list of case numbers.", metavar="NUM")
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
    args = parser.parse_args(sys.argv[2:])
    if args.test is None:
        tests = None
    else:
        tests = [args.test]
    setup_cases(tests=tests, numbers=args.case_num,
                config_file=args.config_file, machine=args.machine,
                work_dir=args.work_dir, baseline_dir=args.baseline_dir)


def _write_run(test, template_name):
    """pickle the test/step info and write the run script"""

    # if compass/__init__.py exists, we're using a local version of the compass
    # package and we'll want to link to that in the tests and steps
    compass_path = os.path.join(os.getcwd(), 'compass')
    if os.path.exists(os.path.join(compass_path, '__init__.py')):
        symlink(compass_path, os.path.join(test['work_dir'], 'compass'))

    # pickle the test or step dictionary for use at runtime
    pickle_file = os.path.join(test['work_dir'],
                               '{}.pickle'.format(test['name']))
    with open(pickle_file, 'wb') as handle:
        pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # write a run script
    generate_run(test, template_name)

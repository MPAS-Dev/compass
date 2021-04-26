import argparse
import sys
import configparser
import os
import pickle

from compass.mpas_cores import get_mpas_cores
from compass.config import add_config, ensure_absolute_paths
from compass.io import symlink
from compass import provenance


def setup_cases(tests=None, numbers=None, config_file=None, machine=None,
                work_dir=None, baseline_dir=None, mpas_model_path=None):
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

    mpas_model_path : str, optional
        The relative or absolute path to the root of a branch where the MPAS
        model has been built

    Returns
    -------
    test_cases : dict of compass.TestCase
        A dictionary of test cases, with the relative path in the work
        directory as keys
    """

    if config_file is None and machine is None:
        raise ValueError('At least one of config_file and machine is needed.')

    if tests is None and numbers is None:
        raise ValueError('At least one of tests or numbers is needed.')

    if work_dir is None:
        work_dir = os.getcwd()

    mpas_cores = get_mpas_cores()

    all_test_cases = dict()
    for mpas_core in mpas_cores:
        for test_group in mpas_core.test_groups.values():
            for test_case in test_group.test_cases.values():
                all_test_cases[test_case.path] = test_case

    test_cases = dict()
    if numbers is not None:
        keys = list(all_test_cases)
        for number in numbers:
            if number >= len(keys):
                raise ValueError('test number {} is out of range.  There are '
                                 'only {} tests.'.format(number, len(keys)))
            path = keys[number]
            test_cases[path] = all_test_cases[path]

    if tests is not None:
        for path in tests:
            if path not in all_test_cases:
                raise ValueError('Test case with path {} is not in '
                                 'test_cases'.format(path))
            test_cases[path] = all_test_cases[path]

    provenance.write(work_dir, test_cases)

    print('Setting up test cases:')
    for path, test_case in test_cases.items():
        setup_case(path, test_case, config_file, machine, work_dir,
                   baseline_dir, mpas_model_path)

    return test_cases


def setup_case(path, test_case, config_file, machine, work_dir, baseline_dir,
               mpas_model_path):
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
        Location of baseslines that can be compared to

    mpas_model_path : str
        The relative or absolute path to the root of a branch where the MPAS
        model has been built
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

    # add the config options for the MPAS core
    mpas_core = test_case.mpas_core.name
    add_config(config, 'compass.{}'.format(mpas_core),
               '{}.cfg'.format(mpas_core))

    # add the config options for the configuration (if defined)
    test_group = test_case.test_group.name
    add_config(config, 'compass.{}.tests.{}'.format(mpas_core, test_group),
               '{}.cfg'.format(test_group), exception=False)

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

    # add the custom config file last, so these options are the defaults
    if config_file is not None:
        config.read(config_file)

    # add the baseline directory for this test case
    if baseline_dir is not None:
        baseline_root = os.path.join(baseline_dir, path)
        config.set('paths', 'baseline_dir', baseline_root)

    # set the mpas_model path from the command line if provided
    if mpas_model_path is not None:
        config.set('paths', 'mpas_model', mpas_model_path)

    # make sure all paths in the paths, namelists and streams sections are
    # absolute paths
    ensure_absolute_paths(config)

    # write out the config file
    test_case_config = '{}.cfg'.format(test_case.name)
    test_case.config_filename = test_case_config
    with open(os.path.join(test_case_dir, test_case_config), 'w') as f:
        config.write(f)

    # iterate over steps
    for step in test_case.steps.values():
        # make the step directory if it doesn't exist
        step_dir = os.path.join(work_dir, step.path)
        try:
            os.makedirs(step_dir)
        except OSError:
            pass

        symlink(os.path.join('..', test_case_config),
                os.path.join(step_dir, test_case_config))

        step.work_dir = step_dir
        step.base_work_dir = work_dir
        step.config_filename = test_case_config
        step.config = config

        # set up the step
        step.setup()

        # process input, output, namelist and streams files
        step.process_inputs_and_outputs()

        # write a run script for each step
        _link_compass(step.work_dir)

        # pickle the test case and step for use at runtime
        pickle_filename = os.path.join(step.work_dir, 'step.pickle')
        with open(pickle_filename, 'wb') as handle:
            pickle.dump((test_case, step), handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    # write a run script for each step
    _link_compass(test_case.work_dir)

    # pickle the test case and step for use at runtime
    pickle_filename = os.path.join(test_case.work_dir, 'test_case.pickle')
    with open(pickle_filename, 'wb') as handle:
        pickle.dump(test_case, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if 'LOAD_COMPASS_ENV' in os.environ:
        script_filename = os.environ['LOAD_COMPASS_ENV']
        # make a symlink to the script for loading the compass conda env.
        symlink(script_filename, os.path.join(work_dir, 'load_compass_env.sh'))


def main():
    parser = argparse.ArgumentParser(
        description='Set up one or more test cases', prog='compass setup')
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
    parser.add_argument("-p", "--mpas_model", dest="mpas_model",
                        help="The path to the build of the MPAS model for the "
                             "core.",
                        metavar="PATH")

    args = parser.parse_args(sys.argv[2:])
    if args.test is None:
        tests = None
    else:
        tests = [args.test]
    setup_cases(tests=tests, numbers=args.case_num,
                config_file=args.config_file, machine=args.machine,
                work_dir=args.work_dir, baseline_dir=args.baseline_dir,
                mpas_model_path=args.mpas_model)


def _link_compass(work_dir):
    # if compass/__init__.py exists, we're using a local version of the compass
    # package and we'll want to link to that in the tests and steps
    compass_path = os.path.join(os.getcwd(), 'compass')
    if os.path.exists(os.path.join(compass_path, '__init__.py')):
        symlink(compass_path, os.path.join(work_dir, 'compass'))

import os
import sys

from compass.examples import tests as example_tests
from compass.ocean import tests as ocean_tests


def collect():
    """
    Get dictionary of all test cases

    Returns
    -------
    testcases : dict
        A dictionary of all test cases
    """

    testcase_list = list()

    for tests in [example_tests, ocean_tests]:
        testcase_list.extend(tests.collect())

    validate(testcase_list)

    testcases = dict()
    for test in testcase_list:
        if test['path'] in testcases:
            raise ValueError('A test with this path has already been added: '
                             '{}'.format(test['path']))
        # set relative paths for steps
        for step in test['steps'].values():
            path = os.path.join(test['core'], test['configuration'],
                                test['subdir'], step['subdir'])
            step['path'] = path
        testcases[test['path']] = test

    return testcases


def validate(testcases):
    """
    Validate the dictionary of test cases to make sure the required dictionary
    entries, modules and functions are present

    Parameters
    ----------
    testcases : list
        A list of all test cases
    """

    for test in testcases:
        for key in ['path', 'name', 'module', 'configure', 'run', 'steps',
                    'description', 'subdir', 'core', 'configuration']:
            if key not in test:
                raise ValueError('a test is missing the "{}" key: {}'.format(
                    key, test))

        if not test['module'] in sys.modules:
            raise ValueError('test {} has a module {} that could not be '
                             'found'.format(test['path'], test['module']))
        module = sys.modules[test['module']]
        for key in ['configure', 'run']:
            if test[key] is not None and not hasattr(module, test[key]):
                raise ValueError('test {}: could not find function '
                                 '{}.{}()'.format(test['path'], test['module'],
                                                  test[key]))

        for step in test['steps'].values():
            for key in ['name', 'module', 'setup', 'run', 'subdir']:
                if key not in step:
                    raise ValueError('a step in {} is missing the "{}" key: '
                                     '{}'.format(test['path'],  key, step))

            if step['module'] not in sys.modules:
                raise ValueError('step {}/{} has a module {} that could not '
                                 'be found'.format(test['path'],
                                                   step['subdir'],
                                                   step['module']))
            module = sys.modules[step['module']]
            for key in ['setup', 'run']:
                if not hasattr(module, step[key]):
                    raise ValueError('step {}/{}: could not find function '
                                     '{}.{}()'.format(test['path'],
                                                      step['subdir'],
                                                      step['module'],
                                                      step[key]))

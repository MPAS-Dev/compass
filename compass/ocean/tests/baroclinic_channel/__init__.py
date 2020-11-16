from compass.ocean.tests.baroclinic_channel import decomp_test, default, \
    restart_test, rpe_test, threads_test


def collect():
    """
    Get a list of testcases in this configuration

    Returns
    -------
    testcases : list
        A list of tests within this configuration
    """
    testcases = list()
    for resolution in ['1km', '4km', '10km']:
        for test in [rpe_test]:
            testcases.append(test.collect(resolution=resolution))
    for resolution in ['10km']:
        for test in [decomp_test, default, restart_test, threads_test]:
            testcases.append(test.collect(resolution=resolution))

    return testcases


def configure(testcase, config):
    """
    Modify the configuration options for this testcase.

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this testcase from the ``collect()``
        function

    config : configparser.ConfigParser
        Configuration options for this testcase, a combination of the defaults
        for the machine, core and configuration
    """
    resolution = testcase['resolution']
    res_params = {'10km': {'nx': 16,
                           'ny': 50,
                           'dc': 10e3},
                  '4km': {'nx': 40,
                          'ny': 126,
                          'dc': 4e3},
                  '1km': {'nx': 160,
                          'ny': 500,
                          'dc': 1e3}}

    if resolution not in res_params:
        raise ValueError('Unsupported resolution {}. Supported values are: '
                         '{}'.format(resolution, list(res_params)))
    res_params = res_params[resolution]
    for param in res_params:
        config.set('baroclinic_channel', param, '{}'.format(res_params[param]))

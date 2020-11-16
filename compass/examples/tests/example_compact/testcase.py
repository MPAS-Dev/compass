from compass.examples.tests.example_compact import step1, step2

from compass.testcase import get_default


# a function for setting up any of the tests in the "example_compact"
# configuration. This function is intended to collect information on all
# available test case within the configuration.
def collect(module, description, resolution):
    """
    Get a dictionary of testcase properties

    Parameters
    ----------
    module : str
        The name of the module for the testcase

    description : str
        The description (long name) of the testcase

    resolution : {'1km', '2km'}
        The resolution of the mesh

    Returns
    -------
    testcase : list
        A list of properties of this test case, including its steps
    """
    # the name of the testcase is the last part of the Python module (the
    # folder it's in, so "test1" or "test2" in the "example_compact"
    # configuration
    name = module.split('.')[-1]
    # A subdirectory for the testcase after setup.  This can be anything that
    # will ensure that the testcase ends up in a unique directory
    subdir = '{}/{}'.format(resolution, name)
    # make a dictionary of steps for this testcase by calling each step's
    # "collect" function
    steps = dict()
    for step_module in [step1, step2]:
        step = step_module.collect(resolution)
        steps[step['name']] = step

    # get some default information for the testcase
    testcase = get_default(module, description, steps, subdir=subdir)
    # add any parameters or other information you would like to have when you
    # are setting up or running the testcase or its steps
    testcase['resolution'] = resolution

    return testcase

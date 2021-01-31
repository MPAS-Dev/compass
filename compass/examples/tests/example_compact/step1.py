import xarray

from mpas_tools.io import write_netcdf

from compass.io import add_input_file, add_output_file


# this function is used to define the step by adding to or modifying the
# information in the "step" dictionary.  You can add or override options in
# the dictionary directly, or you can call functions to add namelist files,
# namelist options from a dictionary, streams files from a file or template,
# input and output files.  At this stage, you cannot set config options so
# information added to "step" should typically be data not available for users
# to change at runtime.  The "testcase" dictionary is here to get information,
# e.g. name or location of the test case, but should not be altered.
def collect(testcase, step):
    """
    Update the dictionary of step properties

    Parameters
    ----------
    testcase : dict
        A dictionary of properties of this test case, which should not be
        modified here

    step : dict
        A dictionary of properties of this step, which can be updated
    """
    # the "testcase" and "step" dictionaries will contain some information that
    # is either added by the framework or passed in to "add_step" as a keyword
    # argument.  In this case, we get the name of the test case that was added
    # by the framework and the resolution, which was passed as a keyword
    # argument to "add_step".
    testcase_name = testcase['name']
    resolution = step['resolution']

    # We will set these 5 options in "step" unless they were already set to
    # something else in a call to "add_step" within the test case  These
    # options are all related to the resources that the test case is allowed
    # to use, but something similar could be done for any data you want ot add
    # to "step"
    defaults = dict(cores=1, min_cores=1, max_memory=1000, max_disk=1000,
                    threads=1)
    for key, value in defaults.items():
        step.setdefault(key, value)

    # Sometimes it is handy to use dictionaries or nested dictionaries to
    # define sets of parameters for different test cases, resolutions, etc.
    # that share the same step.  In the example, we use different file names
    # for the input file for the two different test cases "test1" and "test2"
    targets = {'test1': 'particle_regions.151113.nc',
               'test2': 'layer_depth.80Layer.180619.nc'}

    if testcase_name not in targets:
        raise ValueError('Unsupported test case name {}. Supported test cases '
                         'are: {}'.format(testcase, list(targets)))
    target = targets[testcase_name]

    # one of the required parts of setup is to define any input files from
    # other steps or test cases that are required by this step, and any output
    # files that are produced by this step that might be used in other steps
    # or test cases.  This allows compass to determine dependencies between
    # test cases and their steps.  This can be done either in "collect" or
    # "setup".

    # we will download the file "target" to the initial_condition_database
    # from the data server (we don't have to give a URL) if it hasn't already
    # been downloaded and then make a local link called "input_file.nc" to it
    # in the step.  This won't happen yet (because we haven't decided if we're
    # going to set up this step yet) but we're storing the information about
    # what to do if we do decide we will set up this step later.
    add_input_file(step, filename='input_file.nc', target=target,
                   database='initial_condition_database')

    # effectively, we're promising that this step creates a file
    # "output_file.nc" that other steps could use as an input if they want to
    add_output_file(step, filename='output_file.nc')


# This function gets called to set up the test case.  It can add input, output,
# namelist an streams files, just like "collect".  It can also do things that
# depend on config options for the test case (but shouldn't set config options
# because these might mess up other steps that share the same config options).
# The function must take only the "step" and "config" arguments, so any
# information you need should be added to "step" if it is not available in
# "config"
def setup(step, config):
    """
    Set up the test case in the work directory, including downloading any
    dependencies

    Parameters
    ----------
    step : dict
        A dictionary of properties of this step

    config : configparser.ConfigParser
        Configuration options for this step, a combination of the defaults for
        the machine, core, configuration and test case
    """
    resolution = step['resolution']
    # This is a way to handle a few parameters that are specific to different
    # test cases or resolutions, all of which can be handled by this function
    res_params = {'1km': {'parameter4': 1.0,
                          'parameter5': 500},
                  '2km': {'parameter4': 2.0,
                          'parameter5': 250}}

    # copy the appropriate parameters into the step dict for use in run
    if resolution not in res_params:
        raise ValueError('Unsupported resolution {}. Supported values are: '
                         '{}'.format(resolution, list(res_params)))
    res_params = res_params[resolution]

    # add the parameters for this resolution to the step dictionary so they
    # are available to the run() function
    for param in res_params:
        step[param] = res_params[param]


# This function runs the step.  It must take the 4 arguments "step",
# "test_suite", "config" and "logger".  The step should then perform the main
# "work" of the step such as running the model or doing other computations.
def run(step, test_suite, config, logger):
    """
    Run this step of the test case

    Parameters
    ----------
    step : dict
        A dictionary of properties of this step

    test_suite : dict
        A dictionary of properties of the test suite

    config : configparser.ConfigParser
        Configuration options for this test case

    logger : logging.Logger
        A logger for output from the step
    """
    section = config['example_compact']
    parameter1 = section.getfloat('parameter1')
    parameter2 = section.getboolean('parameter2')
    testcase = step['testcase']

    # we just read in the input file and write it out to the output file
    ds = xarray.open_dataset('input_file.nc')
    write_netcdf(ds, 'output_file.nc')

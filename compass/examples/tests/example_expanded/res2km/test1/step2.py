import xarray

from mpas_tools.io import write_netcdf

from compass.io import add_input_file, add_output_file


# see step1.py for a detailed explanation
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
    step['resolution'] = '2km'

    defaults = dict(cores=1, min_cores=1, max_memory=1000, max_disk=1000,
                    threads=1)
    for key, value in defaults.items():
        step.setdefault(key, value)

    # this time, we won't download the file, we'll get it from step1
    add_input_file(step, filename='input_file.nc',
                   target='../step1/output_file.nc')

    add_output_file(step, filename='output_file.nc')


# this step doesn't have anything to do in the setup function, so we skip it


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
    # Again, we just read in the input and write out the output
    ds = xarray.open_dataset('input_file.nc')
    write_netcdf(ds, 'output_file.nc')

import os
import xarray

from mpas_tools.scrip.from_mpas import scrip_from_mpas

from compass.io import symlink, add_input_file


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
    defaults = dict(cores=1, min_cores=1, max_memory=1000, max_disk=1000,
                    threads=1)
    for key, value in defaults.items():
        step.setdefault(key, value)

    add_input_file(step, filename='README', target='../README')
    add_input_file(step, filename='restart.nc',
                   target='../{}'.format(step['restart_filename']))

    # for now, we won't define any outputs because they include the mesh short
    # name, which is not known at setup time.  Currently, this is safe because
    # no other steps depend on the outputs of this one.


def run(step, test_suite, config, logger):
    """
    Run this step of the testcase

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
    with xarray.open_dataset('restart.nc') as ds:
        mesh_short_name = ds.attrs['MPAS_Mesh_Short_Name']
        mesh_prefix = ds.attrs['MPAS_Mesh_Prefix']
        prefix = 'MPAS_Mesh_{}'.format(mesh_prefix)
        creation_date = ds.attrs['{}_Version_Creation_Date'.format(prefix)]

    try:
        os.makedirs('../assembled_files/inputdata/ocn/mpas-o/{}'.format(
            mesh_short_name))
    except OSError:
        pass

    with_ice_shelf_cavities = step['with_ice_shelf_cavities']

    if with_ice_shelf_cavities:
        nomask_str = '.nomask'
    else:
        nomask_str = ''

    restart_filename = os.path.abspath(
        os.path.join('..', step['restart_filename']))

    # command line execution
    scrip_filename = 'ocean.{}{}.scrip.{}.nc'.format(
        mesh_short_name,  nomask_str, creation_date)

    scrip_from_mpas(restart_filename, scrip_filename)

    symlink('../../../../../scrip/{}'.format(scrip_filename),
            '../assembled_files/inputdata/ocn/mpas-o/{}/{}'.format(
                mesh_short_name, scrip_filename))

    if with_ice_shelf_cavities:
        scrip_mask_filename = 'ocean.{}.mask.scrip.{}.nc'.format(
            mesh_short_name, creation_date)
        scrip_from_mpas(restart_filename, scrip_mask_filename,
                        useLandIceMask=True)

        symlink(
            '../../../../../scrip/{}'.format(
                scrip_mask_filename),
            '../assembled_files/inputdata/ocn/mpas-o/{}/{}'.format(
                mesh_short_name, scrip_mask_filename))

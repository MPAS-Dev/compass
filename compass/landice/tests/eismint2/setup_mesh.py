import xarray

from mpas_tools.planar_hex import make_planar_hex_mesh
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.logging import check_call

from compass.io import add_output_file
from compass.model import make_graph_file


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
    defaults = dict(cores=1, min_cores=1, max_memory=8000, max_disk=8000,
                    threads=1)
    for key, value in defaults.items():
        step.setdefault(key, value)

    add_output_file(step, filename='graph.info')
    add_output_file(step, filename='landice_grid.nc')


# no setup function is needed


def run(step, test_suite, config, logger):
    """
    Run this step of the test case

    Parameters
    ----------
    step : dict
        A dictionary of properties of this step from the ``collect()``
        function, with modifications from the ``setup()`` function.

    test_suite : dict
        A dictionary of properties of the test suite

    config : configparser.ConfigParser
        Configuration options for this test case, a combination of the defaults
        for the machine, core and configuration

    logger : logging.Logger
        A logger for output from the step
   """
    section = config['eismint2']

    nx = section.getint('nx')
    ny = section.getint('ny')
    dc = section.getfloat('dc')

    dsMesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc, nonperiodic_x=False,
                                  nonperiodic_y=False)

    dsMesh = convert(dsMesh, logger=logger)
    write_netcdf(dsMesh, 'mpas_grid.nc')
    dsMesh.close()

    radius = section.get('radius')
    args = ['define_cullMask.py',
            '-f', 'mpas_grid.nc',
            '-m', 'radius',
            '-d', radius]

    check_call(args, logger)

    dsMesh = xarray.open_dataset('mpas_grid.nc')
    dsMesh = cull(dsMesh, logger=logger)
    dsMesh = convert(dsMesh, logger=logger)
    write_netcdf(dsMesh, 'mpas_grid2.nc')

    levels = section.get('levels')
    args = ['create_landice_grid_from_generic_MPAS_grid.py',
            '-i', 'mpas_grid2.nc',
            '-o', 'landice_grid.nc',
            '-l', levels,
            '--thermal',
            '--beta']

    check_call(args, logger)

    make_graph_file(mesh_filename='landice_grid.nc',
                    graph_filename='graph.info')

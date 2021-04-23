from netCDF4 import Dataset as NetCDFFile

from mpas_tools.planar_hex import make_planar_hex_mesh
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.logging import check_call

from compass.model import make_graph_file
from compass.step import Step


class SetupMesh(Step):
    """
    A step for creating a mesh and initial condition for enthalpy benchmark
    test cases
    """
    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='setup_mesh')
        self.add_output_file(filename='graph.info')
        self.add_output_file(filename='landice_grid.nc')

    # no setup() method is needed

    def run(self):
        """
        Run this step of the test case
       """
        logger = self.logger
        section = self.config['enthalpy_benchmark']
        nx = section.getint('nx')
        ny = section.getint('ny')
        dc = section.getfloat('dc')
        levels = section.get('levels')

        dsMesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc, nonperiodic_x=True,
                                      nonperiodic_y=True)

        write_netcdf(dsMesh, 'grid.nc')

        dsMesh = cull(dsMesh, logger=logger)
        dsMesh = convert(dsMesh, logger=logger)
        write_netcdf(dsMesh, 'mpas_grid.nc')

        args = ['create_landice_grid_from_generic_MPAS_grid.py',
                '-i', 'mpas_grid.nc',
                '-o', 'landice_grid.nc',
                '-l', levels,
                '--thermal']

        check_call(args, logger)

        make_graph_file(mesh_filename='landice_grid.nc',
                        graph_filename='graph.info')

        _setup_initial_conditions(section, 'landice_grid.nc')


def _setup_initial_conditions(section, filename):
    """ Add the initial conditions for enthalpy benchmark A """
    thickness = section.getfloat('thickness')
    basal_heat_flux = section.getfloat('basal_heat_flux')
    surface_air_temperature = section.getfloat('surface_air_temperature')
    temperature = section.getfloat('temperature')

    with NetCDFFile(filename, 'r+') as gridfile:
        thicknessVar = gridfile.variables['thickness']
        bedTopography = gridfile.variables['bedTopography']
        basalHeatFlux = gridfile.variables['basalHeatFlux']
        surfaceAirTemperature = gridfile.variables['surfaceAirTemperature']
        temperatureVar = gridfile.variables['temperature']

        thicknessVar[:] = thickness
        bedTopography[:] = 0
        basalHeatFlux[:] = basal_heat_flux
        surfaceAirTemperature[:] = surface_air_temperature
        temperatureVar[:] = temperature

import xarray
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.mesh.creation import build_planar_mesh

from compass.landice.mesh import build_cell_width
from compass.model import make_graph_file
from compass.step import Step


class Mesh(Step):
    """
    A step for creating a mesh and initial condition for greenland test cases

    Attributes
    ----------
    mesh_type : str
        The resolution or mesh type of the test case
    """
    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        mesh_type : str
            The resolution or mesh type of the test case
        """
        super().__init__(test_case=test_case, name='mesh')

        self.add_output_file(filename='graph.info')
        self.add_output_file(filename='GIS.nc')
        self.add_input_file(
            filename='greenland_1km_2020_04_20.epsg3413.icesheetonly.nc',
            target='greenland_1km_2020_04_20.epsg3413.icesheetonly.nc',
            database='')
        self.add_input_file(filename='greenland_2km_2020_04_20.epsg3413.nc',
                            target='greenland_2km_2020_04_20.epsg3413.nc',
                            database='')

    # no setup() method is needed

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        config = self.config
        section = config['high_res_GIS_mesh']

        logger.info('calling build_cell_width')
        cell_width, x1, y1, geom_points, geom_edges, floodMask = \
            build_cell_width(
                self, section_name='high_res_GIS_mesh',
                gridded_dataset='greenland_2km_2020_04_20.epsg3413.nc',
                flood_fill_start=[100, 700])
        logger.info('calling build_planar_mesh')
        build_planar_mesh(cell_width, x1, y1, geom_points,
                          geom_edges, logger=logger)
        dsMesh = xarray.open_dataset('base_mesh.nc')
        logger.info('culling mesh')
        dsMesh = cull(dsMesh, logger=logger)
        logger.info('converting to MPAS mesh')
        dsMesh = convert(dsMesh, logger=logger)
        logger.info('writing grid_converted.nc')
        write_netcdf(dsMesh, 'grid_converted.nc')

        levels = section.get('levels')
        logger.info('calling create_landice_grid_from_generic_MPAS_grid.py')
        args = ['create_landice_grid_from_generic_MPAS_grid.py',
                '-i', 'grid_converted.nc',
                '-o', 'gis_1km_preCull.nc',
                '-l', levels, '-v', 'glimmer']
        check_call(args, logger=logger)

        logger.info('calling interpolate_to_mpasli_grid.py')
        args = ['interpolate_to_mpasli_grid.py', '-s',
                'greenland_1km_2020_04_20.epsg3413.icesheetonly.nc', '-d',
                'gis_1km_preCull.nc', '-m', 'b', '-t']

        check_call(args, logger=logger)

        cullDistance = section.get('cull_distance')
        logger.info('calling define_cullMask.py')
        args = ['define_cullMask.py', '-f',
                'gis_1km_preCull.nc', '-m',
                'distance', '-d', cullDistance]

        check_call(args, logger=logger)

        dsMesh = xarray.open_dataset('gis_1km_preCull.nc')
        dsMesh = cull(dsMesh, logger=logger)
        write_netcdf(dsMesh, 'greenland_culled.nc')

        logger.info('Marking horns for culling')
        args = ['mark_horns_for_culling.py', '-f', 'greenland_culled.nc']
        check_call(args, logger=logger)

        logger.info('culling and converting')
        dsMesh = xarray.open_dataset('greenland_culled.nc')
        dsMesh = cull(dsMesh, logger=logger)
        dsMesh = convert(dsMesh, logger=logger)
        write_netcdf(dsMesh, 'greenland_dehorned.nc')

        logger.info('calling create_landice_grid_from_generic_MPAS_grid.py')
        args = ['create_landice_grid_from_generic_MPAS_grid.py', '-i',
                'greenland_dehorned.nc', '-o',
                'GIS.nc', '-l', levels, '-v', 'glimmer',
                '--beta', '--thermal', '--obs', '--diri']

        check_call(args, logger=logger)

        logger.info('calling interpolate_to_mpasli_grid.py')
        args = ['interpolate_to_mpasli_grid.py', '-s',
                'greenland_1km_2020_04_20.epsg3413.icesheetonly.nc',
                '-d', 'GIS.nc', '-m', 'b']
        check_call(args, logger=logger)

        logger.info('Marking domain boundaries dirichlet')
        args = ['mark_domain_boundaries_dirichlet.py',
                '-f', 'GIS.nc']
        check_call(args, logger=logger)

        logger.info('calling set_lat_lon_fields_in_planar_grid.py')
        args = ['set_lat_lon_fields_in_planar_grid.py', '-f',
                'GIS.nc', '-p', 'gis-gimp']
        check_call(args, logger=logger)

        logger.info('creating graph.info')
        make_graph_file(mesh_filename='GIS.nc',
                        graph_filename='graph.info')

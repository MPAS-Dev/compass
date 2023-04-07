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
    A step for creating a mesh and initial condition for thwaites test cases
    """
    def __init__(self, test_case):
        """
        Create the step
        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='mesh')

        self.add_output_file(filename='graph.info')
        self.add_output_file(filename='Thwaites.nc')
        self.add_input_file(filename='antarctica_8km_2020_10_20.nc',
                            target='antarctica_8km_2020_10_20.nc',
                            database='')
        self.add_input_file(filename='thwaites_minimal.geojson',
                            package='compass.landice.tests.thwaites',
                            target='thwaites_minimal.geojson',
                            database=None)
        self.add_input_file(filename='antarctica_1km_2020_10_20_ASE.nc',
                            target='antarctica_1km_2020_10_20_ASE.nc',
                            database='')

    # no setup() method is needed

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        config = self.config
        section = config['high_res_mesh']

        logger.info('calling build_cell_width')
        cell_width, x1, y1, geom_points, geom_edges, floodMask = \
            build_cell_width(
                self, section_name='high_res_mesh',
                gridded_dataset='antarctica_8km_2020_10_20.nc')
        logger.info('calling build_planar_mesh')
        build_planar_mesh(cell_width, x1, y1, geom_points,
                          geom_edges, logger=logger)
        ds_mesh = xarray.open_dataset('base_mesh.nc')
        logger.info('culling mesh')
        ds_mesh = cull(ds_mesh, logger=logger)
        logger.info('converting to MPAS mesh')
        ds_mesh = convert(ds_mesh, logger=logger)
        logger.info('writing grid_converted.nc')
        write_netcdf(ds_mesh, 'grid_converted.nc')

        levels = section.get('levels')

        logger.info('calling create_landice_grid_from_generic_MPAS_grid.py')
        args = ['create_landice_grid_from_generic_MPAS_grid.py',
                '-i', 'grid_converted.nc',
                '-o', 'ase_1km_preCull.nc',
                '-l', levels, '-v', 'glimmer']
        check_call(args, logger=logger)

        # This step uses a subset of the whole Antarctica dataset trimmed to
        # the Amundsen Sean Embayment, to speed up interpolation.
        # This could also be replaced with the full Antarctic Ice Sheet
        # dataset.
        logger.info('calling interpolate_to_mpasli_grid.py')
        args = ['interpolate_to_mpasli_grid.py', '-s',
                'antarctica_1km_2020_10_20_ASE.nc', '-d',
                'ase_1km_preCull.nc', '-m', 'b', '-t']

        check_call(args, logger=logger)

        # This step is only necessary if you wish to cull a certain
        # distance from the ice margin, within the bounds defined by
        # the GeoJSON file.
        cullDistance = section.get('cull_distance')
        if float(cullDistance) > 0.:
            logger.info('calling define_cullMask.py')
            args = ['define_cullMask.py', '-f',
                    'ase_1km_preCull.nc', '-m'
                    'distance', '-d', cullDistance]

            check_call(args, logger=logger)
        else:
            logger.info('cullDistance <= 0 in config file. '
                        'Will not cull by distance to margin. \n')

        # This step is only necessary because the GeoJSON region
        # is defined by lat-lon.
        logger.info('calling set_lat_lon_fields_in_planar_grid.py')
        args = ['set_lat_lon_fields_in_planar_grid.py', '-f',
                'ase_1km_preCull.nc', '-p', 'ais-bedmap2']

        check_call(args, logger=logger)

        logger.info('calling MpasMaskCreator.x')
        args = ['MpasMaskCreator.x', 'ase_1km_preCull.nc',
                'thwaites_mask.nc', '-f', 'thwaites_minimal.geojson']

        check_call(args, logger=logger)

        logger.info('culling to geojson file')
        ds_mesh = xarray.open_dataset('ase_1km_preCull.nc')
        thwaitesMask = xarray.open_dataset('thwaites_mask.nc')
        ds_mesh = cull(ds_mesh, dsInverse=thwaitesMask, logger=logger)
        write_netcdf(ds_mesh, 'thwaites_culled.nc')

        logger.info('Marking horns for culling')
        args = ['mark_horns_for_culling.py', '-f', 'thwaites_culled.nc']
        check_call(args, logger=logger)

        logger.info('culling and converting')
        ds_mesh = xarray.open_dataset('thwaites_culled.nc')
        ds_mesh = cull(ds_mesh, logger=logger)
        ds_mesh = convert(ds_mesh, logger=logger)
        write_netcdf(ds_mesh, 'thwaites_dehorned.nc')

        logger.info('calling create_landice_grid_from_generic_MPAS_grid.py')
        args = ['create_landice_grid_from_generic_MPAS_grid.py', '-i',
                'thwaites_dehorned.nc', '-o',
                'Thwaites.nc', '-l', levels, '-v', 'glimmer',
                '--beta', '--thermal', '--obs', '--diri']

        check_call(args, logger=logger)

        logger.info('calling interpolate_to_mpasli_grid.py')
        args = ['interpolate_to_mpasli_grid.py', '-s',
                'antarctica_1km_2020_10_20_ASE.nc',
                '-d', 'Thwaites.nc', '-m', 'b']
        check_call(args, logger=logger)

        logger.info('Marking domain boundaries dirichlet')
        args = ['mark_domain_boundaries_dirichlet.py',
                '-f', 'Thwaites.nc']
        check_call(args, logger=logger)

        logger.info('calling set_lat_lon_fields_in_planar_grid.py')
        args = ['set_lat_lon_fields_in_planar_grid.py', '-f',
                'Thwaites.nc', '-p', 'ais-bedmap2']
        check_call(args, logger=logger)

        logger.info('creating graph.info')
        make_graph_file(mesh_filename='Thwaites.nc',
                        graph_filename='graph.info')

import os

from mpas_tools.scrip.from_mpas import scrip_from_mpas

from compass.landice.mesh import (
    build_cell_width,
    build_mali_mesh,
    clean_up_after_interp,
    interp_gridded2mali,
)
from compass.model import make_graph_file
from compass.step import Step


class Mesh(Step):
    """
    A step for creating a mesh and initial condition for isunnguata
    sermia test cases

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
        self.add_output_file(filename='Isunnguata_Sermia.nc')
        self.add_input_file(
            filename='greenland_1km_2024_01_29.epsg3413.icesheetonly.nc',
            target='greenland_1km_2024_01_29.epsg3413.icesheetonly.nc',
            database='')
        self.add_input_file(filename='Isunnguata_Sermia.geojson',
                            package='compass.landice.tests.isunnguata_sermia',
                            target='Isunnguata_Sermia.geojson',
                            database=None)
        self.add_input_file(filename='greenland_2km_2024_01_29.epsg3413.nc',
                            target='greenland_2km_2024_01_29.epsg3413.nc',
                            database='')

    # no setup() method is needed

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        config = self.config
        section_name = 'mesh'
        section = config[section_name]
        mesh_name = 'Isunnguata_Sermia.nc'

        def _specified(value):
            return value is not None and str(value).strip().lower() not in [
                '', 'none']

        data_path = section.get('data_path', fallback=None)
        bedmachine_filename = section.get('bedmachine_filename', fallback=None)
        measures_filename = section.get('measures_filename', fallback=None)

        use_bedmachine_interp = _specified(data_path) and \
            _specified(bedmachine_filename)
        use_measures_interp = _specified(data_path) and \
            _specified(measures_filename)
        do_bespoke_interp = use_bedmachine_interp or use_measures_interp

        if use_bedmachine_interp:
            bedmachine_dataset = os.path.join(data_path, bedmachine_filename)
        else:
            bedmachine_dataset = None

        if use_measures_interp:
            measures_dataset = os.path.join(data_path, measures_filename)
        else:
            measures_dataset = None

        if not do_bespoke_interp:
            logger.info('Skipping optional bespoke interpolation because '
                        '`data_path` and interpolation filenames are '
                        'not specified in config.')

        logger.info('calling build_cell_width')
        cell_width, x1, y1, geom_points, geom_edges, floodMask = \
            build_cell_width(
                self, section_name=section_name,
                gridded_dataset='greenland_2km_2024_01_29.epsg3413.nc')

        build_mali_mesh(
            self, cell_width, x1, y1, geom_points, geom_edges,
            mesh_name=mesh_name, section_name=section_name,
            gridded_dataset='greenland_1km_2024_01_29.epsg3413.icesheetonly.nc',  # noqa
            projection='gis-gimp', geojson_file='Isunnguata_Sermia.geojson')

        if do_bespoke_interp:
            parallel_executable = config.get('parallel', 'parallel_executable')
            nProcs = str(self.cpus_per_task)

            logger.info('creating scrip file for destination mesh')
            dst_scrip_file = f"{mesh_name.split('.')[:-1][0]}_scrip.nc"
            scrip_from_mpas(mesh_name, dst_scrip_file)

            if use_bedmachine_interp:
                interp_gridded2mali(self, bedmachine_dataset, dst_scrip_file,
                                    parallel_executable, nProcs, mesh_name,
                                    'gis-gimp', variables='all')

            if use_measures_interp:
                measures_vars = ['observedSurfaceVelocityX',
                                 'observedSurfaceVelocityY',
                                 'observedSurfaceVelocityUncertainty']
                interp_gridded2mali(self, measures_dataset, dst_scrip_file,
                                    parallel_executable, nProcs, mesh_name,
                                    'gis-gimp', variables=measures_vars)

            clean_up_after_interp(mesh_name)

        logger.info('creating graph.info')
        make_graph_file(mesh_filename=mesh_name,
                        graph_filename='graph.info')

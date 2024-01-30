from compass.landice.mesh import build_cell_width, build_mali_mesh
from compass.model import make_graph_file
from compass.step import Step


class Mesh(Step):
    """
    A step for creating a mesh and initial condition for koge_bugt_s
    test cases

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
        super().__init__(test_case=test_case, name='mesh', cpus_per_task=128,
                         min_cpus_per_task=1)

        self.add_output_file(filename='graph.info')
        self.add_output_file(filename='Koge_Bugt_S.nc')
        self.add_input_file(
            filename='greenland_1km_2024_01_29.epsg3413.icesheetonly.nc',
            target='greenland_1km_2024_01_29.epsg3413.icesheetonly.nc',
            database='')
        self.add_input_file(filename='Koge_Bugt_S.geojson',
                            package='compass.landice.tests.koge_bugt_s',
                            target='Koge_Bugt_S.geojson',
                            database=None)
        self.add_input_file(filename='greenland_8km_2024_01_29.epsg3413.nc',
                            target='greenland_8km_2024_01_29.epsg3413.nc',
                            database='')

    # no setup() method is needed

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        mesh_name = 'Koge_Bugt_S.nc'
        section_name = 'mesh'

        logger.info('calling build_cell_width')
        cell_width, x1, y1, geom_points, geom_edges, floodMask = \
            build_cell_width(
                self, section_name=section_name,
                gridded_dataset='greenland_8km_2024_01_29.epsg3413.nc')

        build_mali_mesh(
            self, cell_width, x1, y1, geom_points, geom_edges,
            mesh_name=mesh_name, section_name=section_name,
            gridded_dataset='greenland_1km_2024_01_29.epsg3413.icesheetonly.nc',  # noqa
            projection='gis-gimp', geojson_file='Koge_Bugt_S.geojson',
            cores=self.cpus_per_task)

        logger.info('creating graph.info')
        make_graph_file(mesh_filename=mesh_name,
                        graph_filename='graph.info')

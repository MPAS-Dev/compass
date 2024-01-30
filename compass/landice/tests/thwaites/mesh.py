from compass.landice.mesh import build_cell_width, build_mali_mesh
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
        super().__init__(test_case=test_case, name='mesh', cpus_per_task=128,
                         min_cpus_per_task=1)

        self.add_output_file(filename='graph.info')
        self.add_output_file(filename='Thwaites.nc')
        self.add_input_file(filename='antarctica_8km_2024_01_29.nc',
                            target='antarctica_8km_2024_01_29.nc',
                            database='')
        self.add_input_file(filename='thwaites_minimal.geojson',
                            package='compass.landice.tests.thwaites',
                            target='thwaites_minimal.geojson',
                            database=None)
        self.add_input_file(filename='antarctica_1km_2024_01_29_ASE.nc',
                            target='antarctica_1km_2024_01_29_ASE.nc',
                            database='')

    # no setup() method is needed

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        mesh_name = 'Thwaites.nc'
        section_name = 'mesh'

        logger.info('calling build_cell_width')
        cell_width, x1, y1, geom_points, geom_edges, floodMask = \
            build_cell_width(
                self, section_name=section_name,
                gridded_dataset='antarctica_8km_2024_01_29.nc')

        build_mali_mesh(
            self, cell_width, x1, y1, geom_points, geom_edges,
            mesh_name=mesh_name, section_name=section_name,
            gridded_dataset='antarctica_1km_2024_01_29_ASE.nc',
            projection='ais-bedmap2', geojson_file='thwaites_minimal.geojson',
            cores=self.cpus_per_task)

        logger.info('creating graph.info')
        make_graph_file(mesh_filename=mesh_name,
                        graph_filename='graph.info')

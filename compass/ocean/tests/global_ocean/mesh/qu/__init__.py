import numpy as np
import time

from compass.ocean.tests.global_ocean.init import Init
from compass.ocean.tests.global_ocean.performance_test import PerformanceTest
from compass.ocean.tests.global_ocean.mesh.mesh import MeshStep


class QUInit(Init):
    """
    A test case for creating and initial condition on a global, quasi-uniform
    MPAS-Ocean mesh
    """
    def configure(self):
        """ Update the number of cores and min_cores for initial_state and
        forward steps """

        super().configure()
        set_qu_cores(self.config)
        if 'ssh_adjustment' in self.steps:
            dt, btr_dt = get_qu_dts(self.config)
            step = self.steps['ssh_adjustment']
            step.add_namelist_options({'config_dt': dt,
                                       'config_btr_dt': btr_dt})

    def run(self):
        if 'ssh_adjustment' in self.steps:
            dt, btr_dt = get_qu_dts(self.config)
            step = self.steps['ssh_adjustment']
            step.update_namelist_at_runtime(options={'config_dt': dt,
                                                     'config_btr_dt': btr_dt},
                                            out_name='namelist.ocean')
        super().run()


class QUPerformanceTest(PerformanceTest):
    """
    A test case for performing a short forward run with a global, quasi-uniform
    MPAS-Ocean mesh and initial condition to assess performance and compare
    with previous results
    """
    def configure(self):
        """ Update the number of cores and min_cores the forward step """

        super().configure()
        set_qu_cores(self.config)
        dt, btr_dt = get_qu_dts(self.config)
        step = self.steps['forward']
        step.add_namelist_options({'config_dt': dt,
                                   'config_btr_dt': btr_dt})

    def run(self):
        dt, btr_dt = get_qu_dts(self.config)
        step = self.steps['forward']
        step.update_namelist_at_runtime(options={'config_dt': dt,
                                                 'config_btr_dt': btr_dt},
                                        out_name='namelist.ocean')
        super().run()


class QUMeshStep(MeshStep):
    """
    A step for creating quasi-uniform meshes at a resolution given by a config
    option

    Attributes
    ----------
    resolution : float
        The resolution of the mesh in km
    """
    def __init__(self, test_case, mesh_name, with_ice_shelf_cavities):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case this step belongs to

        mesh_name : str
            The name of the mesh

        with_ice_shelf_cavities : bool
            Whether the mesh includes ice-shelf cavities
        """

        super().__init__(test_case, mesh_name, with_ice_shelf_cavities,
                         package=self.__module__,
                         mesh_config_filename='qu.cfg')

        self.resolution = None

    def setup(self):
        """
        Set up the test case in the work directory, including downloading any
        dependencies.
        """
        # get the these properties from the config options
        super().setup()
        config = self.config
        self.resolution = config.getfloat('global_ocean_qu', 'resolution')

    def build_cell_width_lat_lon(self):
        """
        Create cell width array for this mesh on a regular latitude-longitude
        grid

        Returns
        -------
        cellWidth : numpy.array
            m x n array of cell width in km

        lon : numpy.array
            longitude in degrees (length n and between -180 and 180)

        lat : numpy.array
            longitude in degrees (length m and between -90 and 90)
        """
        dlon = 10.
        dlat = dlon
        constantCellWidth = self.resolution

        nlat = int(180/dlat) + 1
        nlon = int(360/dlon) + 1
        lat = np.linspace(-90., 90., nlat)
        lon = np.linspace(-180., 180., nlon)

        cellWidth = constantCellWidth * np.ones((lat.size, lon.size))
        return cellWidth, lon, lat


def set_qu_cores(config):
    """
    Set the cores and min cores based on resolution for QU test cases.
    """
    goal_cells_per_core = config.getfloat('global_ocean_qu',
                                          'goal_cells_per_core')
    max_cells_per_core = config.getfloat('global_ocean_qu',
                                         'max_cells_per_core')
    resolution = config.getfloat('global_ocean_qu', 'resolution')

    # a heuristic
    approx_cells = 4e8 / (resolution**2)
    # ideally, about 300 cells per core
    # (make it a multiple of 4 because...it looks better?)
    cores = max(1, 4 * round(approx_cells / (4 * goal_cells_per_core)))
    # In a pinch, about 3000 cells per core
    min_cores = max(1, round(approx_cells / max_cells_per_core))

    config.set('global_ocean', 'init_cores', str(cores))
    config.set('global_ocean', 'init_min_cores', str(min_cores))
    config.set('global_ocean', 'forward_cores', str(cores))
    config.set('global_ocean', 'forward_min_cores', str(min_cores))


def get_qu_dts(config):
    """
    Get the time step and barotorpic time steps
    """
    # dt is proportional to resolution: default 30 seconds per km
    dt_per_km = config.getint('global_ocean_qu', 'dt_per_km')
    btr_dt_per_km = config.getint('global_ocean_qu', 'btr_dt_per_km')
    resolution = config.getfloat('global_ocean_qu', 'resolution')

    dt = dt_per_km * resolution
    # https://stackoverflow.com/a/1384565/7728169
    dt = time.strftime('%H:%M:%S', time.gmtime(dt))

    btr_dt = btr_dt_per_km * resolution
    btr_dt = time.strftime('%H:%M:%S', time.gmtime(btr_dt))

    return dt, btr_dt

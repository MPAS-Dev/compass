from mpas_tools.ocean import build_spherical_mesh
from mpas_tools.ocean import inject_bathymetry
from abc import abstractmethod

from compass.ocean.tests.global_ocean.mesh.cull import cull_mesh
from compass.step import Step


class MeshStep(Step):
    """
    A step for creating a global MPAS-Ocean mesh

    Attributes
    ----------
    mesh_name : str
        The name of the mesh

    with_ice_shelf_cavities : bool
        Whether the mesh includes ice-shelf cavities

    package : str
        The python package for the mesh

    mesh_config_filename : str
        The name of the mesh config file
    """
    def __init__(self, test_case, mesh_name, with_ice_shelf_cavities,
                 package, mesh_config_filename, name='mesh', subdir=None,
                 do_inject_bathymetry=False, preserve_floodplain=False):
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

        package : str
            The python package for the mesh

        mesh_config_filename : str
            The name of the mesh config file

        name : str, optional
            the name of the step

        subdir : str, optional
            the subdirectory for the step.  The default is ``name``

        do_inject_bathymetry : bool, optional
            Whether to interpolate bathymetry from a data file so it
            can be used as a culling criteria

        preserve_floodplain : bool, optional
            Whether to leave land cells in the mesh based on bathymetry
            specified by do_inject_bathymetry
        """
        super().__init__(test_case, name=name, subdir=subdir, ntasks=1,
                         min_tasks=1, cpus_per_task=None,
                         min_cpus_per_task=None, openmp_threads=1)
        for file in ['culled_mesh.nc', 'culled_graph.info',
                     'critical_passages_mask_final.nc']:
            self.add_output_file(filename=file)

        self.mesh_name = mesh_name
        self.with_ice_shelf_cavities = with_ice_shelf_cavities
        self.package = package
        self.mesh_config_filename = mesh_config_filename
        self.do_inject_bathymetry = do_inject_bathymetry
        self.preserve_floodplain = preserve_floodplain

    def setup(self):
        """
        Set up the test case in the work directory, including downloading any
        dependencies.
        """
        # get the these properties from the config options
        config = self.config
        self.cpus_per_task = config.getint('global_ocean',
                                           'mesh_cpus_per_task')
        self.min_cpus_per_task = config.getint('global_ocean',
                                               'mesh_min_cpus_per_task')

    def run(self):
        """
        Run this step of the test case
        """
        with_ice_shelf_cavities = self.with_ice_shelf_cavities
        logger = self.logger
        do_inject_bathymetry = self.do_inject_bathymetry
        preserve_floodplain = self.preserve_floodplain
        floodplain_elevation = self.config.getfloat('global_ocean',
                                                    'floodplain_elevation')

        # only use progress bars if we're not writing to a log file
        use_progress_bar = self.log_filename is None

        # create the base mesh
        cellWidth, lon, lat = self.build_cell_width_lat_lon()
        build_spherical_mesh(cellWidth, lon, lat, out_filename='base_mesh.nc',
                             logger=logger, use_progress_bar=use_progress_bar,
                             do_inject_bathymetry=do_inject_bathymetry,
                             preserve_floodplain=preserve_floodplain,
                             floodplain_elevation=floodplain_elevation)

        cull_mesh(with_critical_passages=True, logger=logger,
                  use_progress_bar=use_progress_bar,
                  preserve_floodplain=preserve_floodplain,
                  with_cavities=with_ice_shelf_cavities)

        if do_inject_bathymetry:
            inject_bathymetry(mesh_file='culled_mesh.nc')

    @abstractmethod
    def build_cell_width_lat_lon(self):
        """
        A function for creating cell width array for this mesh on a regular
        latitude-longitude grid.  Child classes need to override this function
        to return the expected data

        Returns
        -------
        cellWidth : numpy.array
            m x n array of cell width in km

        lon : numpy.array
            longitude in degrees (length n and between -180 and 180)

        lat : numpy.array
            longitude in degrees (length m and between -90 and 90)
        """
        pass

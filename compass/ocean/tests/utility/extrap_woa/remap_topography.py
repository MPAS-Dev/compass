from pyremap import LatLonGridDescriptor, Remapper

from compass.step import Step


class RemapTopography(Step):
    """
    A step for remapping bathymetry and ice-shelf topography from one
    latitude-longitude grid to another
    """

    def __init__(self, test_case):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.utility.extrap_woa.ExtraWoa
            The test case this step belongs to
        """
        super().__init__(test_case, name='remap_topography', ntasks=None,
                         min_tasks=None)
        self.add_input_file(filename='woa.nc',
                            target='../combine/woa_combined.nc')
        self.add_output_file(filename='topography_remapped.nc')

    def setup(self):
        """
        Set up the step in the work directory, including downloading any
        dependencies.
        """
        super().setup()
        config = self.config
        topo_filename = config.get('extrap_woa', 'topo_filename')

        self.add_input_file(
            filename='topography.nc',
            target=topo_filename,
            database='bathymetry_database')

        self.ntasks = config.getint('extrap_woa', 'remap_ntasks')
        self.min_tasks = config.getint('extrap_woa', 'remap_min_tasks')

    def constrain_resources(self, available_resources):
        """
        Constrain ``cpus_per_task`` and ``ntasks`` based on the number of
        cores available to this step

        Parameters
        ----------
        available_resources : dict
            The total number of cores available to the step
        """
        config = self.config
        self.ntasks = config.getint('extrap_woa', 'remap_ntasks')
        self.min_tasks = config.getint('extrap_woa', 'remap_min_tasks')
        super().constrain_resources(available_resources)

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        logger = self.logger
        parallel_executable = config.get('parallel', 'parallel_executable')

        method = config.get('extrap_woa', 'remap_method')

        in_descriptor = LatLonGridDescriptor.read(fileName='topography.nc',
                                                  lonVarName='lon',
                                                  latVarName='lat')

        in_mesh_name = in_descriptor.meshName

        out_descriptor = LatLonGridDescriptor.read(fileName='woa.nc',
                                                   lonVarName='lon',
                                                   latVarName='lat')

        out_mesh_name = out_descriptor.meshName

        mapping_file_name = \
            f'map_{in_mesh_name}_to_{out_mesh_name}_{method}.nc'
        remapper = Remapper(in_descriptor, out_descriptor, mapping_file_name)

        remapper.build_mapping_file(method=method, mpiTasks=self.ntasks,
                                    tempdir='.', logger=logger,
                                    esmf_parallel_exec=parallel_executable)

        remapper.remap_file(inFileName='topography.nc',
                            outFileName='topography_remapped.nc',
                            logger=logger)

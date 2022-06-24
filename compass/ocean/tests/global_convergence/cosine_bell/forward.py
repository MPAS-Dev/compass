import time

from compass.model import ModelStep


class Forward(ModelStep):
    """
    A step for performing forward MPAS-Ocean runs as part of the cosine bell
    test case

    Attributes
    ----------
    resolution : int
        The resolution of the (uniform) mesh in km
    """

    def __init__(self, test_case, resolution, mesh_name):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_convergence.cosine_bell.CosineBell
            The test case this step belongs to

        resolution : int
            The resolution of the (uniform) mesh in km

        mesh_name : str
            The name of the mesh
        """
        super().__init__(test_case=test_case,
                         name=f'{mesh_name}_forward',
                         subdir=f'{mesh_name}/forward',
                         openmp_threads=1)

        self.resolution = resolution

        # make sure output is double precision
        self.add_streams_file('compass.ocean.streams', 'streams.output')

        self.add_namelist_file(
            'compass.ocean.tests.global_convergence.cosine_bell',
            'namelist.forward')
        self.add_streams_file(
            'compass.ocean.tests.global_convergence.cosine_bell',
            'streams.forward')

        self.add_input_file(filename='init.nc',
                            target='../init/initial_state.nc')
        self.add_input_file(filename='graph.info',
                            target='../mesh/graph.info')

        self.add_output_file(filename='output.nc')

    def setup(self):
        """
        Set namelist options base on config options
        """
        super().setup()
        dt = self.get_dt()
        self.add_namelist_options({'config_dt': dt})

    def constrain_resources(self, available_cores):
        """
        Before we constrain the resources, we'll update the target and minimum
        number of tasks from config options

        Parameters
        ----------
        available_cores : int
            The total number of cores available to the step
        """
        config = self.config
        resolution = self.resolution

        ntasks = config.getint('cosine_bell', f'QU{resolution}_ntasks')
        min_tasks = config.getint('cosine_bell', f'QU{resolution}_min_tasks')
        self.ntasks = ntasks
        self.min_tasks = min_tasks

        super().constrain_resources(available_cores)

    def runtime_setup(self):
        """
        Update the resources and time step in case the user has update config
        options
        """
        super().runtime_setup()

        # update dt in case the user has changed dt_per_km
        dt = self.get_dt()
        self.update_namelist_at_runtime(options={'config_dt': dt},
                                        out_name='namelist.ocean')

    def get_dt(self):
        """
        Get the time step

        Returns
        -------
        dt : str
            the time step in HH:MM:SS
        """
        config = self.config
        # dt is proportional to resolution: default 30 seconds per km
        dt_per_km = config.getint('cosine_bell', 'dt_per_km')

        dt = dt_per_km * self.resolution
        # https://stackoverflow.com/a/1384565/7728169
        dt = time.strftime('%H:%M:%S', time.gmtime(dt))

        return dt

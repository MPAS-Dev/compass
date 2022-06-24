from compass.model import ModelStep


class Init(ModelStep):
    """
    A step for creating a mesh and initial condition for General Ocean
    Turbulence Model (GOTM) test cases
    """
    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.gotm.default.Default
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='init', ntasks=1,
                         min_tasks=1, openmp_threads=1)

        self.add_namelist_file('compass.ocean.tests.gotm.default',
                               'namelist.init', mode='init')

        self.add_streams_file('compass.ocean.tests.gotm.default',
                              'streams.init', mode='init')

        self.add_input_file(filename='mesh.nc', target='../mesh/mesh.nc')
        self.add_input_file(filename='graph.info', target='../mesh/graph.info')

        self.add_output_file('ocean.nc')

    def runtime_setup(self):
        """
        Update some namelist options from config options
        """
        config = self.config
        replacements = dict()
        replacements['config_periodic_planar_vert_levels'] = \
            config.get('gotm', 'vert_levels')
        replacements['config_periodic_planar_bottom_depth'] = \
            config.get('gotm', 'bottom_depth')
        self.update_namelist_at_runtime(options=replacements)

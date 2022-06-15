from compass.model import run_model
from compass.step import Step


class Init(Step):
    """
    A step for running a spherical harmonic transformation
    for the shperical_harmonic_transfrom test case
    """
    def __init__(self, test_case, resolution, algorithm, order):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.spherical_harmonic_transform.qu_convergence.QuConvergence
            The test case this step belongs to

        resolution : int
            The resolution of the (uniform) mesh in km

        algorithm : str
            Use either the 'parallel' or 'serial' algorithm

        order : int
            - For algorithm = 'parallel', the order of the shperical
              harmonic transform
            - For algorithm = 'serial', the number of latitudes in the
              Gaussian grid
        """
        super().__init__(test_case=test_case,
                         name=f'QU{resolution}_init_{algorithm}_{order}',
                         subdir=f'QU{resolution}/init/{algorithm}/{order}',
                         ntasks=36, min_tasks=1)

        package = \
            'compass.ocean.tests.spherical_harmonic_transform.qu_convergence'

        self.add_namelist_file(package, 'namelist.init', mode='init')
        self.add_streams_file(package, 'streams.init', mode='init')

        self.add_input_file(filename='mesh.nc',
                            target='../../../mesh/mesh.nc')

        self.add_input_file(filename='graph.info',
                            target='../../../mesh/graph.info')

        self.add_model_as_input()

        self.add_output_file(filename='initial_state.nc')

        init_options = dict()
        if algorithm == 'parallel':
            init_options['config_use_parallel_self_attraction_loading'] \
                = '.true.'
            init_options['config_parallel_self_attraction_loading_order'] \
                = str(order)

        else:
            init_options['config_use_parallel_self_attraction_loading'] \
                = '.false.'
            init_options['config_nLatitude'] = str(order)
            init_options['config_nLongitude'] = str(2*order)
            self.add_input_file(
                filename='mpas_to_grid.nc',
                target=f'../../../mesh/mpas_to_grid_{order}.nc')
            self.add_input_file(
                filename='grid_to_mpas.nc',
                target=f'../../../mesh/grid_to_mpas_{order}.nc')

        self.add_namelist_options(options=init_options, mode='init')

    def run(self):
        """
        Run this step of the testcase
        """

        run_model(self)

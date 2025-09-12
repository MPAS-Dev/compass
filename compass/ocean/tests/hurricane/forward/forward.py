import numpy as np

from compass.model import run_model
from compass.step import Step


class ForwardStep(Step):
    """
    A step for performing forward MPAS-Ocean runs as part of hurricane test
    cases.

    Attributes
    ----------
    mesh : compass.ocean.tests.global_ocean.mesh.Mesh
        The test case that produces the mesh for this run

    init : compass.ocean.tests.hurricane.init.Init
        The test case that produces the initial condition for this run

    use_lts: bool
        Whether local time-stepping is used
    """
    def __init__(self, test_case, mesh, init, use_lts, wetdry,
                 name='forward', subdir=None):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.hurricane.forward.Forward
            The test case this step belongs to

        mesh : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case that produces the mesh for this run

        init : compass.ocean.tests.hurricane.init.Init
            The test case that produces the initial condition for this run

        use_lts : bool
            Whether local time-stepping is to be used

        name : str, optional
            the name of the step

        subdir : str, optional
            the subdirectory for the step.  The default is ``name``
        """
        self.mesh = mesh
        self.init = init
        self.use_lts = use_lts

        super().__init__(test_case=test_case, name=name)

        if use_lts == 'LTS':

            self.add_namelist_file(
                'compass.ocean.tests.hurricane.lts.forward', 'namelist.ocean')
            self.add_streams_file(
                'compass.ocean.tests.hurricane.lts.forward', 'streams.ocean')

        elif use_lts == 'FB_LTS':

            self.add_namelist_file(
                'compass.ocean.tests.hurricane.fblts.forward',
                'namelist.ocean')
            self.add_streams_file(
                'compass.ocean.tests.hurricane.fblts.forward',
                'streams.ocean')

        else:

            self.add_namelist_file(
                'compass.ocean.tests.hurricane.forward', 'namelist.ocean')
            self.add_streams_file(
                'compass.ocean.tests.hurricane.forward', 'streams.ocean')

            mesh_package = mesh.package
            self.add_namelist_file(mesh_package, 'namelist.ocean')

        if wetdry == 'subgrid':
            self.add_namelist_file(
                'compass.ocean.tests.hurricane.forward',
                'namelist.ocean.subgrid')
            self.add_streams_file(
                'compass.ocean.tests.hurricane.forward',
                'streams.ocean_subgrid')

        if wetdry == 'subgrid':
            initial_state_target = \
                f'{init.path}/initial_state/ocean_subgrid_final.nc'
        else:
            initial_state_target = \
                f'{init.path}/initial_state/ocean.nc'
        self.add_input_file(filename='input.nc',
                            work_dir_target=initial_state_target)
        self.add_input_file(
            filename='atmospheric_forcing.nc',
            work_dir_target=f'{init.path}/interpolate/atmospheric_forcing.nc')

        file_in = 'topographic_wave_drag.nc'
        self.add_input_file(
            filename='topographic_wave_drag.nc',
            work_dir_target=f'{init.path}/wave_drag/{file_in}')

        self.add_input_file(
            filename='points.nc',
            work_dir_target=f'{init.path}/pointstats/points.nc')
        self.add_input_file(
            filename='graph.info',
            work_dir_target=f'{init.path}/initial_state/graph.info')

        self.add_model_as_input()

    def setup(self):
        """
        Set up the test case in the work directory, including downloading any
        dependencies
        """
        self._get_resources()

    def constrain_resources(self, available_resources):
        """
        Update resources at runtime from config options
        """
        self._get_resources()
        super().constrain_resources(available_resources)

    def update_namelist_pio(self, out_name=None):
        """
        Modify the namelist so the number of PIO tasks and the stride between
        them consistent with the number of nodes and cores (one PIO task per
        node).

        Parameters
        ----------
        out_name : str, optional
            The name of the namelist file to write out, ``namelist.<core>`` by
            default
        """
        config = self.config
        cores = self.ntasks * self.cpus_per_task

        if out_name is None:
            out_name = f'namelist.{self.mpas_core.name}'

        cores_per_node = config.getint('parallel', 'cores_per_node')

        # update PIO tasks based on the machine settings and the available
        # number or cores
        pio_num_iotasks = 2 * int(np.ceil(cores / cores_per_node))
        pio_stride = self.ntasks // pio_num_iotasks
        if pio_stride > cores_per_node:
            raise ValueError(f'Not enough nodes for the number of cores.  '
                             f'cores: {cores}, cores per node: '
                             f'{cores_per_node}')

        replacements = {'config_pio_num_iotasks': f'{pio_num_iotasks}',
                        'config_pio_stride': f'{pio_stride}'}

        self.update_namelist_at_runtime(options=replacements,
                                        out_name=out_name)

    def run(self):
        """
        Run this step of the testcase
        """
        run_model(self)

    def _get_resources(self):
        # get the these properties from the config options
        config = self.config
        self.ntasks = config.getint('hurricane', 'forward_ntasks')
        self.min_tasks = config.getint('hurricane', 'forward_min_tasks')
        self.openmp_threads = config.getint('hurricane', 'forward_threads')

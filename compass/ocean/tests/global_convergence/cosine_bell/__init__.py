from compass.config import CompassConfigParser
from compass.testcase import TestCase

from compass.mesh.spherical import QuasiUniformSphericalMeshStep, \
    IcosahedralMeshStep
from compass.ocean.tests.global_convergence.cosine_bell.init import Init
from compass.ocean.tests.global_convergence.cosine_bell.forward import Forward
from compass.ocean.tests.global_convergence.cosine_bell.analysis import \
    Analysis


class CosineBell(TestCase):
    """
    A test case for creating a global MPAS-Ocean mesh

    Attributes
    ----------
    resolutions : list of int
        A list of mesh resolutions

    icosahedral : bool
        Whether to use icosahedral, as opposed to less regular, JIGSAW meshes
    """
    def __init__(self, test_group, icosahedral):
        """
        Create test case for creating a global MPAS-Ocean mesh

        Parameters
        ----------
        test_group : compass.ocean.tests.cosine_bell.GlobalOcean
            The global ocean test group that this test case belongs to

        icosahedral : bool
            Whether to use icosahedral, as opposed to less regular, JIGSAW
            meshes
        """
        if icosahedral:
            subdir = 'icos/cosine_bell'
        else:
            subdir = 'qu/cosine_bell'
        super().__init__(test_group=test_group, name='cosine_bell',
                         subdir=subdir)
        self.resolutions = None
        self.icosahedral = icosahedral

        # add the steps with default resolutions so they can be listed
        config = CompassConfigParser()
        config.add_from_package(self.__module__, f'{self.name}.cfg')
        self._setup_steps(config)

    def configure(self):
        """
        Set config options for the test case
        """
        config = self.config
        config.add_from_package('compass.mesh', 'mesh.cfg')

        config.set('spherical_mesh', 'mpas_mesh_filename', 'mesh.nc')

        # set up the steps again in case a user has provided new resolutions
        self._setup_steps(config)

        init_options = dict()
        for option in ['temperature', 'salinity', 'lat_center', 'lon_center',
                       'radius', 'psi0', 'vel_pd']:
            init_options[f'config_cosine_bell_{option}'] = \
                config.get('cosine_bell', option)

        for step in self.steps.values():
            if 'init' in step.name:
                step.add_namelist_options(options=init_options, mode='init')

        self.update_cores()

    def update_cores(self):
        """ Update the number of cores and min_tasks for each forward step """

        config = self.config

        goal_cells_per_core = config.getfloat('cosine_bell',
                                              'goal_cells_per_core')
        max_cells_per_core = config.getfloat('cosine_bell',
                                             'max_cells_per_core')

        for resolution in self.resolutions:
            if self.icosahedral:
                mesh_name = f'Icos{resolution}'
            else:
                mesh_name = f'QU{resolution}'
            # a heuristic based on QU30 (65275 cells) and QU240 (10383 cells)
            approx_cells = 6e8 / resolution**2
            # ideally, about 300 cells per core
            # (make it a multiple of 4 because...it looks better?)
            ntasks = max(1,
                         4*round(approx_cells / (4 * goal_cells_per_core)))
            # In a pinch, about 3000 cells per core
            min_tasks = max(1,
                            round(approx_cells / max_cells_per_core))
            step = self.steps[f'{mesh_name}_forward']
            step.ntasks = ntasks
            step.min_tasks = min_tasks

            config.set('cosine_bell', f'{mesh_name}_ntasks', str(ntasks),
                       comment=f'Target core count for {resolution} km mesh')
            config.set('cosine_bell', f'{mesh_name}_min_tasks',
                       str(min_tasks),
                       comment=f'Minimum core count for {resolution} km mesh')

    def _setup_steps(self, config):
        """ setup steps given resolutions """
        if self.icosahedral:
            default_resolutions = '60, 120, 240, 480'
        else:
            default_resolutions = '60, 90, 120, 150, 180, 210, 240'

        # set the default values that a user may change before setup
        config.set('cosine_bell', 'resolutions', default_resolutions,
                   comment='a list of resolutions (km) to test')

        # get the resolutions back, perhaps with values set in the user's
        # config file
        resolutions = config.getlist('cosine_bell', 'resolutions', dtype=int)

        if self.resolutions is not None and self.resolutions == resolutions:
            return

        # start fresh with no steps
        self.steps = dict()
        self.steps_to_run = list()

        self.resolutions = resolutions

        for resolution in resolutions:
            if self.icosahedral:
                mesh_name = f'Icos{resolution}'
            else:
                mesh_name = f'QU{resolution}'

            name = f'{mesh_name}_mesh'
            subdir = f'{mesh_name}/mesh'
            if self.icosahedral:
                self.add_step(IcosahedralMeshStep(
                    test_case=self, name=name, subdir=subdir,
                    cell_width=resolution))
            else:
                self.add_step(QuasiUniformSphericalMeshStep(
                    test_case=self, name=name, subdir=subdir,
                    cell_width=resolution))

            self.add_step(Init(test_case=self, mesh_name=mesh_name))

            self.add_step(Forward(test_case=self, resolution=resolution,
                                  mesh_name=mesh_name))

        self.add_step(Analysis(test_case=self, resolutions=resolutions,
                               icosahedral=self.icosahedral))

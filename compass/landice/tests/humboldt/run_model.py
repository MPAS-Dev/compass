from compass.model import run_model, make_graph_file
from compass.step import Step


class RunModel(Step):
    """
    A step for performing forward MALI runs as part of Humboldt test cases.

    Attributes
    ----------
    mesh_type : str
        The resolution or mesh type of the test case

    velo_solver : str
        The velocity solver used for the test case

    calving_law : str
        The calving law used for the test case

    suffixes : list of str, optional
        a list of suffixes for namelist and streams files produced
        for this step.  Most steps most runs will just have a
        ``namelist.landice`` and a ``streams.landice`` (the default) but
        the ``restart_run`` step of the ``restart_test`` runs the model
        twice, the second time with ``namelist.landice.rst`` and
        ``streams.landice.rst``

    mesh_file : str
        The name of the mesh file being used

    forcing_file : str
        The name of the forcing file being used

    """
    def __init__(self, test_case, velo_solver, mesh_type,
                 name='run_model',
                 calving_law=None,
                 damage=None,
                 face_melt=False,
                 subdir=None, cores=1,
                 min_cores=None, threads=1, suffixes=None):
        """
        Create a new test case

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        velo_solver : {'sia', 'FO', 'none'}
            The velocity solver setting to use for this test case

        calving_law: {'none', 'floating', 'eigencalving',
                      'specified_calving_velocity', 'von_Mises_stress',
                      'damagecalving', 'ismip6_retreat'}, optional
            The calving law setting to use for this test case. If not
            specified, set to 'none'.

        damage : str
            The damage method used for the test case

        face_melt : bool
            Whether to include face melting

        mesh_type : {'1km', '3km'}
            The resolution or mesh type of the test case

        name : str, optional
            the name of the test case

        subdir : str, optional
            the subdirectory for the step.  The default is ``name``

        cores : int, optional
            the number of cores the step would ideally use.  If fewer cores
            are available on the system, the step will run on all available
            cores as long as this is not below ``min_cores``

        min_cores : int, optional
            the number of cores the step requires.  If the system has fewer
            than this number of cores, the step will fail

        threads : int, optional
            the number of threads the step will use

        suffixes : list of str, optional
            a list of suffixes for namelist and streams files produced
            for this step.  Most steps most runs will just have a
            ``namelist.landice`` and a ``streams.landice`` (the default) but
            the ``restart_run`` step of the ``restart_test`` runs the model
            twice, the second time with ``namelist.landice.rst`` and
            ``streams.landice.rst``
        """
        self.mesh_type = mesh_type
        self.velo_solver = velo_solver
        assert self.velo_solver in {'sia', 'FO', 'none'}, \
            "Value of velo_solver must be one of {'sia', 'FO', 'none'}"
        if calving_law:
            self.calving_law = calving_law
        else:
            self.calving_law = 'none'
        assert self.calving_law in {'none', 'floating', 'eigencalving',
                                    'specified_calving_velocity',
                                    'von_Mises_stress',
                                    'damagecalving', 'ismip6_retreat'}, \
            "Value of calving_law must be one of {'none', 'floating', " \
            "'eigencalving', 'specified_calving_velocity', " \
            "'von_Mises_stress', 'damagecalving', 'ismip6_retreat'}"
        if not damage is None:
            assert damage in {'threshold', }, \
                "Value of damage must be one of {'threshold', }."

        if suffixes is None:
            suffixes = ['landice']
        self.suffixes = suffixes
        if min_cores is None:
            min_cores = cores
        super().__init__(test_case=test_case, name=name, subdir=subdir,
                         cores=cores, min_cores=min_cores, threads=threads)

        # Commented code to make use of mesh generation step
        # Note it will not include uReconstructX/Y or muFriction!
        # self.add_input_file(filename='landice_grid.nc',
        #                     target='../mesh/Humboldt_1to10km.nc')
        # self.add_input_file(filename='graph.info',
        #                     target='../mesh/graph.info')

        # download and link one of the premade meshes and forcing files
        if self.mesh_type == '1km':
            self.mesh_file = 'Humboldt_1to10km_r04_20210615.nc'
            self.forcing_file = 'Humboldt_1to10km_MIROC5-rcp85_ismip-gis.nc'
        elif self.mesh_type == '3km':
            self.mesh_file = 'Humboldt_3to30km_r04_20210615.nc'
            self.forcing_file = 'Humboldt_3to30km_MIROC5-rcp85_ismip6-gis.nc'
        self.add_input_file(filename=self.mesh_file, target=self.mesh_file,
                            database='')
        self.add_input_file(filename=self.forcing_file,
                            target=self.forcing_file,
                            database='')

        if velo_solver == 'FO':
            self.add_input_file(filename='albany_input.yaml',
                                package='compass.landice.tests.humboldt',
                                copy=True)

        self.add_model_as_input()

        self.add_output_file(filename='output.nc')

        for suffix in suffixes:
            self.add_namelist_file(
                'compass.landice.tests.humboldt', 'namelist.landice',
                out_name='namelist.{}'.format(suffix))
            options = {'config_velocity_solver': "'{}'".format(velo_solver),
                       'config_calving': "'{}'".format(calving_law)}
            # optionally add damage and facemelt options if included
            if damage == 'threshold':
                options['config_calculate_damage'] = '.true.'
                options['config_damage_calving_method'] = "'threshold'"
                options['config_damage_calving_threshold'] = '0.5'
            if face_melt == True:
                options['config_front_mass_bal_grounded'] = "'ismip6'"
            # now add accumulated options to namelist
            self.add_namelist_options(options=options,
                                      out_name='namelist.{}'.format(suffix))

            stream_replacements = {'HUMBOLDT_INPUT_FILE': self.mesh_file,
                                   'HUMBOLDT_FORCING_FILE': self.forcing_file}
            self.add_streams_file(
                'compass.landice.tests.humboldt', 'streams.landice.template',
                out_name='streams.{}'.format(suffix),
                template_replacements = stream_replacements)


    # no setup() is needed

    def run(self):
        """
        Run this step of the test case
        """
        make_graph_file(mesh_filename=self.mesh_file,
                        graph_filename='graph.info')
        for suffix in self.suffixes:
            run_model(step=self, namelist='namelist.{}'.format(suffix),
                      streams='streams.{}'.format(suffix))

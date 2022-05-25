import os
from compass.model import run_model, make_graph_file
from compass.step import Step


class RunModel(Step):
    """
    A step for performing forward MALI runs as part of Antarctica test cases.

    Attributes
    ----------
    mesh_file : str
        The name of the mesh file being used

    mesh_type : str
        The resolution or mesh type of the test case
    """

    def __init__(self, test_case, mesh_type, name='run_model'):
        """
        Create a new test case

        Parameters
        ----------
        mesh_file : str
            The name of the mesh file being used

        mesh_type : {'high', 'mid'}
            The resolution or mesh type of the test case
        """

        self.mesh_file = None
        self.mesh_type = mesh_type

        super().__init__(test_case=test_case, name=name)

    def setup(self, velo_solver="FO"):
        config = self.config
        section = config['ismip6_run_ais']
        base_path_mali = section.get('base_path_mali')
        calving_law = section.get('calving_law')
        damage = section.get('damage')
        procs = section.get('procs')

        if calving_law not in ['none', 'floating', 'eigencalving',
                               'specified_calving_velocity',
                               'von_Mises_stress',
                               'damagecalving', 'ismip6_retreat']:
            raise ValueError("Value of calving_law must be one of {'none', "
                             "'floating', "
                             "'eigencalving', 'specified_calving_velocity', "
                             "'von_Mises_stress', 'damagecalving', "
                             "'ismip6_retreat'}")

        self.add_input_file(filename='albany_input.yaml',
                            package='compass.landice.tests.ismip6_run_ais',
                            copy=True)

        self.add_model_as_input()

        self.add_output_file(filename='output.nc')

        # Todo: confirm the number of cores needed for the high mesh
        # We estimate that 200-1000 cells should be allocated for one core
        res_param = {
                'high': {'mesh_file': 'Antarctic_1to10km.nc',
                         'cores': 25000,
                         'min_cores': 5000},
                'mid': {'mesh_file': 'Antarctic_8to80km_20220407.nc',
                        'cores': 400,
                        'min_cores': 80}
                }

        res_param = res_param[self.mesh_type]
        self.mesh_file = res_param['mesh_file']
        self.cores = res_param['cores']
        self.min_cores = res_param['min_cores']

        self.add_input_file(filename=self.mesh_file,
                            target=os.path.join(base_path_mali,
                                                self.mesh_file))

        # Todo: upload the AIS meshes to the database
#        self.add_input_file(filename=self.mesh_file, target=self.mesh_file,
#                            database='')

        self.add_namelist_file(
            'compass.landice.tests.ismip6_run_ais', 'namelist.landice',
            out_name='namelist.landice')
        options = {'config_velocity_solver': f"'{velo_solver}'",
                   'config_calving': f"'{calving_law}'"}

        if damage == 'threshold':
            options['config_calculate_damage'] = '.true.'
            options['config_damage_calving_method'] = "'threshold'"
            options['config_damage_calving_threshold'] = '0.5'

        # now add accumulated options to namelist
        self.add_namelist_options(options=options,
                                  out_name='namelist.landice')

        period_endyear = section.get("period_endyear")
        model = section.get("model")
        scenario = section.get("scenario")
        base_path_forcing = section.get('base_path_forcing')
        fname_basal_coeff = section.get('fname_basal_coeff')

        target = f'{base_path_forcing}/ocean_forcing/thermal_forcing/' \
                 f'{model}_{scenario}/1995-{period_endyear}/' \
                 f'processed_TF_{model}_{scenario}_{period_endyear}.nc'
        self.add_input_file(filename='thermal_forcing.nc',
                            target=target)
        target = f'{base_path_forcing}/atmosphere_forcing/' \
                 f'{model}_{scenario}/1995-{period_endyear}/' \
                 f'processed_SMB_{model}_{scenario}_{period_endyear}.nc'
        self.add_input_file(filename='smb.nc',
                            target=target)
        target = f'{base_path_forcing}/ocean_forcing/basal_melt/' \
                 f'parametrizations/{fname_basal_coeff}'
        self.add_input_file(filename='basal_coeff.nc',
                            target=target)
        target = f'{base_path_forcing}/ocean_forcing/shelf_melt_offset/' \
                 f'{model}_{scenario}/1995-{period_endyear}/' \
                 f'processed_shelfmelt_offset_{model}_{scenario}.nc'
        self.add_input_file(filename='shelf_melt_offset.nc',
                            target=target)

        stream_replacements = {'input_file_ais': self.mesh_file,
                               'file_ais_TF_forcing': 'thermal_forcing.nc',
                               'file_ais_SMB_forcing': 'smb.nc',
                               'file_ais_basal_param': 'basal_coeff.nc',
                               'file_ais_shelf_melt_offset': 'shelf_melt_'
                                                             'offset.nc'
                               }

        self.add_streams_file(
            'compass.landice.tests.ismip6_run_ais', 'streams.landice.template',
            out_name='streams.landice',
            template_replacements=stream_replacements)

    def run(self):
        """
        Run this step of the test case
        """
        make_graph_file(mesh_filename=self.mesh_file,
                        graph_filename='graph.info')
        run_model(step=self, namelist='namelist.landice',
                  streams='streams.landice')

import os

from mpas_tools.logging import check_call

from compass.step import Step


class DomainFiles(Step):
    """
    A step for building domain files for meshes tailored to hurricane-induced
    coastal flooding

    Attributes
    ----------
    mesh : compass.mesh.spherical.SphericalBaseStep
        The base mesh step containing input files to this step
    """

    def __init__(self, test_case):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.hurricane.files_for_e3sm.FilesForE3SM
            The test case this step belongs to
        """
        super().__init__(test_case, name='domain_files')
        self.mesh = test_case.mesh
        self.creation_date = test_case.creation_date

    def setup(self):
        """
        Set up the step in the work directory, including downloading any
        dependencies.
        """
        super().setup()

        atm_grid = self.config.get('files_for_e3sm', 'atm_grid')
        ocn_grid = self.mesh.mesh_name
        creation_date = self.creation_date

        map_file_path = os.path.join(
            self.test_case.steps['forcing_maps'].path,
            f'map_{ocn_grid}_to_{atm_grid}_traave.{creation_date}.nc',
        )
        self.add_input_file(
            filename='map_ocn_to_atm_traave.nc', work_dir_target=map_file_path)
        self.add_output_file(
            filename=f'domain.lnd.{atm_grid}_{ocn_grid}.{creation_date}.nc')
        self.add_output_file(
            filename=f'domain.ocn.{atm_grid}_{ocn_grid}.{creation_date}.nc')
        self.add_output_file(
            filename=f'domain.ocn.{ocn_grid}.{creation_date}.nc')

    def run(self):
        """
        Run this step of the test case
        """
        super().run()
        self._domain_files()

    def _domain_files(self):
        """
        Create domain files
        """

        section = self.config['files_for_e3sm']
        domain_files_exe = section.get('domain_files_exe')
        atm_grid = section.get('atm_grid')
        ocn_grid = self.mesh.mesh_name

        logger = self.logger
        logger.info(
            f'Create domain files for {atm_grid}_{ocn_grid} grid pair.'
        )

        args = [
            'python', domain_files_exe,
            '--date-stamp', self.creation_date,
            '-m', 'map_ocn_to_atm_traave.nc',
            '-o', ocn_grid,
            '-l', atm_grid,
        ]
        check_call(args, logger)

        logger.info('  Done.')

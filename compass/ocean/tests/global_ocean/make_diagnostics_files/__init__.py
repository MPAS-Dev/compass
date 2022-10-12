import os
import xarray
from datetime import datetime

from mpas_tools.scrip.from_mpas import scrip_from_mpas

from compass.io import symlink
from compass.testcase import TestCase
from compass.step import Step
from compass.ocean.tests.global_ocean.files_for_e3sm.diagnostics_files import \
    make_diagnostics_files
from compass.ocean.tests.global_ocean.files_for_e3sm.e3sm_to_cmip_maps import \
    make_e3sm_to_cmip_maps


class MakeDiagnosticsFiles(TestCase):
    """
    A test case for making diagnostics files (mapping files and region masks)
    from an existing mesh.
    """
    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.global_ocean.GlobalOcean
            The global ocean test group that this test case belongs to
        """
        super().__init__(test_group=test_group, name='make_diagnostics_files')

        self.add_step(E3smToCmpiMaps(test_case=self))
        self.add_step(DiagnosticsFiles(test_case=self))

    def configure(self):
        """
        Modify the configuration options for this test case
        """
        self.config.add_from_package(
           'compass.ocean.tests.global_ocean.make_diagnostics_files',
           'make_diagnostics_files.cfg', exception=True)

    def run(self):
        """
        Run each step of the testcase
        """
        cores = self.config.getint( 'make_diagnostics_files', 'cores')
        self.steps['diagnostics_files'].cpus_per_task = cores
        self.steps['e3sm_to_cmip_maps'].ntasks = cores

        # run the step
        super().run()


class E3smToCmpiMaps(Step):
    """
    A step for making e3sm_to_cmip mapping files
    """
    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.make_diagnostics_files.MakeDiagnosticsFiles
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='e3sm_to_cmip_maps')

        # add both scrip files, since we don't know in advance which to use
        self.add_input_file(
            filename='cmip6_180x360_scrip.20181001.nc',
            target='cmip6_180x360_scrip.20181001.nc',
            database='map_database')

        self.add_input_file(
            filename='cmip6_720x1440_scrip.20181001.nc',
            target='cmip6_720x1440_scrip.20181001.nc',
            database='map_database')

    def run(self):
        """
        Run this step of the test case
        """

        config = self.config
        section = config['make_diagnostics_files']
        mesh_filename = section.get('mesh_filename')

        symlink(os.path.join('..', mesh_filename), 'restart.nc')

        with xarray.open_dataset('restart.nc') as ds:
            if 'MPAS_Mesh_Short_Name' in ds.attrs:
                mesh_short_name = ds.attrs['MPAS_Mesh_Short_Name']
            else:
                mesh_short_name = section.get('mesh_name')
            if 'MPAS_Mesh_Prefix' in ds.attrs:
                mesh_prefix = ds.attrs['MPAS_Mesh_Prefix']
                prefix = f'MPAS_Mesh_{mesh_prefix}'
                creation_date = ds.attrs[f'{prefix}_Version_Creation_Date']
            else:
                creation_date = config.get('global_ocean', 'creation_date')
                if creation_date == 'autodetect':
                    now = datetime.now()
                    creation_date = now.strftime("%y%m%d")

        scrip_from_mpas('restart.nc', 'ocean.scrip.nc')

        make_e3sm_to_cmip_maps(self.config, self.logger, mesh_short_name,
                               creation_date, self.subdir, self.ntasks)


class DiagnosticsFiles(Step):
    """
    A step for making diagnostics files (mapping files and region masks) from
    an existing mesh.
    """
    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.make_diagnostics_files.MakeDiagnosticsFiles
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='diagnostics_files')

    def run(self):
        """
        Run this step of the test case
        """

        config = self.config
        section = config['make_diagnostics_files']

        mesh_filename = section.get('mesh_filename')
        with_ice_shelf_cavities = section.getboolean('with_ice_shelf_cavities')

        symlink(os.path.join('..', mesh_filename), 'restart.nc')

        with xarray.open_dataset('restart.nc') as ds:
            if 'MPAS_Mesh_Short_Name' in ds.attrs:
                mesh_short_name = ds.attrs['MPAS_Mesh_Short_Name']
            else:
                mesh_short_name = section.get('mesh_name')

        make_diagnostics_files(self.config, self.logger, mesh_short_name,
                               with_ice_shelf_cavities, self.cpus_per_task)

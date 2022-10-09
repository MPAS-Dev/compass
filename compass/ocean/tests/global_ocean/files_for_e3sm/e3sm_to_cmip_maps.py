import os
import xarray

from mpas_tools.logging import check_call

from compass.io import symlink
from compass.step import Step


class E3smToCmipMaps(Step):
    """
    A step for creating mapping files from the MPAS-Ocean mesh to a standard
    CMIP6 mesh
    """
    def __init__(self, test_case, restart_filename):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.files_for_e3sm.FilesForE3SM
            The test case this step belongs to

        restart_filename : str
            A restart file from the end of the dynamic adjustment test case to
            use as the basis for an E3SM initial condition
        """

        super().__init__(test_case, name='e3sm_to_cmip_map', ntasks=1,
                         min_tasks=1, openmp_threads=1)

        self.add_input_file(filename='README', target='../README')
        self.add_input_file(filename='restart.nc',
                            target=f'../{restart_filename}')

        self.add_input_file(filename='../scrip/ocean.scrip.nc')

        # add both scrip files, since we don't know in advance which to use
        self.add_input_file(
            filename='cmip6_180x360_scrip.20181001.nc',
            target='cmip6_180x360_scrip.20181001.nc',
            database='map_database')

        self.add_input_file(
            filename='cmip6_720x1440_scrip.20181001.nc',
            target='cmip6_720x1440_scrip.20181001.nc',
            database='map_database')

        self.add_output_file(filename='map_mpas_to_cmip6_aave.nc')
        self.add_output_file(filename='map_mpas_to_cmip6_mono.nc')
        self.add_output_file(filename='map_mpas_to_cmip6_nco.nc')

    def run(self):
        """
        Run this step of the testcase
        """
        with xarray.open_dataset('restart.nc') as ds:
            mesh_short_name = ds.attrs['MPAS_Mesh_Short_Name']
            mesh_prefix = ds.attrs['MPAS_Mesh_Prefix']
            prefix = f'MPAS_Mesh_{mesh_prefix}'
            creation_date = ds.attrs[f'{prefix}_Version_Creation_Date']

        make_e3sm_to_cmip_maps(self.config, self.logger, mesh_short_name,
                               creation_date, self.subdir)


def make_e3sm_to_cmip_maps(config, logger, mesh_short_name, creation_date,
                           subdir):
    """
    Make mapping file from the MPAS-Ocean mesh to the CMIP6 grid

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for this test case

    logger : logging.Logger
        A logger for output from the step

    mesh_short_name : str
        The E3SM short name of the mesh

    creation_date : str
        The date to append to the mapping files

    subdir : str
        The subdirectory this function is run from, for symlinking into
        ``assembled_files``
    """

    link_dir = f'../assembled_files/diagnostics/e3sm_to_cmip/maps'

    try:
        os.makedirs(link_dir)
    except OSError:
        pass

    src_scrip_filename = 'ocean.scrip.nc'
    cmip6_grid_res = config.get('files_for_e3sm', 'cmip6_grid_res')
    if cmip6_grid_res == '180x360':
        dst_scrip_filename = f'cmip6_180x360_scrip.20181001.nc'
    elif cmip6_grid_res == '720x1440':
        dst_scrip_filename = f'cmip6_720x1440_scrip.20181001.nc'
    else:
        raise ValueError(f'Unexpected cmip6_grid_res: {cmip6_grid_res}')

    map_methods = dict(aave='conserve', mono='fv2fv_flx', nco='nco')
    for suffix, map_method in map_methods.items():
        local_map_filename = f'map_mpas_to_cmip6_{suffix}.nc'
        args = ['ncremap', f'--alg_typ={map_method}',
                f'--grd_src={src_scrip_filename}',
                f'--grd_dst={dst_scrip_filename}',
                f'--map={local_map_filename}']
        check_call(args, logger=logger)

        map_filename = \
            f'map_{mesh_short_name}_to_cmip6_{cmip6_grid_res}_{suffix}.{creation_date}.nc'

        symlink(f'../../../{subdir}/{local_map_filename}',
                f'{link_dir}/{map_filename}')

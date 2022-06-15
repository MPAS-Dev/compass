import os
import xarray

from mpas_tools.scrip.from_mpas import scrip_from_mpas

from compass.io import symlink
from compass.step import Step


class Scrip(Step):
    """
    A step for creating SCRIP files from the MPAS-Ocean mesh

    with_ice_shelf_cavities : bool
        Whether the mesh includes ice-shelf cavities
    """
    def __init__(self, test_case, restart_filename, with_ice_shelf_cavities):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.files_for_e3sm.FilesForE3SM
            The test case this step belongs to

        restart_filename : str
            A restart file from the end of the dynamic adjustment test case to
            use as the basis for an E3SM initial condition

        with_ice_shelf_cavities : bool
            Whether the mesh includes ice-shelf cavities
        """

        super().__init__(test_case, name='scrip', ntasks=1,
                         min_tasks=1, openmp_threads=1)

        self.add_input_file(filename='README', target='../README')
        self.add_input_file(filename='restart.nc',
                            target='../{}'.format(restart_filename))

        self.with_ice_shelf_cavities = with_ice_shelf_cavities

        # for now, we won't define any outputs because they include the mesh
        # short name, which is not known at setup time.  Currently, this is
        # safe because no other steps depend on the outputs of this one.

    def run(self):
        """
        Run this step of the testcase
            """
        with_ice_shelf_cavities = self.with_ice_shelf_cavities

        with xarray.open_dataset('restart.nc') as ds:
            mesh_short_name = ds.attrs['MPAS_Mesh_Short_Name']
            mesh_prefix = ds.attrs['MPAS_Mesh_Prefix']
            prefix = 'MPAS_Mesh_{}'.format(mesh_prefix)
            creation_date = ds.attrs['{}_Version_Creation_Date'.format(prefix)]

        try:
            os.makedirs('../assembled_files/inputdata/ocn/mpas-o/{}'.format(
                mesh_short_name))
        except OSError:
            pass

        if with_ice_shelf_cavities:
            nomask_str = '.nomask'
        else:
            nomask_str = ''

        scrip_filename = 'ocean.{}{}.scrip.{}.nc'.format(
            mesh_short_name,  nomask_str, creation_date)

        scrip_from_mpas('restart.nc', scrip_filename)

        symlink('../../../../../scrip/{}'.format(scrip_filename),
                '../assembled_files/inputdata/ocn/mpas-o/{}/{}'.format(
                    mesh_short_name, scrip_filename))

        if with_ice_shelf_cavities:
            scrip_mask_filename = 'ocean.{}.mask.scrip.{}.nc'.format(
                mesh_short_name, creation_date)
            scrip_from_mpas('restart.nc', scrip_mask_filename,
                            useLandIceMask=True)

            symlink(
                '../../../../../scrip/{}'.format(
                    scrip_mask_filename),
                '../assembled_files/inputdata/ocn/mpas-o/{}/{}'.format(
                    mesh_short_name, scrip_mask_filename))

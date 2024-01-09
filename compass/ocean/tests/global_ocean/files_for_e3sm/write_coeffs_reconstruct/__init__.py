import os

from mpas_tools.logging import check_call

from compass.io import symlink
from compass.ocean.tests.global_ocean.files_for_e3sm.files_for_e3sm_step import (  # noqa: E501
    FilesForE3SMStep,
)
from compass.ocean.tests.global_ocean.forward import ForwardStep


class WriteCoeffsReconstruct(ForwardStep, FilesForE3SMStep):
    """
    A step for writing out ``coeffs_reconstruct`` for a given MPAS mesh
    """

    def __init__(self, test_case, mesh, init):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        mesh : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case that produces the mesh for this run

        init : compass.ocean.tests.global_ocean.init.Init
            The test case that produces the initial condition for this run
        """
        super().__init__(test_case=test_case, mesh=mesh, init=init,
                         time_integrator='split_explicit_ab2',
                         name='write_coeffs_reconstruct')

        package = 'compass.ocean.tests.global_ocean.files_for_e3sm.' \
                  'write_coeffs_reconstruct'

        self.add_namelist_file(package, 'namelist.reconstruct')
        self.add_streams_file(package, 'streams.reconstruct')

    def run(self):
        """
        Run this step of the testcase
        """
        super().run()

        # perform a test reconstruction to make sure things are working
        args = ['vector_reconstruct',
                '-i', 'restart.nc',
                '-o', 'velocity_components.nc',
                '-w', 'coeffs_reconstruct.nc',
                '-v', 'normalVelocity',
                '--out_variables', 'velocity']
        check_call(args=args, logger=self.logger)

        reconstruct_dir = \
            '../assembled_files/diagnostics/mpas_analysis/reconstruct'
        try:
            os.makedirs(reconstruct_dir)
        except FileExistsError:
            pass

        reconstruct_filename = f'{self.mesh_short_name}_coeffs_reconstruct.nc'
        # make links in diagnostics directory
        symlink(os.path.abspath('coeffs_reconstruct.nc'),
                f'{reconstruct_dir}/{reconstruct_filename}')

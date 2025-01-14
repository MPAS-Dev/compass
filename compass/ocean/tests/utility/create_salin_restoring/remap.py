import os
import pathlib
from datetime import datetime
from glob import glob

from mpas_tools.logging import check_call

from compass.parallel import run_command
from compass.step import Step


class Remap(Step):
    """
    A step for remapping sea-surface salinity to a cubed-sphere grid

    Attributes
    ----------
    resolution : int
        face subdivisions

    datestring : str
        The datestamp to append to filenames
    """

    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.utility.create_salin_restoring.CreateSalinRestoring
            The test case this step belongs to
        """  # noqa: E501
        super().__init__(
            test_case, name='remap', ntasks=None, min_tasks=None,
        )
        self.resolution = None
        self.datestring = None

    def setup(self):
        """
        Set up the step in the work directory, including downloading any
        dependencies.
        """
        super().setup()

        config = self.config
        section = config['salinity_restoring']

        now = datetime.now()

        datestring = now.strftime("%Y%m%d")
        self.datestring = datestring

        in_filename = f'woa23_decav_0.25_sss_monthly_extrap.{datestring}.nc'
        target = f'../extrap/{in_filename}'
        self.add_input_file(filename=in_filename, target=target)

        self._set_res_and_outputs(update=False)

        # Get ntasks and min_tasks
        self.ntasks = section.getint('ntasks')
        self.min_tasks = section.getint('min_tasks')

    def constrain_resources(self, available_resources):
        """
        Constrain ``cpus_per_task`` and ``ntasks`` based on the number of
        cores available to this step

        Parameters
        ----------
        available_resources : dict
            The total number of cores available to the step
        """
        config = self.config
        section = config['salinity_restoring']
        self.ntasks = section.getint('ntasks')
        self.min_tasks = section.getint('min_tasks')
        super().constrain_resources(available_resources)

    def run(self):
        """
        Run this step of the test case
        """
        self._set_res_and_outputs(update=True)
        self._create_target_scrip_file()
        self._remap()
        self._cleanup()

    def _set_res_and_outputs(self, update):
        """
        Set or update the resolution and output filenames based on config
        options
        """
        config = self.config
        section = config['salinity_restoring']
        resolution = section.getint('resolution')
        if update and resolution == self.resolution:
            return

        self.resolution = resolution
        datestring = self.datestring

        # Start over with empty outputs
        self.outputs = []

        scrip_filename = f'ne{resolution}_{datestring}.scrip.nc'
        self.add_output_file(scrip_filename)

        out_filename = \
            f'woa23_decav_ne{resolution}_sss_monthly_extrap.{datestring}.nc'
        self.add_output_file(out_filename)

        if update:
            # We need to set absolute paths
            step_dir = self.work_dir
            self.outputs = [os.path.abspath(os.path.join(step_dir, filename))
                            for filename in self.outputs]

    def _create_target_scrip_file(self):
        """
        Create SCRIP file for either the x.xxxx degree (lat-lon) mesh or the
        NExxx (cubed-sphere) mesh, depending on the value of `self.target_grid`
        References:
          https://acme-climate.atlassian.net/wiki/spaces/DOC/pages/872579110/
          Running+E3SM+on+New+Atmosphere+Grids
        """
        logger = self.logger
        logger.info(f'Create SCRIP file for ne{self.resolution} mesh')

        out_filename = self.outputs[0]
        stem = pathlib.Path(out_filename).stem
        netcdf4_filename = f'{stem}.netcdf4.nc'

        # Create EXODUS file
        args = [
            'GenerateCSMesh', '--alt', '--res', f'{self.resolution}',
            '--file', f'ne{self.resolution}.g',
        ]
        check_call(args, logger)

        # Create SCRIP file
        args = [
            'ConvertMeshToSCRIP', '--in', f'ne{self.resolution}.g',
            '--out', netcdf4_filename,
        ]
        check_call(args, logger)

        # writing out directly to NETCDF3_64BIT_DATA is either very slow or
        # unsupported, so use ncks
        args = [
            'ncks', '-O', '-5',
            netcdf4_filename,
            out_filename,
        ]
        check_call(args, logger)

        logger.info('  Done.')

    def _remap(self):
        """
        Remap salinity restoring
        """
        logger = self.logger
        logger.info('Remapping salinity restoring to cubed-sphere grid')

        # Parse config
        config = self.config
        section = config['salinity_restoring']
        method = section.get('method')

        in_filename = self.inputs[0]
        out_filename = self.outputs[1]

        mapping_filename = f'map_0.25_deg_to_ne{self.resolution}_{method}.nc'

        self._create_weights(in_filename, mapping_filename)
        self._remap_to_target(
            in_filename, mapping_filename, out_filename,
        )

        logger.info('  Done.')

    def _create_weights(self, in_filename, out_filename):
        """
        Create weights file for remapping to cubed-sphere grid

        Parameters
        ----------
        in_filename : str
            source file name

        out_filename : str
            weights file name
        """
        config = self.config
        method = config.get('salinity_restoring', 'method')

        # Generate weights file
        args = [
            'ESMF_RegridWeightGen',
            '--source', in_filename,
            '--destination', self.outputs[0],
            '--weight', out_filename,
            '--method', method,
            '--netcdf4',
            '--src_regional',
            '--ignore_unmapped',
        ]
        run_command(
            args, self.cpus_per_task, self.ntasks,
            self.openmp_threads, config, self.logger,
        )

    def _remap_to_target(self, in_filename, mapping_filename, out_filename):
        """
        Remap to `self.target_grid`. Filenames are passed as parameters so
        that the function can be applied to GEBCO and BedMachine.

        Parameters
        ----------
        in_filename : str
            source file name

        mapping_filename : str
            weights file name

        out_filename : str
            remapped file name
        """
        # Build command args
        args = [
            'ncremap',
            '-m', mapping_filename,
            '--vrb=1',
            in_filename,
            out_filename
        ]

        # Remap to target grid
        check_call(args, self.logger)

    def _cleanup(self):
        """
        Clean up work directory
        """
        logger = self.logger
        logger.info('Cleaning up work directory')

        # Remove PETxxx.RegridWeightGen.Log files
        for f in glob('*.RegridWeightGen.Log'):
            os.remove(f)

        logger.info('  Done.')

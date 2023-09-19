import os
from datetime import datetime
from glob import glob

import numpy as np
from mpas_tools.logging import check_call
from mpas_tools.scrip.from_mpas import scrip_from_mpas

from compass.io import symlink
from compass.step import Step


class FilesForE3sm(Step):
    """
    A step for creating an E3SM ocean initial condition from the results of
    the ssh-adjustment process

    Attributes
    ----------
    resolution : float
        The horizontal resolution (km) of the test case

    experiment : {'Ocean0', 'Ocean1', 'Ocean2'}
        The ISOMIP+ experiment

    out_dir : str
        The subdirectory where files will be "assembled" within the same
        directory structure as the inputdata for E3SM

    creation_date : str
        The date today, to append to file names

    init_filename : str
        The filename for the E3SM initial condition

    forcing_filename : str
        The filename for the E3SM forcing

    graph_filename : str
        The filename for the E3SM graph file

    no_mask_scrip_filename : str
        The filename for the E3SM scrip file with land-ice included

    mask_scrip_filename : str
        The filename for the E3SM scrip file without land-ice (just open ocean)

    """
    def __init__(self, test_case, resolution, experiment):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolution : float
            The horizontal resolution (km) of the test case

        experiment : {'Ocean0', 'Ocean1', 'Ocean2'}
            The ISOMIP+ experiment
        """

        super().__init__(test_case, name='files_for_e3sm', ntasks=1,
                         min_tasks=1, openmp_threads=1)

        self.resolution = resolution
        self.experiment = experiment
        self.out_dir = None
        self.creation_date = None
        self.init_filename = None
        self.forcing_filename = None
        self.graph_filename = None
        self.no_mask_scrip_filename = None
        self.mask_scrip_filename = None

    def setup(self):
        resolution = self.resolution
        experiment = self.experiment
        if resolution == int(resolution):
            res_string = f'{int(resolution)}km'
        else:
            res_string = f'{resolution}km'

        mesh_short_name = f'IsomipPlus{res_string}{experiment}'
        now = datetime.now()
        creation_date = now.strftime("%Y%m%d")

        out_dir = f'assembled_files/inputdata/ocn/mpas-o/{mesh_short_name}'
        out_filename = f'mpaso.{mesh_short_name}.{creation_date}.nc'
        self.add_input_file(filename=out_filename,
                            target='../ssh_adjustment/adjusted_init.nc')
        self.init_filename = out_filename
        self.add_output_file(f'{out_dir}/{out_filename}')

        out_filename = f'mpaso.forcing.{mesh_short_name}.{creation_date}.nc'
        self.add_input_file(
            filename=out_filename,
            target='../initial_state/init_mode_forcing_data.nc')
        self.forcing_filename = out_filename
        self.add_output_file(f'{out_dir}/{out_filename}')

        graph_filename = f'mpas-o.graph.info.{creation_date}'
        self.add_input_file(filename=graph_filename,
                            target='../cull_mesh/culled_graph.info')
        self.add_output_file(f'{out_dir}/{graph_filename}')
        self.graph_filename = graph_filename

        self.out_dir = out_dir
        self.creation_date = creation_date

        out_filename = \
            f'{mesh_short_name}.nomask.scrip.{creation_date}.nc'
        self.add_output_file(out_filename)
        self.add_output_file(f'{out_dir}/{out_filename}')
        self.no_mask_scrip_filename = out_filename

        out_filename = f'{mesh_short_name}.scrip.{creation_date}.nc'
        self.add_output_file(out_filename)
        self.add_output_file(f'{out_dir}/{out_filename}')
        self.mask_scrip_filename = out_filename

    def run(self):
        """
        Run this step of the testcase
        """
        try:
            os.makedirs(self.out_dir)
        except OSError:
            pass

        self._symlink_initial_condition()
        self._make_partition_files()
        self._make_scrip_files()

    def _symlink_initial_condition(self):
        dest_filename = self.init_filename
        symlink(
            f'../../../../../{dest_filename}',
            f'{self.out_dir}/{dest_filename}')

        dest_filename = self.forcing_filename
        symlink(
            f'../../../../../{dest_filename}',
            f'{self.out_dir}/{dest_filename}')

    def _make_partition_files(self):
        logger = self.logger
        graph_filename = self.graph_filename
        print(graph_filename)

        ncells = sum(1 for _ in open(graph_filename))
        min_graph_size = int(ncells / 6000)
        max_graph_size = int(ncells / 100)
        logger.info(f'Creating graph files between {min_graph_size} and '
                    f'{max_graph_size}')
        n_power2 = 2**np.arange(1, 21)
        n_multiples12 = 12 * np.arange(1, 9)

        ncores = n_power2
        for power10 in range(3):
            ncores = np.concatenate([ncores, 10**power10 * n_multiples12])

        for n in ncores:
            if min_graph_size <= n <= max_graph_size:
                args = ['gpmetis', graph_filename, f'{n}']
                check_call(args, logger)

        # create link in assembled files directory
        files = glob('mpas-o.graph.info.*')
        for file in files:
            symlink(f'../../../../../{file}',
                    f'{self.out_dir}/{file}')

    def _make_scrip_files(self):
        init_filename = self.init_filename
        scrip_filename = self.no_mask_scrip_filename
        scrip_from_mpas(init_filename, scrip_filename)
        symlink(f'../../../../../{scrip_filename}',
                f'{self.out_dir}/{scrip_filename}')

        scrip_filename = self.mask_scrip_filename
        scrip_from_mpas(init_filename, scrip_filename, useLandIceMask=True)
        symlink(f'../../../../../{scrip_filename}',
                f'{self.out_dir}/{scrip_filename}')

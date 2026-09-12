import os

from compass.landice.tests.ismip7_run.ismip7_ais.stage_experiment import (
    configure_experiment_namelist_and_streams,
    symlink_experiment_inputs,
)
from compass.model import make_graph_file, run_model
from compass.step import Step

# The package that contains this module, used to locate the streams
# override template below
RESOURCE_LOCATION = 'compass.landice.tests.ismip7_run.ismip7_ais'

# A short, historical-like "test" experiment used only by the
# decomposition and restart tests.  It reuses the same input/forcing
# data as the 'historical_CESM2-WACCM' experiment defined in
# compass.landice.tests.ismip7_run.ismip7_ais, but only runs for a few
# days so the tests are cheap to run.
SHORT_TEST_EXPERIMENT = 'historical_CESM2-WACCM'
SHORT_TEST_EXP_INFO = {
    'scenario': 'historical', 'model': 'CESM2-WACCM',
    'start_time': '2000-01-01_00:00:00',
    'stop_time': '2000-01-06_00:00:00',
    'is_historical': True}

# Time at which the restart test breaks the run into two segments
RESTART_TIME = '2000-01-04_00:00:00'


class RunModel(Step):
    """
    A step for performing a short forward MALI run of the ISMIP7 AIS
    configuration, for use by the decomposition and restart tests.

    Attributes
    ----------
    suffixes : list of str
        a list of suffixes for namelist and streams files produced for
        this step.  Most runs will just have a ``namelist.landice`` and
        a ``streams.landice`` (the default), but the ``restart_run``
        step of the ``restart_test`` runs the model twice, the second
        time with ``namelist.landice.rst`` and ``streams.landice.rst``
    """

    def __init__(self, test_case, name, subdir, ntasks, min_tasks=None,
                 openmp_threads=1, suffixes=None):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        name : str
            the name of the step

        subdir : str
            the subdirectory for the step

        ntasks : int
            the number of tasks the step would ideally use

        min_tasks : int, optional
            the number of tasks the step requires.  The default is
            ``ntasks``

        openmp_threads : int, optional
            the number of OpenMP threads the step will use

        suffixes : list of str, optional
            a list of suffixes for namelist and streams files produced
            for this step.  The default is ``['landice']``.  If two
            suffixes are given, the model is run twice: once for the
            first part of the simulation (without a restart) and once
            for the remainder (restarting from the restart file written
            by the first part)
        """
        if suffixes is None:
            suffixes = ['landice']
        self.suffixes = suffixes
        if min_tasks is None:
            min_tasks = ntasks
        super().__init__(test_case=test_case, name=name, subdir=subdir,
                         ntasks=ntasks, min_tasks=min_tasks,
                         openmp_threads=openmp_threads)

        self.add_output_file(filename='output/output_2d_2000.nc')

    def setup(self):
        """
        Set up the step
        """
        config = self.config
        section = config['ismip7_run_ais']
        init_cond_path = section.get('init_cond_path')

        filenames = symlink_experiment_inputs(
            self, SHORT_TEST_EXPERIMENT, SHORT_TEST_EXP_INFO)

        for index, suffix in enumerate(self.suffixes):
            exp_info = dict(SHORT_TEST_EXP_INFO)
            if len(self.suffixes) > 1 and index == 0:
                # first segment of a restart run stops partway through
                exp_info['stop_time'] = RESTART_TIME
            configure_experiment_namelist_and_streams(
                self, exp_info, filenames, out_name=suffix)

            # shorten the output/timeaveraging intervals so the tests
            # produce daily output despite their short duration, and add
            # a few extra variables useful for validating decomposition
            # and restart consistency
            self.add_namelist_options(
                options={'config_timeaveraging_interval':
                         "'0000-00-01_00:00:00'"},
                out_name=f'namelist.{suffix}')
            output_clobber_mode = 'truncate' if index == 0 else 'overwrite'
            self.add_streams_file(
                RESOURCE_LOCATION, 'streams.override.template',
                out_name=f'streams.{suffix}',
                template_replacements={
                    'output_clobber_mode': output_clobber_mode})

            if len(self.suffixes) > 1 and index == 1:
                # the second segment of a restart run picks up from the
                # restart file written partway through the first segment
                self.add_namelist_options(
                    options={'config_do_restart': '.true.',
                             'config_start_time': "'file'"},
                    out_name=f'namelist.{suffix}')

        self.add_input_file(
            filename='albany_input.yaml',
            package=RESOURCE_LOCATION,
            copy=True)

        make_graph_file(mesh_filename=init_cond_path,
                        graph_filename=os.path.join(self.work_dir,
                                                    'graph.info'))

        self.add_model_as_input()

    def run(self):
        """
        Run this step of the test case
        """
        for suffix in self.suffixes:
            run_model(step=self, namelist=f'namelist.{suffix}',
                      streams=f'streams.{suffix}')

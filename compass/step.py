import os
from lxml import etree
from importlib.resources import path
import shutil
import numpy
import stat
import grp
import progressbar

from compass.io import download, symlink
import compass.namelist
import compass.streams


class Step:
    """
    The base class for a step of a test cases, such as setting up a mesh,
    creating an initial condition, or running the MPAS core forward in time.
    The step is the smallest unit of work in compass that can be run on its
    own by a user, though users will typically run full test cases or test
    suites.

    Below, the terms "input" and "output" refer to inputs and outputs to the
    step itself, not necessarily the MPAS model.  In fact, the MPAS model
    itself is often an input to the step.

    Attributes
    ----------
    name : str
        the name of the test case

    test_case : compass.TestCase
        The test case this step belongs to

    test_group : compass.TestGroup
        The test group the test case belongs to

    mpas_core : compass.MpasCore
        The MPAS core the test group belongs to

    subdir : str
        the subdirectory for the step

    path : str
        the path within the base work directory of the step, made up of
        ``mpas_core``, ``test_group``, the test case's ``subdir`` and the
        step's ``subdir``

    cpus_per_task : int, optional
        the number of cores per task the step would ideally use.  If
        fewer cores per node are available on the system, the step will
        run on all available cores as long as this is not below
        ``min_cpus_per_task``

    min_cpus_per_task : int, optional
        the number of cores per task the step requires.  If the system
        has fewer than this number of cores per node, the step will fail

    ntasks : int, optional
        the number of tasks the step would ideally use.  If too few
        cores are available on the system to accommodate the number of
        tasks and the number of cores per task, the step will run on
        fewer tasks as long as as this is not below ``min_tasks``

    min_tasks : int, optional
        the number of tasks the step requires.  If the system has too
        few cores to accommodate the number of tasks and cores per task,
        the step will fail

    openmp_threads : int
        the number of OpenMP threads to use

    max_memory : int
        the amount of memory that the step is allowed to use in MB.
        This is currently just a placeholder for later use with task
        parallelism

    input_data : list of dict
        a list of dict used to define input files typically to be
        downloaded to a database and/or symlinked in the work directory

    inputs : list of str
        a list of absolute paths of input files produced from ``input_data`` as
        part of setting up the step.  These input files must all exist at run
        time or the step will raise an exception

    outputs : list of str
        a list of absolute paths of output files produced by this step (or
        cached) and available as inputs to other test cases and steps.  These
        files must exist after the test has run or an exception will be raised

    namelist_data : dict
        a dictionary used internally to keep track of updates to the default
        namelist options from calls to
        :py:meth:`compass.Step.add_namelist_file`
        and :py:meth:`compass.Step.add_namelist_options`

    streams_data : dict
        a dictionary used internally to keep track of updates to the default
        streams from calls to :py:meth:`compass.Step.add_streams_file`

    config : compass.config.CompassConfigParser
        Configuration options for this test case, a combination of the defaults
        for the machine, core and configuration

    machine_info : mache.MachineInfo
        Information about E3SM supported machines

    config_filename : str
        The local name of the config file that ``config`` has been written to
        during setup and read from during run

    work_dir : str
        The step's work directory, defined during setup as the combination
        of ``base_work_dir`` and ``path``

    base_work_dir : str
        The base work directory

    logger : logging.Logger
        A logger for output from the step

    log_filename : str
        At run time, the name of a log file where output/errors from the step
        are being logged, or ``None`` if output is to stdout/stderr

    cached : bool
        Whether to get all of the outputs for the step from the database of
        cached outputs for this MPAS core

    run_as_subprocess : bool
        Whether to run this step as a subprocess, rather than just running
        it directly from the test case.  It is useful to run a step as a
        subprocess if there is not a good way to redirect output to a log
        file (e.g. if the step calls external code that, in turn, calls
        additional subprocesses).
    """

    def __init__(self, test_case, name, subdir=None, cpus_per_task=1,
                 min_cpus_per_task=1, ntasks=1, min_tasks=1,
                 openmp_threads=1, max_memory=None, cached=False,
                 run_as_subprocess=False):
        """
        Create a new test case

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        name : str
            the name of the test case

        subdir : str, optional
            the subdirectory for the step.  The default is ``name``

        cpus_per_task : int, optional
            the number of cores per task the step would ideally use.  If
            fewer cores per node are available on the system, the step will
            run on all available cores as long as this is not below
            ``min_cpus_per_task``

        min_cpus_per_task : int, optional
            the number of cores per task the step requires.  If the system
            has fewer than this number of cores per node, the step will fail

        ntasks : int, optional
            the number of tasks the step would ideally use.  If too few
            cores are available on the system to accommodate the number of
            tasks and the number of cores per task, the step will run on
            fewer tasks as long as as this is not below ``min_tasks``

        min_tasks : int, optional
            the number of tasks the step requires.  If the system has too
            few cores to accommodate the number of tasks and cores per task,
            the step will fail

        openmp_threads : int
            the number of OpenMP threads to use

        max_memory : int, optional
            the amount of memory that the step is allowed to use in MB.
            This is currently just a placeholder for later use with task
            parallelism

        cached : bool, optional
            Whether to get all of the outputs for the step from the database of
            cached outputs for this MPAS core

        run_as_subprocess : bool
            Whether to run this step as a subprocess, rather than just running
            it directly from the test case.  It is useful to run a step as a
            subprocess if there is not a good way to redirect output to a log
            file (e.g. if the step calls external code that, in turn, calls
            additional subprocesses).
        """
        self.name = name
        self.test_case = test_case
        self.mpas_core = test_case.mpas_core
        self.test_group = test_case.test_group
        if subdir is not None:
            self.subdir = subdir
        else:
            self.subdir = name

        self.cpus_per_task = cpus_per_task
        self.min_cpus_per_task = min_cpus_per_task
        self.ntasks = ntasks
        self.min_tasks = min_tasks
        self.openmp_threads = openmp_threads
        self.max_memory = max_memory

        self.path = os.path.join(self.mpas_core.name, self.test_group.name,
                                 test_case.subdir, self.subdir)

        self.run_as_subprocess = run_as_subprocess

        # child steps (or test cases) will add to these
        self.input_data = list()
        self.inputs = list()
        self.outputs = list()
        self.namelist_data = dict()
        self.streams_data = dict()

        # these will be set later during setup
        self.config = None
        self.machine_info = None
        self.config_filename = None
        self.work_dir = None
        self.base_work_dir = None

        # these will be set before running the step
        self.logger = None
        self.log_filename = None

        # output caching
        self.cached = cached

    def set_resources(self, cpus_per_task=None, min_cpus_per_task=None,
                      ntasks=None, min_tasks=None, openmp_threads=None,
                      max_memory=None):
        """
        Update the resources for the subtask.  This can be done within init,
        ``setup()`` or ``runtime_setup()`` of this step, or init,
        ``configure()`` or ``run()`` for the test case that this step belongs
        to.

        Parameters
        ----------
        cpus_per_task : int, optional
            the number of cores per task the step would ideally use.  If
            fewer cores per node are available on the system, the step will
            run on all available cores as long as this is not below
            ``min_cpus_per_task``

        min_cpus_per_task : int, optional
            the number of cores per task the step requires.  If the system
            has fewer than this number of cores per node, the step will fail

        ntasks : int, optional
            the number of tasks the step would ideally use.  If too few
            cores are available on the system to accommodate the number of
            tasks and the number of cores per task, the step will run on
            fewer tasks as long as as this is not below ``min_tasks``

        min_tasks : int, optional
            the number of tasks the step requires.  If the system has too
            few cores to accommodate the number of tasks and cores per task,
            the step will fail

        openmp_threads : int, optional
            the number of OpenMP threads to use

        max_memory : int, optional
            the amount of memory that the step is allowed to use in MB.
            This is currently just a placeholder for later use with task
            parallelism
        """
        if cpus_per_task is not None:
            self.cpus_per_task = cpus_per_task
        if min_cpus_per_task is not None:
            self.min_cpus_per_task = min_cpus_per_task
        if ntasks is not None:
            self.ntasks = ntasks
        if min_tasks is not None:
            self.min_tasks = min_tasks
        if openmp_threads is not None:
            self.openmp_threads = openmp_threads
        if max_memory is not None:
            self.max_memory = max_memory

    def constrain_resources(self, available_cores):
        """
        Constrain ``cpus_per_task`` and ``ntasks`` based on the number of
        cores available to this step

        Parameters
        ----------
        available_cores : int
            The total number of cores available to the step
        """
        if self.ntasks == 1:
            # just one task so only need to worry about cpus_per_task
            self.cpus_per_task = min(self.cpus_per_task, available_cores)
            cores = self.cpus_per_task
        elif self.cpus_per_task == self.min_cpus_per_task:
            available_tasks = available_cores // self.cpus_per_task
            self.ntasks = min(self.ntasks, available_tasks)
            cores = self.ntasks * self.cpus_per_task
        else:
            raise ValueError('Unsupported step configuration in which '
                             'ntasks > 1 and  '
                             'cpus_per_task != min_cpus_per_task')

        min_cores = self.min_cpus_per_task * self.min_tasks
        if cores < min_cores:
            raise ValueError(
                f'Available cores ({cores}) is below the minimum of '
                f'{min_cores} for step {self.name}')

    def setup(self):
        """
        Set up the test case in the work directory, including downloading any
        dependencies.  The step should override this function to perform setup
        operations such as generating namelist and streams files, adding inputs
        and outputs.
        """
        pass

    def runtime_setup(self):
        """
        Update attributes of the step at runtime before calling the ``run()``
        method.  The most common reason to override this method is to
        determine the number of cores and threads to run with.  It may also
        be useful for performing small (python) tasks such as creating
        graph and partition files before running a parallel command. When
        running with parallel tasks in the future, this method will be called
        for each step in serial before steps are run in task parallel.
        """
        pass

    def run(self):
        """
        Run the step.  Every child class must override this method to perform
        the main work.
        """
        pass

    def add_input_file(self, filename=None, target=None, database=None,
                       url=None, work_dir_target=None, package=None,
                       copy=False):
        """
        Add an input file to the step (but not necessarily to the MPAS model).
        The file can be local, a symlink to a file that will be created in
        another step, a symlink to a file in one of the databases for files
        cached after download, and/or come from a specified URL.

        Parameters
        ----------
        filename : str, optional
            The relative path of the input file within the step's work
            directory. The default is the file name (without the path) of
            ``target``.

        target : str, optional
            A file that will be the target of a symlink to ``filename``.  If
            ``database`` is not specified, this should be an absolute path or a
            relative path from the step's work directory.  If ``database`` is
            specified, this is a relative path within the database and the name
            of the remote file to download.

        database : str, optional
            The name of a database for caching local files.  This will be a
            subdirectory of the local cache directory for this core.  If
            ``url`` is not provided, the URL for downloading the file will be
            determined by combining the base URL of the data server, the
            relative path for the core, ``database`` and ``target``.

        url : str, optional
            The URL (including file name) for downloading the file.  This
            option should be set if the file is not in a database on the data
            server. The ``filename``, ``target`` and ``database`` are not
            added to URL even if they are provided.

        work_dir_target : str, optional
            Same as ``target`` but with a path relative to the base work
            directory.  This is useful if it is not easy to determine the
            relative path between the step's work directory and the target.

        package : str or package, optional
            A package within ``compass`` from which the file should be linked

        copy : bool, optional
            Whether to make a copy of the file, rather than a symlink
        """
        if filename is None:
            if target is None:
                raise ValueError('At least one of local_name and target are '
                                 'required.')
            filename = os.path.basename(target)

        self.input_data.append(dict(filename=filename, target=target,
                                    database=database, url=url,
                                    work_dir_target=work_dir_target,
                                    package=package, copy=copy))

    def add_output_file(self, filename):
        """
        Add the output file that must be produced by this step and may be made
        available as an input to steps, perhaps in other test cases.  This file
        must exist after the test has run or an exception will be raised.

        Parameters
        ----------
        filename : str
            The relative path of the output file within the step's work
            directory
        """
        self.outputs.append(filename)

    def add_model_as_input(self):
        """
        make a link to the model executable and add it to the inputs
        """
        self.add_input_file(filename='<<<model>>>')

    def add_namelist_file(self, package, namelist, out_name=None,
                          mode='forward'):
        """
        Add a file with updates to namelist options to the step to be parsed
        when generating a complete namelist file if and when the step gets set
        up.

        Parameters
        ----------
        package : Package
            The package name or module object that contains ``namelist``

        namelist : str
            The name of the namelist replacements file to read from

        out_name : str, optional
            The name of the namelist file to write out, ``namelist.<core>`` by
            default

        mode : {'init', 'forward'}, optional
            The mode that the model will run in
        """
        if out_name is None:
            out_name = f'namelist.{self.mpas_core.name}'

        if out_name not in self.namelist_data:
            self.namelist_data[out_name] = list()

        namelist_list = self.namelist_data[out_name]

        namelist_list.append(dict(package=package, namelist=namelist,
                                  mode=mode))

    def add_namelist_options(self, options, out_name=None, mode='forward'):
        """
        Add the namelist replacements to be parsed when generating a namelist
        file if and when the step gets set up.

        Parameters
        ----------
        options : dict
            A dictionary of options and value to replace namelist options with
            new values

        out_name : str, optional
            The name of the namelist file to write out, ``namelist.<core>`` by
            default

        mode : {'init', 'forward'}, optional
            The mode that the model will run in
        """
        if out_name is None:
            out_name = f'namelist.{self.mpas_core.name}'

        if out_name not in self.namelist_data:
            self.namelist_data[out_name] = list()

        namelist_list = self.namelist_data[out_name]

        namelist_list.append(dict(options=options, mode=mode))

    def update_namelist_at_runtime(self, options, out_name=None):
        """
        Update an existing namelist file with additional options.  This would
        typically be used for namelist options that are only known at runtime,
        not during setup, typically those related to the number of nodes and
        cores.

        Parameters
        ----------
        options : dict
            A dictionary of options and value to replace namelist options with
            new values

        out_name : str, optional
            The name of the namelist file to write out, ``namelist.<core>`` by
            default
        """

        if out_name is None:
            out_name = f'namelist.{self.mpas_core.name}'

        print(f'Warning: replacing namelist options in {out_name}')
        for key, value in options.items():
            print(f'{key} = {value}')

        filename = os.path.join(self.work_dir, out_name)

        namelist = compass.namelist.ingest(filename)

        namelist = compass.namelist.replace(namelist, options)

        compass.namelist.write(namelist, filename)

    def update_namelist_pio(self, out_name=None):
        """
        Modify the namelist so the number of PIO tasks and the stride between
        them consistent with the number of nodes and cores (one PIO task per
        node).

        Parameters
        ----------
        out_name : str, optional
            The name of the namelist file to write out, ``namelist.<core>`` by
            default
        """
        config = self.config
        cores = self.ntasks * self.cpus_per_task

        if out_name is None:
            out_name = f'namelist.{self.mpas_core.name}'

        cores_per_node = config.getint('parallel', 'cores_per_node')

        # update PIO tasks based on the machine settings and the available
        # number or cores
        pio_num_iotasks = int(numpy.ceil(cores / cores_per_node))
        pio_stride = self.ntasks // pio_num_iotasks
        if pio_stride > cores_per_node:
            raise ValueError(f'Not enough nodes for the number of cores.  '
                             f'cores: {cores}, cores per node: '
                             f'{cores_per_node}')

        replacements = {'config_pio_num_iotasks': f'{pio_num_iotasks}',
                        'config_pio_stride': f'{pio_stride}'}

        self.update_namelist_at_runtime(options=replacements,
                                        out_name=out_name)

    def add_streams_file(self, package, streams, template_replacements=None,
                         out_name=None, mode='forward'):
        """
        Add a streams file to the step to be parsed when generating a complete
        streams file if and when the step gets set up.

        Parameters
        ----------
        package : Package
            The package name or module object that contains the streams file

        streams : str
            The name of the streams file to read from

        template_replacements : dict, optional
            A dictionary of replacements, in which case ``streams`` must be a
            Jinja2 template to be rendered with these replacements

        out_name : str, optional
            The name of the streams file to write out, ``streams.<core>`` by
            default

        mode : {'init', 'forward'}, optional
            The mode that the model will run in
        """
        if out_name is None:
            out_name = f'streams.{self.mpas_core.name}'

        if out_name not in self.streams_data:
            self.streams_data[out_name] = list()

        self.streams_data[out_name].append(
            dict(package=package, streams=streams,
                 replacements=template_replacements, mode=mode))

    def update_streams_at_runtime(self, package, streams,
                                  template_replacements, out_name=None):
        """
        Update the streams files during the run phase of this step using the
        given template and replacements.  This may be useful for updating
        streams based on config options that a user may have changed before
        running the step.

        Parameters
        ----------
        package : Package
            The package name or module object that contains the streams file

        streams : str
            The name of a Jinja2 template to be rendered with replacements

        template_replacements : dict
            A dictionary of replacements

        out_name : str, optional
            The name of the streams file to write out, ``streams.<core>`` by
            default
        """

        if out_name is None:
            out_name = f'streams.{self.mpas_core.name}'

        if template_replacements is not None:
            print(f'Warning: updating streams in {out_name} using the '
                  f'following template and replacements:')
            print(f'{package} {streams}')
            for key, value in template_replacements.items():
                print(f'{key} = {value}')

        filename = os.path.join(self.work_dir, out_name)

        tree = etree.parse(filename)
        tree = compass.streams.read(package, streams, tree=tree,
                                    replacements=template_replacements)
        compass.streams.write(tree, filename)

    def process_inputs_and_outputs(self):
        """
        Process the inputs to and outputs from a step added with
        :py:meth:`compass.Step.add_input_file` and
        :py:meth:`compass.Step.add_output_file`.  This includes downloading
        files, making symlinks, and converting relative paths to absolute
        paths.

        Also generates namelist and streams files
       """
        mpas_core = self.mpas_core.name
        step_dir = self.work_dir
        config = self.config

        # process the outputs first because cached outputs will add more inputs
        if self.cached:
            # forget about the inputs -- we won't used them, but we will add
            # the cached outputs as inputs
            self.input_data = list()
            for output in self.outputs:
                filename = os.path.join(self.path, output)
                if filename not in self.mpas_core.cached_files:
                    raise ValueError(f'The file {filename} has not been added '
                                     f'to the cache database')
                target = self.mpas_core.cached_files[filename]
                self.add_input_file(
                    filename=output,
                    target=target,
                    database='compass_cache')

        inputs = []
        databases_with_downloads = set()
        for entry in self.input_data:
            filename = entry['filename']
            target = entry['target']
            database = entry['database']
            url = entry['url']
            work_dir_target = entry['work_dir_target']
            package = entry['package']
            copy = entry['copy']

            if filename == '<<<model>>>':
                model = self.config.get('executables', 'model')
                filename = os.path.basename(model)
                target = os.path.abspath(model)

            if package is not None:
                if target is None:
                    target = filename
                with path(package, target) as package_path:
                    target = str(package_path)

            if work_dir_target is not None:
                target = os.path.join(self.base_work_dir, work_dir_target)

            if target is not None:
                download_target = target
            else:
                download_target = filename

            download_path = None

            if database is not None:
                # we're downloading a file to a cache of a database (if it's
                # not already there.
                if url is None:
                    base_url = config.get('download', 'server_base_url')
                    core_path = config.get('download', 'core_path')
                    url = f'{base_url}/{core_path}/{database}'

                    url = f'{url}/{target}'

                database_root = config.get(
                    'paths', f'{mpas_core}_database_root')
                download_path = os.path.join(database_root, database,
                                             download_target)
                if not os.path.exists(download_path):
                    database_subdir = os.path.join(database_root, database)
                    databases_with_downloads.add(database_subdir)
            elif url is not None:
                download_path = download_target

            if url is not None:
                download_target = download(url, download_path, config)
                if target is not None:
                    # this is the absolute path that we presumably want
                    target = download_target

            if target is not None:
                filepath = os.path.join(step_dir, filename)
                if copy:
                    shutil.copy(target, filepath)
                else:
                    symlink(target, filepath)
                inputs.append(target)
            else:
                inputs.append(filename)

        if len(databases_with_downloads) > 0:
            self._fix_permissions(databases_with_downloads)

        # convert inputs and outputs to absolute paths
        self.inputs = [os.path.abspath(os.path.join(step_dir, filename)) for
                       filename in inputs]

        self.outputs = [os.path.abspath(os.path.join(step_dir, filename)) for
                        filename in self.outputs]

        self._generate_namelists()
        self._generate_streams()

    def _generate_namelists(self):
        """
        Writes out a namelist file in the work directory with new values given
        by parsing the files and dictionaries in the step's ``namelist_data``.
        """

        if self.cached:
            # no need for namelists
            return

        step_work_dir = self.work_dir
        config = self.config

        for out_name in self.namelist_data:

            replacements = dict()

            mode = None

            for entry in self.namelist_data[out_name]:
                if mode is None:
                    mode = entry['mode']
                else:
                    assert mode == entry['mode']
                if 'options' in entry:
                    # this is a dictionary of replacement namelist options
                    options = entry['options']
                else:
                    options = compass.namelist.parse_replacements(
                        entry['package'], entry['namelist'])
                replacements.update(options)

            defaults_filename = config.get('namelists', mode)
            out_filename = f'{step_work_dir}/{out_name}'

            namelist = compass.namelist.ingest(defaults_filename)

            namelist = compass.namelist.replace(namelist, replacements)

            compass.namelist.write(namelist, out_filename)

    def _generate_streams(self):
        """
        Writes out a streams file in the work directory with new values given
        by parsing the files and dictionaries in the step's ``streams_data``.
        """
        if self.cached:
            # no need for streams
            return

        step_work_dir = self.work_dir
        config = self.config

        for out_name in self.streams_data:

            # generate the streams file
            tree = None

            mode = None

            for entry in self.streams_data[out_name]:
                if mode is None:
                    mode = entry['mode']
                else:
                    assert mode == entry['mode']

                tree = compass.streams.read(
                    package=entry['package'],
                    streams_filename=entry['streams'],
                    replacements=entry['replacements'], tree=tree)

            defaults_filename = config.get('streams', mode)
            out_filename = f'{step_work_dir}/{out_name}'

            defaults_tree = etree.parse(defaults_filename)

            defaults = next(defaults_tree.iter('streams'))
            streams = next(tree.iter('streams'))

            for stream in streams:
                compass.streams.update_defaults(stream, defaults)

            # remove any streams that aren't requested
            for default in defaults:
                found = False
                for stream in streams:
                    if stream.attrib['name'] == default.attrib['name']:
                        found = True
                        break
                if not found:
                    defaults.remove(default)

            compass.streams.write(defaults_tree, out_filename)

    def _fix_permissions(self, databases):
        """
        Fix permissions on the databases where files were downloaded so
        everyone in the group can read/write to them
        """
        config = self.config

        if not config.has_option('e3sm_unified', 'group'):
            return

        group = config.get('e3sm_unified', 'group')

        new_uid = os.getuid()
        new_gid = grp.getgrnam(group).gr_gid

        write_perm = (stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP |
                      stat.S_IWGRP | stat.S_IROTH)
        exec_perm = (stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR |
                     stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP |
                     stat.S_IROTH | stat.S_IXOTH)

        mask = stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO

        print('changing permissions on downloaded files')

        # first the base directories that don't seem to be included in
        # os.walk()
        for directory in databases:
            try:
                dir_stat = os.stat(directory)
            except OSError:
                continue

            perm = dir_stat.st_mode & mask

            if dir_stat.st_uid != new_uid:
                # current user doesn't own this dir so let's move on
                continue

            if perm == exec_perm and dir_stat.st_gid == new_gid:
                continue

            try:
                os.chown(directory, new_uid, new_gid)
                os.chmod(directory, exec_perm)
            except OSError:
                continue

        files_and_dirs = []
        for base in databases:
            for root, dirs, files in os.walk(base):
                files_and_dirs.extend(dirs)
                files_and_dirs.extend(files)

        widgets = [progressbar.Percentage(), ' ', progressbar.Bar(),
                   ' ', progressbar.ETA()]
        bar = progressbar.ProgressBar(widgets=widgets,
                                      maxval=len(files_and_dirs)).start()
        progress = 0
        for base in databases:
            for root, dirs, files in os.walk(base):
                for directory in dirs:
                    progress += 1
                    bar.update(progress)

                    directory = os.path.join(root, directory)

                    try:
                        dir_stat = os.stat(directory)
                    except OSError:
                        continue

                    if dir_stat.st_uid != new_uid:
                        # current user doesn't own this dir so let's move on
                        continue

                    perm = dir_stat.st_mode & mask

                    if perm == exec_perm and dir_stat.st_gid == new_gid:
                        continue

                    try:
                        os.chown(directory, new_uid, new_gid)
                        os.chmod(directory, exec_perm)
                    except OSError:
                        continue

                for file_name in files:
                    progress += 1
                    bar.update(progress)
                    file_name = os.path.join(root, file_name)
                    try:
                        file_stat = os.stat(file_name)
                    except OSError:
                        continue

                    if file_stat.st_uid != new_uid:
                        # current user doesn't own this file so let's move on
                        continue

                    perm = file_stat.st_mode & mask

                    if perm & stat.S_IXUSR:
                        # executable, so make sure others can execute it
                        new_perm = exec_perm
                    else:
                        new_perm = write_perm

                    if perm == new_perm and file_stat.st_gid == new_gid:
                        continue

                    try:
                        os.chown(file_name, new_uid, new_gid)
                        os.chmod(file_name, new_perm)
                    except OSError:
                        continue

        bar.finish()
        print('  done.')

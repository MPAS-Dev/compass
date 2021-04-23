import os
import stat
from jinja2 import Template
from importlib import resources
from lxml import etree

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

    cores : int
        the number of cores the step would ideally use.  If fewer cores
        are available on the system, the step will run on all available
        cores as long as this is not below ``min_cores``

    min_cores : int
        the number of cores the step requires.  If the system has fewer
        than this number of cores, the step will fail

    threads : int
        the number of threads the step will use

    max_memory : int
        the amount of memory that the step is allowed to use in MB.
        This is currently just a placeholder for later use with task
        parallelism

    max_disk : int
        the amount of disk space that the step is allowed to use in MB.
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
        a list of absolute paths of output files produced by this step and
        available as inputs to other test cases and steps.  These files must
        exist after the test has run or an exception will be raised

    namelist_data : dict
        a dictionary used internally to keep track of updates to the default
        namelist options from calls to
        :py:meth:`compass.Step.add_namelist_file`
        and :py:meth:`compass.Step.add_namelist_options`

    streams_data : dict
        a dictionary used internally to keep track of updates to the default
        streams from calls to :py:meth:`compass.Step.add_streams_file`

    config : configparser.ConfigParser
        Configuration options for this test case, a combination of the defaults
        for the machine, core and configuration

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
    """

    def __init__(self, test_case, name, subdir=None, cores=1, min_cores=1,
                 threads=1, max_memory=1000, max_disk=1000):
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

        cores : int, optional
            the number of cores the step would ideally use.  If fewer cores
            are available on the system, the step will run on all available
            cores as long as this is not below ``min_cores``

        min_cores : int, optional
            the number of cores the step requires.  If the system has fewer
            than this number of cores, the step will fail

        threads : int, optional
            the number of threads the step will use

        max_memory : int, optional
            the amount of memory that the step is allowed to use in MB.
            This is currently just a placeholder for later use with task
            parallelism

        max_disk : int, optional
            the amount of disk space that the step is allowed to use in MB.
            This is currently just a placeholder for later use with task
            parallelism
        """
        self.name = name
        self.test_case = test_case
        self.mpas_core = test_case.mpas_core
        self.test_group = test_case.test_group
        if subdir is not None:
            self.subdir = subdir
        else:
            self.subdir = name

        self.cores = cores
        self.min_cores = min_cores
        self.threads = threads
        self.max_memory = max_memory
        self.max_disk = max_disk

        self.path = os.path.join(self.mpas_core.name, self.test_group.name,
                                 test_case.subdir, self.subdir)

        # child steps (or test cases) will add to these
        self.input_data = list()
        self.inputs = list()
        self.outputs = list()
        self.namelist_data = dict()
        self.streams_data = dict()

        # these will be set later during setup
        self.config = None
        self.config_filename = None
        self.work_dir = None
        self.base_work_dir = None

        # these will be set before running the step
        self.logger = None
        self.log_filename = None

    def setup(self):
        """
        Set up the test case in the work directory, including downloading any
        dependencies.  The step should override this function to perform setup
        operations such as generating namelist and streams files, adding inputs
        and outputs.
        """
        pass

    def run(self):
        """
        Run the step.  The step should override this function to perform the
        main work.
        """
        pass

    def add_input_file(self, filename=None, target=None, database=None,
                       url=None, work_dir_target=None):
        """
        Add an input file to the step.  The file can be local, a symlink to
        a file that will be created in another step, a symlink to a file in one
        of the databases for files cached after download, and/or come from a
        specified URL.

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
            The base URL for downloading ``target`` (if provided, or
            ``filename`` if not).  This option should be set if the file is not
            in a database on the data server. The file's URL is determined by
            combining ``url`` with the filename (without the directory) from
            ``target`` (or ``filename`` if ``target`` is not provided).
            ``database`` is not included in the file's URL even if it is
            provided.

        work_dir_target : str, optional
            Same as ``target`` but with a path relative to the base work
            directory.  This is useful if it is not easy to determine the
            relative path between the step's work directory and the target.
        """
        if filename is None:
            if target is None:
                raise ValueError('At least one of local_name and target are '
                                 'required.')
            filename = os.path.basename(target)

        self.input_data.append(dict(filename=filename, target=target,
                                    database=database, url=url,
                                    work_dir_target=work_dir_target))

    def add_output_file(self, filename):
        """
        Add the output file to the step

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
        model = self.config.get('executables', 'model')
        model_basename = os.path.basename(model)
        self.add_input_file(filename=model_basename,
                            target=os.path.abspath(model))

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
            out_name = 'namelist.{}'.format(self.mpas_core.name)

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
            new values.

        out_name : str, optional
            The name of the namelist file to write out, ``namelist.<core>`` by
            default

        mode : {'init', 'forward'}, optional
            The mode that the model will run in
        """
        if out_name is None:
            out_name = 'namelist.{}'.format(self.mpas_core.name)

        if out_name not in self.namelist_data:
            self.namelist_data[out_name] = list()

        namelist_list = self.namelist_data[out_name]

        namelist_list.append(dict(options=options, mode=mode))

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
            out_name = 'streams.{}'.format(self.mpas_core.name)

        if out_name not in self.streams_data:
            self.streams_data[out_name] = list()

        self.streams_data[out_name].append(
            dict(package=package, streams=streams,
                 replacements=template_replacements, mode=mode))

    def generate(self):
        """
        Generate a ``run.py`` script for the step in the work directory.
        This is the script that a user can call to run the step on its own.
        """

        self._process_inputs_and_outputs()
        self._generate_namelists()
        self._generate_streams()

        template = Template(
            resources.read_text('compass.step', 'step.template'))
        test_case = {'name': self.test_case.name}
        step = {'name': self.name,
                'config_filename': self.config_filename}
        work_dir = self.work_dir
        script = template.render(test_case=test_case, step=step)

        run_filename = os.path.join(work_dir, 'run.py')
        with open(run_filename, 'w') as handle:
            handle.write(script)

        # make sure it has execute permission
        st = os.stat(run_filename)
        os.chmod(run_filename, st.st_mode | stat.S_IEXEC)

    def _process_inputs_and_outputs(self):
        """
        Process the inputs to and outputs from a step added with
        :py:meth:`compass.Step.add_input_file` and
        :py:meth:`compass.Step.add_output_file`.  This includes downloading
        files, making symlinks, and converting relative paths to absolute
        paths.
       """
        mpas_core = self.mpas_core.name
        step_dir = self.work_dir
        config = self.config

        inputs = []
        for entry in self.input_data:
            filename = entry['filename']
            target = entry['target']
            database = entry['database']
            url = entry['url']
            work_dir_target = entry['work_dir_target']

            if work_dir_target is not None:
                target = os.path.join(self.base_work_dir, work_dir_target)

            download_target = None
            download_path = None

            if database is not None:
                # we're downloading a file to a cache of a database (if it's
                # not already there.
                if url is None:
                    base_url = config.get('download', 'server_base_url')
                    core_path = config.get('download', 'core_path')
                    url = '{}/{}/{}'.format(base_url, core_path, database)

                if target is None:
                    target = filename

                download_target = target

                database_root = config.get(
                    'paths', '{}_database_root'.format(mpas_core))
                download_path = os.path.join(database_root, database)
            elif url is not None:
                if target is None:
                    download_target = filename
                    download_path = '.'
                else:
                    download_path, download_target = os.path.split(target)

            if url is not None:
                download_target = download(download_target, url, config,
                                           download_path)
                if target is not None:
                    # this is the absolute path that we presumably want
                    target = download_target

            if target is not None:
                symlink(target, os.path.join(step_dir, filename))
                inputs.append(target)
            else:
                inputs.append(filename)

        # convert inputs and outputs to absolute paths
        self.inputs = [os.path.abspath(os.path.join(step_dir, filename)) for
                       filename in inputs]

        self.outputs = [os.path.abspath(os.path.join(step_dir, filename)) for
                        filename in self.outputs]

    def _generate_namelists(self):
        """
        Writes out a namelist file in the work directory with new values given
        by parsing the files and dictionaries in the step's ``namelist_data``.
        """

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
            out_filename = '{}/{}'.format(step_work_dir, out_name)

            namelist = compass.namelist.ingest(defaults_filename)

            namelist = compass.namelist.replace(namelist, replacements)

            compass.namelist.write(namelist, out_filename)

    def _generate_streams(self):
        """
        Writes out a streams file in the work directory with new values given
        by parsing the files and dictionaries in the step's ``streams_data``.
        """

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
            out_filename = '{}/{}'.format(step_work_dir, out_name)

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

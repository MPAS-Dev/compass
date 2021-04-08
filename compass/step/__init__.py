import os
import stat
from jinja2 import Template
from importlib import resources

from compass.io import download, symlink


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
    """

    def __init__(self, name, subdir=None, cores=1, min_cores=1,
                 max_memory=1000, max_disk=1000):
        """
        Create a new test case

        Parameters
        ----------
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
        self.mpas_core = None
        self.test_group = None
        self.test_case = None

        self.cores = cores
        self.min_cores = min_cores
        self.max_memory = max_memory
        self.max_disk = max_disk

        self.inputs = list()
        self.outputs = list()

        self.config = None
        self.config_filename = None

        self.logger = None
        self.log_filename = None

        if subdir is not None:
            self.subdir = subdir
        else:
            self.subdir = name

        self.path = None
        self.work_dir = None
        self.base_work_dir = None

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
                       url=None):
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
        """
        if filename is None:
            if target is None:
                raise ValueError('At least one of local_name and target are '
                                 'required.')
            filename = os.path.basename(target)

        self.inputs.append(dict(filename=filename, target=target,
                                database=database, url=url))

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

    def generate(self):
        """
        Generate a ``run.py`` script for the test case or step.
        """

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

    def process_inputs_and_outputs(self):
        """
        Process the inputs to and outputs from a step added with
        :py:meth:`compass.Step.add_input_file` and
        :py:meth:`compass.Step.add_output_file`.  This includes downloading
        files, making symlinks, and converting relative paths to absolute
        paths.
       """
        mpas_core = self.mpas_core
        step_dir = self.work_dir
        config = self.config

        inputs = []
        for entry in self.inputs:
            filename = entry['filename']
            target = entry['target']
            database = entry['database']
            url = entry['url']

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

.. _dev_framework:

Framework
=========

All of the :ref:`dev_packages` that are not in the three cores (``examples``,
``landice`` and ``ocean``) belong to the ``compass`` framework.  Some of these
modules and packages are used by the :ref:`dev_command_line`, while others are
meant to be called within test cases and steps to simplify tasks like adding
input and output files, downloading data sets, building up config files,
namelists and streams files, set up and run the MPAS model, and verify the
output by comparing steps with one another or against a baseline.

.. _dev_config:

Configuration
-------------

The ``compass.config`` includes functions for creating and manipulating config
options and :ref:`config_files`.


The :py:func:`compass.config.add_config()` function can be used to add the
contents of a config file within a package to the current config parser.
Examples of this can be found in most test cases as well as
:py:func:`compass.setup.setup_case()`. Here is a typical example from
:py:func:`compass.landice.tests.enthalpy_benchmark.A.configure()`:

.. code-block:: python

    add_config(config, 'compass.landice.tests.enthalpy_benchmark.A',
               'A.cfg', exception=True)

The second and third arguments are the name of a package containing the config
file and the name of the config file itself, respectively.  You can see that
the file is in the path ``compass/landice/tests/enthalpy_benchmark/A``
(replacing the ``.`` in the module name with ``/``).  In this case, we know
that the config file should always exist, so we would like the code to raise
and exception (``exception=True``) if the file is not found.  This is the
default behavior.  In some cases, you would like the code to add the config
options if the config file exists and do nothing if it does not.  This can
be useful if a common ``configure()`` function is being used for all test
cases in a configuration, as in this example from
:py:func:`compass.ocean.tests.global_ocean.configure()`:

.. code-block:: python

    name = testcase['name']
    add_config(config, 'compass.ocean.tests.global_ocean.{}'.format(name),
               '{}.cfg'.format(name), exception=False)

When this is called within the ``mesh`` test case, nothing will happend because
``compass/ocean/tests/global_ocean/mesh`` does not contain a ``mesh.cfg`` file.
The config files for meshes are handled differently, since they aren't
associated with a particular test case:

.. code-block:: python

    mesh_name = testcase['mesh_name']
    package, prefix = get_mesh_package(mesh_name)
    add_config(config, package, '{}.cfg'.format(prefix), exception=True)

In this case, we get the package for the mesh using
:py:func:`compass.ocean.tests.global_ocean.mesh.mesh.get_mesh_package()`,
(e.g. ``compass.ocean.tests.global_ocean.mesh.qu240`` for the ``QU240`` and
``QUwISC240`` meshes), then it adds config options from this file.  Since we
require each mesh to have config options (to define the vertical grid and the
metadata to be added to the mesh, at the very least), we use ``exception=True``
so an exception will be raised if no config file is found.

The ``config`` module also contains 3 functions that are intended for internal
use by the framework itself. Test-case developers will typically not need to
call these functions directly.

The :py:func:`compass.config.duplicate_config()` function can be used to make a
deep copy of a ``config`` object to changes can be made without affecting the
original.

The :py:func:`compass.config.ensure_absolute_paths()` function is used
internally by the framework to check update config options in the ``paths``,
``namelists``, ``streams``, and ``executables`` sections of the config file
have absolute paths, using the location one of the commands from the
:ref:`dev_command_line` were called.

The :py:func:`compass.config.get_source_file()` function is used to get an
absolute path for a file using one of the config options defined in the
``paths`` section.  This function is used by the framework as part of
downloading files (e.g. to a defined database), see :ref:`dev_io`.

.. _dev_logging:

Logging
-------

Compass does not have its own module for logging, instead making use of
``mpas_tools.logging``.  This is because a common strategy for logging to
either stdout/stderr or to a log file is needed between ``compass`` and
``mpas_tools``.  To get details on how this module works in general, see
`MPAS-Tools' Logging <http://mpas-dev.github.io/MPAS-Tools/stable/logging.html>`_
as well as the APIs for :py:class:`mpas_tools.logging.LoggingContext` and
:py:func:`mpas_tools.logging.check_call`.

For the most part, the ``compass`` framework handles logging for you, so
test-case developers won't have to create your own ``logger`` objects.  They
are arguments to the test case's :ref:`dev_testcase_run` or step's
:ref:`dev_step_run`.  If you run a step on its own, no log file is created
and logging happens to ``stdout``/``stderr``.  If you run the full test case,
each step gets logged to its own log file within the test case's work
directory.  If you run a test suite, each test case and its steps get logged
to a file in the ``case_output`` directory of the suite's work directory.

Although the logger will capture ``print`` statements, anywhere with a
``run()`` function or the functions called inside that function, it is a good
idea to call ``logger.info`` instead of ``print`` to be explicit about the
expectation that the output may go to a log file.

Even more important, subprocesses that produce output should always be called
with :py:func:`mpas_tools.logging.check_call`, passing in the ``logger`` that
is an argument to the ``run()`` function.  Otherwise, output will go to
``stdout``/``stderr`` even when the intention is to write all output to a
log file.  Whereas logging can capture ``stdout``/``stderr`` to make sure that
the ``print`` statements actually go to log files when desired, there is no
similar trick for automatically capturing the output from direct calls to
``subprocess`` functions.  Here is a code snippet from
:py:func:`compass.landice.tests.dome.setup_mesh.run()`:

.. code-block:: python

    from mpas_tools.logging import check_call


    def run(step, test_suite, config, logger):
        ...
        section = config['dome']
        ...
        levels = section.getfloat('levels')
        args = ['create_landice_grid_from_generic_MPAS_grid.py',
                '-i', 'mpas_grid.nc',
                '-o', 'landice_grid.nc',
                '-l', levels]

        check_call(args, logger)
        ...


This example calls the script ``create_landice_grid_from_generic_MPAS_grid.py``
from ``mpas_tools`` with several arguments, making use of the ``logger``.

.. _dev_io:

IO
--

The most common functions for test-case developers to use from the
``compass.io`` module are :py:func:`compass.io.add_input_file()` and
:py:func:`compass.io.add_output_file()`.

.. _dev_io_input:

Input files
^^^^^^^^^^^

Typically, a step will add input files with
:py:func:`compass.io.add_input_file()` in its :ref:`dev_step_collect`: or
:ref:`dev_step_setup` function.  It is also possible to add inputs in the
test case's :ref:`dev_testcase_collect` function.

It is possible to simply supply the path to an input file as ``filename``
without any other arguments to ``add_input_file()``.  In this case, the file
name is either an absolute path or a relative path with respect to the step's
work directory:

.. code-block:: python

    from compass.io import add_input_file

    def collect(testcase, step):
        ...
        add_input_file(step, filename='../setup_mesh/landice_grid.nc')

This is not typically how ``add_input_file()`` is used because input files are
usually not directly in the step's work directory.

.. _dev_io_input_symlinks:

Symlinks to input files
^^^^^^^^^^^^^^^^^^^^^^^
The most common type of input file is the output from another step. Rather than
just giving the file name directly, as in the example above, the preference is
to place a symbolic link in the work directory.  This makes it much easier to
see if the file is missing (because symlink will show up as broken) and allows
you to refer to a short, local name for the file rather than its full path:

.. code-block:: python

    import xarray

    from compass.io import add_input_file


    def collect(testcase, step):
        ...
        add_input_file(step, filename='landice_grid.nc',
                       target='../setup_mesh/landice_grid.nc')

    ...

    def run(step, test_suite, config, logger):
       ...
       with xarray.open_dataset('landice_grid.nc') as ds:
           ...

A symlink is not actually created when ``add_input_file()`` is called.  This
will not happen until the step gets set up, after calling its
:ref:`dev_step_setup` function (if any).

.. _dev_io_input_compass:

Input files from compass
^^^^^^^^^^^^^^^^^^^^^^^^

Another common need is to symlink a data file from within the configuration or
test case:

.. code-block:: python

    from importlib.resources import path

    from compass.io import add_input_file


    def collect(testcase, step):
        ...
        filename = 'enthA_analy_result.mat'
        with path('compass.landice.tests.enthalpy_benchmark.A', filename) as \
                target:
            add_input_file(step, filename=filename, target=str(target))

Here, we use a :py:class:`importlib.resources.path` object as the target of the
symlink (converting it to a string: ``str(target)``), which lets python take
care of figuring out where ``compass`` is installed so it can find the path to
the resource.

.. _dev_io_input_download:

Downloading input files
^^^^^^^^^^^^^^^^^^^^^^^

The final type of input file is one that is downloaded and stored locally.
Typically, to save ourselves the time of downloading large files and to reduce
potential problems on systems with firewalls, we cache the downloaded files in
a location where they can be shared between users and reused over time.  These
"databases" are subdirectories of the core's database root on the
`LCRC server <https://web.lcrc.anl.gov/public/e3sm/mpas_standalonedata/>`_.

To add an input file from a database, call ``add_input_file()`` with the
``database`` argument:

.. code-block:: python

    add_input_file(
        step,  filename='topography.nc',
        target='BedMachineAntarctica_and_GEBCO_2019_0.05_degree.200128.nc',
        database='bathymetry_database')

In this example from
:py:func:`compass.ocean.tests.global_ocean.init.initial_state.setup()`, the
file ``BedMachineAntarctica_and_GEBCO_2019_0.05_degree.200128.nc`` slated for
later downloaded from
`MPAS-Ocean's bathymetry database <https://web.lcrc.anl.gov/public/e3sm/mpas_standalonedata/mpas-ocean/bathymetry_database/>`_.
The file will be stored in the subdirectory ``bathymetry_database`` of the path
in the ``ocean_database_root`` config option in the ``paths`` section of the
config file.  The ``ocean_database_root`` option (or the equivalent for other
cores) is set either by selecting one of the :ref:`supported_machines` or in
the user's config file.

It is also possible to download files directly from a URL and store them in
the step's working directory:

.. code-block:: python

    add_input_file(
        step,  filename='dome_varres_grid.nc',
        url='https://web.lcrc.anl.gov/public/e3sm/mpas_standalonedata/'
            'mpas-albany-landice/dome_varres_grid.nc')

We recommend against this practice except for very small files.

.. _dev_io_output:

Output files
^^^^^^^^^^^^

We require that all steps provide a list of any output files that other steps
are allowed to use as inputs.  This helps us keep track of dependencies and
will be used in the future to enable steps to run in parallel as long as they
don't depend on each other.  Adding an output files is pretty straightforward:

.. code-block:: python

    add_output_file(step, filename='output_file.nc')

:py:func:`compass.io.add_output_file()` can be called in a step's
:ref:`dev_step_collect`: or :ref:`dev_step_setup` function or (less commonly)
in the test case's :ref:`dev_testcase_collect` function.

The relative path in ``filename`` is with respect to the step's work directory,
and is converted to an absolute path internally before the step is run.


.. _dev_io_symlink:

Symlinks
^^^^^^^^

You can also create your own symlinks that aren't input files (e.g. for a
README file that the user might want to have available) using
:py:func:`compass.io.symlink()`:

.. code-block:: python

    from importlib.resources import path

    from compass.io import symlink


    def configure(testcase, config):
        ...
        with path('compass.ocean.tests.global_ocean.files_for_e3sm', 'README') as \
                target:
            symlink(str(target), '{}/README'.format(testcase['work_dir']))

In this example, we get the path to a README file within ``compass`` and make
a local symlink to it in the test case's work directory.  We did this with
``symlink()`` rather than ``add_input_file()`` because we want this link to
be within the test case's work directory, not the step's work directory.  We
must do this in ``configure()`` rather than ``collect()`` because we do not
know if the test case will be set up at all (or in what work directory) during
``collect()``.


.. _dev_io_download:

Download
^^^^^^^^

You can download files more directly if you need to using
:py:func:`compass.io.download()`, though we recommend using
:py:func:`compass.io.add_input_file()` whenever possible because it is more
flexible and takes care of more of the details of symlinking the local file
and adding it as an input to the step.  No current test cases use
``download()`` directly, but an example might look like this:

.. code-block:: python

    from compass.io import symlink, download

    def setup(step, config):

        step_dir = step['work_dir']
        database_root = config.get('paths', 'ocean_database_root')
        download_path = os.path.join(database_root, 'bathymetry_database')

        remote_filename = \
            'BedMachineAntarctica_and_GEBCO_2019_0.05_degree.200128.nc'
        local_filename = 'topography.nc'

        download(
            file_name=remote_filename,
            url='https://web.lcrc.anl.gov/public/e3sm/mpas_standalonedata/'
                'mpas-ocean/bathymetry_database',
            config=config, dest_path=download_path)

        symlink(os.path.join(download_path, remote_filename),
                os.path.join(step_dir, 'topography.nc'))

In this example, the remote file
`BedMachineAntarctica_and_GEBCO_2019_0.05_degree.200128.nc <https://web.lcrc.anl.gov/public/e3sm/mpas_standalonedata/mpas-ocean/bathymetry_databaseBedMachineAntarctica_and_GEBCO_2019_0.05_degree.200128.nc>`_
gets downloaded into the bathymetry database (if it's not already there).
Then, we create a local symlink called ``topography.nc`` to the file in the
bathymetry database.

.. _dev_model:

Model
-----

Running MPAS
^^^^^^^^^^^^

If a step involves running MPAS, the model executable can be linked and added
as an input by calling :py:func:`compass.model.add_model_as_input()`.  This
way, if the user has forgotten to compile the model, this will be obvious by
the broken symlink and the step will immediately fail because of the missing
input.  The path to the executable is automatically detected based on the
work directory for the step and the config options.

To run MPAS, call :py:func:`compass.model.run_model()`.  By default, this
function first updates the namelist options associated with the
`PIO library <https://ncar.github.io/ParallelIO/>`_ and partition the mesh
across MPI tasks, as we sill discuss in a moment, before running the model.
You can provide non-default names for the graph, namelist and streams files.
The number of cores and threads is determined from the `step` dictionary and
must be set in the step's :ref:`dev_step_collect` or :ref:`dev_step_setup`
(i.e. before calling :ref:`dev_step_run`) so that the ``compass`` framework can
ensure that the required resources are available.

Partitioning the mesh
^^^^^^^^^^^^^^^^^^^^^

The function :py:func:`compass.model.partition()` calls the graph partitioning
executable (`gpmetis <https://arc.vt.edu/userguide/metis/>`_ by default) to
divide up the MPAS mesh across cores.  If you call
:py:func:`compass.model.run_model()` with `partition_graph=True` (the default),
this function is called automatically.

In some circumstances, a step may need to partition the mesh separately from
running the model.  Typically, this applies to cases where the model is run
multiple times with the same partition and we don't want to waste time
creating the same partition over and over.  For such cases, you can call
:py:func:`compass.model.partition()` and then provide `partition_graph=False`
to later calls to :py:func:`compass.model.run_model()`.

Updating PIO namelist options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use :py:func:`compass.model.update_namelist_pio()` to automatically set
the MPAS namelist options ``config_pio_num_iotasks`` and ``config_pio_stride``
such that there is 1 PIO task per node of the MPAS run.  This is particularly
useful for PIO v1, which we have found performs much better in this
configuration than when there is 1 PIO task per core, the MPAS default.  When
running with PIO v2, we have found little performance difference between the
MPAS default and the ``compass`` default of one task per node, so we feel this
is a safe default.

By default, this function is called within :py:func:`compass.model.run_model()`.
If the same namelist file is used for multiple model runs, it may be useful to
update the number of PIO tasks only once.  In this case, use
``update_pio=False`` when calling ``run_model()`` after call
:py:func:`compass.model.update_namelist_pio()` yourself.

If you wish to use the MPAS default behavior of 1 PIO task per core, or wish to
set ``config_pio_num_iotasks`` and ``config_pio_stride`` yourself, simply
use ``update_pio=False`` when calling ``run_model()``.


Making a graph file
^^^^^^^^^^^^^^^^^^^

Some ``compass`` test cases take advantage of the fact that the
`MPAS-Tools cell culler <http://mpas-dev.github.io/MPAS-Tools/stable/mesh_conversion.html#cell-culler>`_
can produce a graph file as part of the process of culling cells from an
MPAS mesh.  In test cases that do not require cells to be culled, you can
call :py:func:`compass.model.make_graph_file()` to produce a graph file from
an MPAS mesh file.  Optionally, you can provide the name of an MPAS field on
cells in the mesh file that gives different weight to different cells
(``weight_field``) in the partitioning process.

.. _dev_namelist:

Namelist
--------

Cores, configurations, and test cases can provide namelist files that are used
to replace default namelist options before MPAS gets run.  Namelist files
within the ``compass`` package must start with the prefix ``namelist.`` to
ensure that they are included when we build the package.

Adding a namelist file to a step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Typically, a step that runs MPAS will include one or more calls to
:py:func:`compass.namelist.add_namelist_file()` within :ref:`dev_step_collect`
or :ref:`dev_step_setup`.  Calling this function simply adds the file to a
list within the ``step`` dictionary that will be parsed if an when
:py:func:`compass.namelist.generate_namelist()` gets called to create the
namelist.  (This way, it is safe to add namelist files to a step in
``collect()`` even if that test case will never get set up or run.)

The format of the namelist file is simply a list of namelist options and
the replacement values:

.. code-block:: none

    config_write_output_on_startup = .false.
    config_run_duration = '0000_00:15:00'
    config_use_mom_del2 = .true.
    config_implicit_bottom_drag_coeff = 1.0e-2
    config_use_cvmix_background = .true.
    config_cvmix_background_diffusion = 0.0
    config_cvmix_background_viscosity = 1.0e-4

Since all MPAS namelist options must have unique names, we do not worry about
which specific namelist within the file each belongs to.

A typical namelist file is added by passing the ``step`` dictionary, along with
a package where the namelist file is located and the name of the input namelist
file within that package:

.. code-block:: python

    add_namelist_file(step, 'compass.ocean.tests.baroclinic_channel',
                      'namelist.forward')

If the namelist should have a different name than the default
(``namelist.<core>``), the name can be given via the ``out_name`` keyword
argument.

Namelist values are replaced by the files (or options, see below) in the
sequence they are given.  This way, you can add the namelist substitutions for
the configuration first, and then override those with the replacements for
the test case or step.

Adding namelist options to a step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes, it is easier to replace namelist options using a dictionary within
the code, rather than a namelist file.  This is appropriate when there are only
1 or 2 options to replace (so creating a file seems like overkill) or when the
namelist options rely on values that are determined by the code (e.g. different
values for different resolutions).  Simply create a dictionary of replacements
and call :py:func:`compass.namelist.add_namelist_options()` at either the
``collect()`` or ``setup()`` stage of the test case.  These replacements are
parsed, along with replacements from files, in the order they are added.  Thus,
you could add replacements from a namelist file for the configuration, test
case, or step, then override them with namelist options in a dictionary for the
test case or step, as in this example:

.. code-block:: python

    add_namelist_file(step, 'compass.ocean.tests.baroclinic_channel',
                      'namelist.forward')
    add_namelist_file(step, 'compass.ocean.tests.baroclinic_channel',
                      'namelist.{}.forward'.format(step['resolution']))
    if 'nu' in step:
        # update the viscosity to the requested value
        options = {'config_mom_del2': '{}'.format(step['nu'])}
        add_namelist_options(step, options)

Here, we get default options for "forward" steps, then for the resolution of
the test case from namelist files, then update the viscosity ``nu``, which is
an option passed in when creating this step.

.. note::

  Namelist values must be of type ``str``, so use ``'{}'.format(value)`` to
  convert a numerical value to a string.

Generating a namelist file
^^^^^^^^^^^^^^^^^^^^^^^^^^

Calls to :py:func:`compass.namelist.add_namelist_file()` and
:py:func:`compass.namelist.add_namelist_options()` queue up replacements but
they are only parsed when you call :py:func:`compass.namelist.generate_namelist()`.
If your namelist has the default name (``namelist.<core>``) and the model will
be run in ``forward`` mode, you just need to provide the ``step`` dictionary
and config options.  You can give the file a different name or select ``init``
mode if you need to.

The namelist is typically generated in :ref:`dev_step_setup`.  It cannot be
generated during ``collect()`` because the work directory is not known and
anyway we do not want to perform any file creation at all during ``collect()``.
It could also be generated during ``run()``, but we do not recommend this
because it would not give the user a chance to modify namelist options
themselves before running.

Updating a namelist file
^^^^^^^^^^^^^^^^^^^^^^^^

It is sometimes useful to update namelist options after a namelist has already
been generated with :py:func:`compass.namelist.generate_namelist()`.  This
typically happens during ``run()`` for options that cannot be known beforehand,
particularly options related to the number of cores and threads.  In such
cases, call :py:func:`compass.namelist.update()`:

.. code-block:: python

    from compass.namelist import update

    ...

    replacements = {'config_pio_num_iotasks': '{}'.format(pio_num_iotasks),
                    'config_pio_stride': '{}'.format(pio_stride)}

    update(replacements=replacements, step_work_dir=step_dir,
           out_name=namelist)

.. _dev_streams:

Streams
-------

Cores, configurations, and test cases can provide streams files that are used
to define new streams or update default streams before MPAS runs.  Streams
files within the ``compass`` package must start with the prefix ``streams.`` to
ensure that they are included when we build the package.

Streams files are a bit more complicated than :ref:`dev_namelist` files because
streams files are XML documents, requiring some slightly more sophisticated
parsing.

Adding a streams file to a step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Typically, a step that runs MPAS will include one or more calls to
:py:func:`compass.streams.add_streams_file()` within :ref:`dev_step_collect`
or :ref:`dev_step_setup`.  Calling this function simply adds the file to a
list within the ``step`` dictionary that will be parsed if an when
:py:func:`compass.streams.generate_streams()` gets called to create the
streams file.  (This way, it is safe to add streams files to a step in
``collect()`` even if that test case will never get set up or run.)

The format of the streams file is essentially the same as the default and
generated streams file, e.g.:

.. code-block:: xml

    <streams>

    <immutable_stream name="mesh"
                      filename_template="init.nc"/>

    <immutable_stream name="input"
                      filename_template="init.nc"/>

    <immutable_stream name="restart"/>

    <stream name="output"
            type="output"
            filename_template="output.nc"
            output_interval="0000_00:00:01"
            clobber_mode="truncate">

        <var_struct name="tracers"/>
        <var name="xtime"/>
        <var name="normalVelocity"/>
        <var name="layerThickness"/>
    </stream>

    </streams>

These are all streams that are already defined in the default forward streams
for MPAS-Ocean, so the defaults will be updated.  If only the attributes of
a stream are given, the contents of the stream (the ``var``, ``var_struct``
and ``var_array`` tags within the stream) are taken from the defaults.  If
any contents are given, as for the ``output`` stream in the example above, they
replace the default contents.  ``compass`` does not include a way to add or
remove contents from the defaults, just keep the default contents or replace
them all.  (Legacy COMPASS had such an option but it was found to be mostly
confusing and difficult to keep synchronized with the MPAS code.)

A typical streams file is added by passing the ``step`` dictionary, along with
a package where the streams file is located and the name of the input streams
file within that package:

.. code-block:: python

    add_streams_file(step, 'compass.ocean.tests.baroclinic_channel',
                     'streams.forward')

If the streams file should have a different name than the default
(``streams.<core>``), the name can be given via the ``out_name`` keyword
argument.

Adding a template streams file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main difference between namelists and streams files is that there is no
direct equivalent for streams of :py:func:`compass.namelist.add_namelist_options()`.
It is simply too confusing to try to define streams within the code.

Instead, :py:func:`compass.streams.add_streams_file()` includes a keyword
argument ``template_replacements``.  If you provide a dictionary of
replacements to this argument, the input streams file will be treated as a
`Jinja2 template <https://jinja.palletsprojects.com/>`_ that is rendered
using the provided replacements.  Here is an example of such a template streams
file:

.. code-block:: xml

    <streams>

    <stream name="output"
            output_interval="{{ output_interval }}"/>
    <immutable_stream name="restart"
                      filename_template="../restarts/rst.$Y-$M-$D_$h.$m.$s.nc"
                      output_interval="{{ restart_interval }}"/>

    </streams>

And here is how it would be added, along with replacements:

.. code-block:: python

    stream_replacements = {
        'output_interval': '00-00-01_00:00:00',
        'restart_interval': '00-00-01_00:00:00'}
    add_streams_file(step, module, 'streams.template',
                     template_replacements=stream_replacements)

    ...

    stream_replacements = {
        'output_interval': '00-00-01_00:00:00',
        'restart_interval': '00-00-01_00:00:00'}
    add_streams_file(step, module, 'streams.template',
                     template_replacements=stream_replacements)

In this example, taken from
:py:func:`compass.ocean.tests.global_ocean.mesh.qu240.spinup.collect()`, we
are creating a series of steps that will be used to perform dynamic adjustment
of the ocean model, each of which might have different durations and restart
intervals.  Rather than creating a streams file for each step of the spin up,
we reuse the same template with just a few appropriate replacements.  Thus,
calls to :py:func:`compass.streams.add_streams_file()` with
``template_replacements`` are qualitatively similar to namelist calls to
:py:func:`compass.namelist.add_namelist_options()`.


Generating a streams file
^^^^^^^^^^^^^^^^^^^^^^^^^

Calls to :py:func:`compass.streams.add_streams_file()` queue up streams files
or templates but they are only parsed when you call
:py:func:`compass.streams.generate_streams()`. If your output streams file has
the default name (``streams.<core>``) and the model will be run in ``forward``
mode, you just need to provide the ``step`` dictionary and config options.  You
can give the file a different name or select ``init`` mode if you need to.

The streams file is typically generated in :ref:`dev_step_setup`.  It cannot be
generated during ``collect()`` because the work directory is not known and
anyway we do not want to perform any file creation at all during ``collect()``.
It could also be generated during ``run()``, but we do not recommend this
because it would not give the user a chance to modify streams file themselves
before running.

.. _dev_validation:

Validation
----------

Test cases should typically include validation of variables and/or timers.
This validation is a critical part of running test suites and comparing them
to baselines.

Validating variables
^^^^^^^^^^^^^^^^^^^^

The function :py:func:`compass.validate.compare_variables()` can be used to
compare variables in a file with a given relative path (``filename1``) with
a the same variables in another file (``filename2``) and/or against a baseline.

As a simple example:

.. code-block:: python

    variables = ['temperature', 'salinity', 'layerThickness', 'normalVelocity']
    compare_variables(variables, config, work_dir=testcase['work_dir'],
                      filename1='forward/output.nc')

In this case, comparison will only take place if a baseline run is provided
when the test case is set up (see :ref:`dev_compass_setup` or
:ref:`dev_compass_suite`), since the keyword argument ``filename2`` was not
provided.  If a baseline is provided, the 4 prognostic variables are compared
between the file ``forward/output.nc`` and the same file in the corresponding
location within the baseline.

Here is a slightly more complex example:

.. code-block:: python

    variables = ['temperature', 'salinity', 'layerThickness', 'normalVelocity']
    steps = testcase['steps_to_run']
    if '4proc' in steps and '8proc' in steps:
        compare_variables(variables, config, work_dir=testcase['work_dir'],
                          filename1='4proc/output.nc',
                          filename2='8proc/output.nc')

In this case, we only perform the comparison if both ``4proc`` and ``8proc``
steps have been run (otherwise, we cannot be sure the data we want will be
available).  If so, we compare the 4 prognostic variable in ``4proc/output.nc``
with the same in ``8proc/output.nc`` to make sure they are identical.  If
a baseline directory was provided, these 4 variables in each file will also be
compared with those in the corresponding files in the baseline.

In any of these cases, if comparison fails, a ``ValueError`` is raised and
execution of the test case is terminated.

Typical output will look like this:

.. code-block:: none

    Beginning variable comparisons for all time levels of field 'temperature'. Note any time levels reported are 0-based.
        Pass thresholds are:
           L1: 0.00000000000000e+00
           L2: 0.00000000000000e+00
           L_Infinity: 0.00000000000000e+00
    0:  l1: 0.00000000000000e+00  l2: 0.00000000000000e+00  linf: 0.00000000000000e+00
    1:  l1: 0.00000000000000e+00  l2: 0.00000000000000e+00  linf: 0.00000000000000e+00
    2:  l1: 0.00000000000000e+00  l2: 0.00000000000000e+00  linf: 0.00000000000000e+00
     ** PASS Comparison of temperature between /home/xylar/data/mpas/test_nightly_latest/ocean/baroclinic_channel/10km/threads_test/1thread/output.nc and
        /home/xylar/data/mpas/test_nightly_latest/ocean/baroclinic_channel/10km/threads_test/2thread/output.nc
    Beginning variable comparisons for all time levels of field 'salinity'. Note any time levels reported are 0-based.
        Pass thresholds are:
           L1: 0.00000000000000e+00
           L2: 0.00000000000000e+00
           L_Infinity: 0.00000000000000e+00
    0:  l1: 0.00000000000000e+00  l2: 0.00000000000000e+00  linf: 0.00000000000000e+00
    1:  l1: 0.00000000000000e+00  l2: 0.00000000000000e+00  linf: 0.00000000000000e+00
    2:  l1: 0.00000000000000e+00  l2: 0.00000000000000e+00  linf: 0.00000000000000e+00
     ** PASS Comparison of salinity between /home/xylar/data/mpas/test_nightly_latest/ocean/baroclinic_channel/10km/threads_test/1thread/output.nc and
        /home/xylar/data/mpas/test_nightly_latest/ocean/baroclinic_channel/10km/threads_test/2thread/output.nc
    Beginning variable comparisons for all time levels of field 'layerThickness'. Note any time levels reported are 0-based.
        Pass thresholds are:
           L1: 0.00000000000000e+00
           L2: 0.00000000000000e+00
           L_Infinity: 0.00000000000000e+00
    0:  l1: 0.00000000000000e+00  l2: 0.00000000000000e+00  linf: 0.00000000000000e+00
    1:  l1: 0.00000000000000e+00  l2: 0.00000000000000e+00  linf: 0.00000000000000e+00
    2:  l1: 0.00000000000000e+00  l2: 0.00000000000000e+00  linf: 0.00000000000000e+00
     ** PASS Comparison of layerThickness between /home/xylar/data/mpas/test_nightly_latest/ocean/baroclinic_channel/10km/threads_test/1thread/output.nc and
        /home/xylar/data/mpas/test_nightly_latest/ocean/baroclinic_channel/10km/threads_test/2thread/output.nc
    Beginning variable comparisons for all time levels of field 'normalVelocity'. Note any time levels reported are 0-based.
        Pass thresholds are:
           L1: 0.00000000000000e+00
           L2: 0.00000000000000e+00
           L_Infinity: 0.00000000000000e+00
    0:  l1: 0.00000000000000e+00  l2: 0.00000000000000e+00  linf: 0.00000000000000e+00
    1:  l1: 0.00000000000000e+00  l2: 0.00000000000000e+00  linf: 0.00000000000000e+00
    2:  l1: 0.00000000000000e+00  l2: 0.00000000000000e+00  linf: 0.00000000000000e+00
     ** PASS Comparison of normalVelocity between /home/xylar/data/mpas/test_nightly_latest/ocean/baroclinic_channel/10km/threads_test/1thread/output.nc and
        /home/xylar/data/mpas/test_nightly_latest/ocean/baroclinic_channel/10km/threads_test/2thread/output.nc


Norms
^^^^^

In the unlikely circumstance that you would like to allow comparison to pass
with non-zero differences between variables, you can supply keyword arguments
``l1_norm``, ``l2_norm`` and/or ``linf_norm`` to give the desired maximum
values for these norms, above which the comparison will fail, raising a
``ValueError``.  These norms only affect the comparison between ``filename1``
and ``filename2``, not with the baseline (which always uses 0.0 for these
norms).

Validating timers
^^^^^^^^^^^^^^^^^

Timer validation is qualitatively similar to variable validation except that
no error are raised, meaning that the user must manually look at the
comparison and make a judgment call about whether any changes in timing are
large enough to indicate performance problems.

Calls to :py:func:`compass.validate.compare_timers()` include a list of MPAS
timers to compare and at least 1 directory where MPAS has been run and timers
for the run are available.

Here is a typical call:

.. code-block:: python

    timers = ['time integration']
    compare_timers(timers, config, work_dir, rundir1='forward')

Typical output will look like:

.. code-block:: none

    Comparing timer time integration:
                 Base: 0.92264
              Compare: 0.82317
       Percent Change: -10.781019682649793%
              Speedup: 1.1208377370409515

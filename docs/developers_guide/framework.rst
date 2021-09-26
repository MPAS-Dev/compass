.. _dev_framework:

Framework
=========

All of the :ref:`dev_packages` that are not in the two cores (``landice`` and
``ocean``) belong to the ``compass`` framework.  Some of these
modules and packages are used by the :ref:`dev_command_line`, while others are
meant to be called within test cases and steps to simplify tasks like adding
input and output files, downloading data sets, building up config files,
namelists and streams files, setting up and running the MPAS model, and
verifying the output by comparing steps with one another or against a baseline.

.. _dev_list:

list module
-----------

The :py:func:`compass.list.list_cases()`, :py:func:`compass.list.list_machines()`
and :py:func:`compass.list.list_suites()` functions are used by the
``compass list`` command to list test cases, supported machines and test
suites, respectively.  These functions are not currently used anywhere else
in ``compass``.

.. _dev_setup:

setup module
~~~~~~~~~~~~

The :py:func:`compass.setup.setup_cases()` and :py:func:`compass.setup.setup_case()`
functions are used by ``compass setup`` and ``compass suite`` to set up a list
of test cases and a single test case, respectively, in a work directory.
Subdirectories will be created for each test case and its steps; input,
namelist and streams files will be downloaded, symlinked and/or generated
in the setup process. A `pickle file <https://docs.python.org/3/library/pickle.html>`_
called ``test_case.pickle`` will be written to each test case directory
containing the test-case object for later use in calls to ``compass run``.
Similarly, a file ``step.pickle`` containing both the step and test-case
objects will be written to each step directory, allowing the step to be run
on its own with ``compass run``.  In contrast to :ref:`config_files`, these
pickle files are not intended for users (or developers) to read or modify.
Properties of the test-case and step objects are not intended to change between
setting up and running a test suite, test case or step.

.. _dev_clean:

clean module
~~~~~~~~~~~~

The :py:func:`compass.clean.clean_cases()` function is used by
``compass clean`` and ``compass suite`` to delete the constants of a test-case
subdirectory in the work directory.

.. _dev_suite:

suite module
~~~~~~~~~~~~

The :py:func:`compass.suite.setup_suite()` and :py:func:`compass.suite.clean_suite()`
functions are used by ``compass suite`` to set up or clean up a test suite in a
work directory.  Setting up a test suite includes setting up the test cases
(see :ref:`dev_setup`), writing out a :ref:`dev_provenance` file, and saving
a pickle file containing a python dictionary that defines the test suite for
later use by ``compass run``.  The "target" and "minimum" number of cores
required for running the test suite are displayed.  The "target" is the maximum
of the ``cores`` attribute of all steps in the test suite.  This is the number
of cores to run on to complete the test suite as quickly as possible, with the
caveat that many cores may sit idle for some fraction of the runtime.  The
"minimum" number of cores is the maximum of the ``min_cores`` attribute for
all steps int he suite, indicating the fewest cores that the test may be run
with before at least some steps in the suite will fail.

.. _dev_run:

run module
~~~~~~~~~~

The :py:func:`compass.run.run_suite()`, :py:func:`compass.run.run_test_case()`,
and :py:func:`compass.run.run_step()` functions are used to run a test suite,
test case or step, respectively, from the base, test case or step work
directory, respectively, using ``compass run``.  Each of these functions reads
a local pickle file to retrieve information about the test suite, test case
and/or step that was stored during setup.

:py:func:`compass.run.run_suite()` runs each test case in the test suite in
the order that they are given in the text file defining the suite
(``compass/<mpas_core>/suites/<suite_name>.txt``).  It displays a ``PASS`` or
``FAIL`` message for the test execution, as well as similar messages for
validation involving output within the test case or suite and validation
against a baseline (depending on the implementation of the ``validate()``
method in the test case and whether a baseline was provided during setup).
Output from test cases and their steps are stored in log files in
the ``case_output`` subdirectory of the base work directory.

:py:func:`compass.run.run_test_case()` and :py:func:`compass.run.run_step()`
run a single test case.  In the latter case, only the selected step from the
test case is run, skipping any others.  If running the full test case, output
from individual steps are stored in log files ``<step>.log`` in the test case's
work directory.  The results of validation (if any) are displayed in the final
stage of running the test case.

.. _dev_cache:

cache module
~~~~~~~~~~~~

The :py:func:`compass.cache.update_cache()` function is used by
``compass cache`` to copy step outputs to the ``compass_cache`` database on
the LCRC server and to update ``<mpas_core>_cached_files.json`` files that
contain a mapping between these cached files and the original outputs.  This
functionality enables running steps with :ref:`dev_step_cached_output`, which
can be used to skip time-consuming initialization steps for faster development
and debugging.

.. _dev_config:

Config files
------------

You can add a config file within an MPAS core with the same name as the MPAS
core plus a ``.cfg`` extension (e.g. ``ocean.cfg``).  This file will
automatically get loaded during setup.  Similarly, you can add config files
within test groups or test cases with the same naming convention (e.g.
``global_ocean.cfg`` for the ``global_ocean`` test group or ``cosine_bell.cfg``
for the ``cosine_bell`` test case).  Each of these config files will be loaded
automatically during setup without you needing to add them explicitly.

If you need to add additional config files that serve other purposes, you can
use the test case's In a test case's :py:meth:`compass.TestCase.add_config()`
method within the test case's constructor (``__init__()`` method).  For
example, a slightly simplified version of the
:py:class:`compass.ocean.tests.global_ocean.mesh.Mesh` test case might load
a config file specific to the global ocean mesh as follows:

.. code-block:: python

    def __init__(self, test_group, mesh_name):
        ...
        self.add_config(package=mesh_step.package,
                        filename=mesh_step.mesh_config_filename,
                        exception=True)
        ...

The ``package`` and ``filename`` arguments are the name of a package containing
the config file and the name of the config file itself, respectively.  In this
case, we know that the config file should always exist, so we would like the
code to raise an exception (``exception=True``) if the file is not found.  This
is the default behavior.  In some cases, you would like the code to add the
config options if the config file exists and do nothing if it does not, in
which case you should pass ``exception=False``.

There is also a framework-level module, ``compass.config`` that includes
functions for creating and manipulating config options and :ref:`config_files`.

The :py:func:`compass.config.add_config()` function can be used to add the
contents of a config file within a package to the current config parser.
An example of this is the
:py:class:`compass.ocean.tests.global_convergence.cosine_bell.CosineBell` test
case:

.. code-block:: python

    def __init__(self, test_group):
        ...
        config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())
        add_config(config, self.__module__, '{}.cfg'.format(self.name))
        self._setup_steps(config)

In this particular case, the config file is loaded to get a list of default
resolutions for steps in the test case before the steps are created.

The ``config`` module also contains 3 functions that are intended for internal
use by the framework itself. Test-case developers will typically not need to
call these functions directly.

The :py:func:`compass.config.duplicate_config()` function can be used to make a
deep copy of a ``config`` object so changes can be made without affecting the
original.

The :py:func:`compass.config.ensure_absolute_paths()` function is used
internally by the framework to check and update config options in the
``paths``, ``namelists``, ``streams``, and ``executables`` sections of the
config file to make sure they have absolute paths. The absolute paths are
determined from the location where one of the tools from the compass
:ref:`dev_command_line` was called.

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
test-case developers won't have to create their own ``logger`` objects.  They
are arguments to the test case's :ref:`dev_test_case_run` or step's
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
:py:meth:`compass.landice.tests.dome.setup_mesh.SetupMesh.run()`:

.. code-block:: python

    from mpas_tools.logging import check_call


    def run(self):
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

A lot of I/O related tasks are handled internally in the step class
:py:class:`compass.Step`.  Some of the lower level functions can be called
directly if need be.

.. _dev_io_symlink:

Symlinks
~~~~~~~~

You can create your own symlinks that aren't input files (e.g. for a
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
~~~~~~~~

You can download files more directly if you need to using
:py:func:`compass.io.download()`, though we recommend using
:py:meth:`compass.Step.add_input_file()` whenever possible because it is more
flexible and takes care of more of the details of symlinking the local file
and adding it as an input to the step.  No current test cases use
``download()`` directly, but an example might look like this:

.. code-block:: python

    from compass.io import symlink, download

    def setup(self):

        step_dir = self.work_dir
        database_root = self.config.get('paths', 'ocean_database_root')
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
~~~~~~~~~~~~

Steps that run the MPAS model should call the
:py:meth:`compass.Step.add_model_as_input()` method from
their ``__init__()`` method.

To run MPAS, call :py:func:`compass.model.run_model()`.  By default, this
function first updates the namelist options associated with the
`PIO library <https://ncar.github.io/ParallelIO/>`_ and partitions the mesh
across MPI tasks, as we will discuss in a moment, before running the model.
You can provide non-default names for the graph, namelist and streams files.
The number of cores and threads is determined from the ``cores``, ``min_cores``
and ``threads`` attributes of the step object, set in its
constructor or :ref:`dev_step_setup` method (i.e. before calling
:ref:`dev_step_run`) so that the ``compass`` framework can ensure that the
required resources are available.

Partitioning the mesh
~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
``update_pio=False`` when calling ``run_model()``, then call
:py:func:`compass.model.update_namelist_pio()` yourself.

If you wish to use the MPAS default behavior of 1 PIO task per core, or wish to
set ``config_pio_num_iotasks`` and ``config_pio_stride`` yourself, simply
use ``update_pio=False`` when calling ``run_model()``.


Making a graph file
~~~~~~~~~~~~~~~~~~~

Some ``compass`` test cases take advantage of the fact that the
`MPAS-Tools cell culler <http://mpas-dev.github.io/MPAS-Tools/stable/mesh_conversion.html#cell-culler>`_
can produce a graph file as part of the process of culling cells from an
MPAS mesh.  In test cases that do not require cells to be culled, you can
call :py:func:`compass.model.make_graph_file()` to produce a graph file from
an MPAS mesh file.  Optionally, you can provide the name of an MPAS field on
cells in the mesh file that gives different weight to different cells
(``weight_field``) in the partitioning process.

.. _dev_validation:

Validation
----------

Test cases should typically include validation of variables and/or timers.
This validation is a critical part of running test suites and comparing them
to baselines.

Validating variables
~~~~~~~~~~~~~~~~~~~~

The function :py:func:`compass.validate.compare_variables()` can be used to
compare variables in a file with a given relative path (``filename1``) with
the same variables in another file (``filename2``) and/or against a baseline.

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
    compare_variables(variables, config, work_dir=testcase['work_dir'],
                      filename1='4proc/output.nc',
                      filename2='8proc/output.nc')

In this case, we compare the 4 prognostic variables in ``4proc/output.nc``
with the same in ``8proc/output.nc`` to make sure they are identical.  If
a baseline directory was provided, these 4 variables in each file will also be
compared with those in the corresponding files in the baseline.

By default, the comparison will only be performed if both the ``4proc`` and
``8proc`` steps have been run (otherwise, we cannot be sure the data we want
will be available).  If one of the steps was not run (if the user is running
steps one at a time or has altered the ``steps_to_run`` config option to remove
some steps), the function will skip validation, logging a message that
validation was not performed because of the missing step(s).  You can pass
the keyword argument ``skip_if_step_not_run=False`` to force validation to run
(and possibly to fail because the output is not available) even if the user did
not run the step involved in the validation.

In any of these cases, if comparison fails, the failure is stored in the
``validation`` attribute of the test case, and a ``ValueError`` will be raised
later by the framework, terminating execution of the test case.

If ``quiet=False``, typical output will look like this:

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

If ``quiet=True`` (the default), there is only an indication that the
comparison passed for each variable:

.. code-block:: none

    temperature          Time index: 0, 1, 2
      PASS /home/xylar/data/mpas/test_20210616/further_validation/ocean/baroclinic_channel/10km/threads_test/1thread/output.nc

           /home/xylar/data/mpas/test_20210616/further_validation/ocean/baroclinic_channel/10km/threads_test/2thread/output.nc

    salinity             Time index: 0, 1, 2
      PASS /home/xylar/data/mpas/test_20210616/further_validation/ocean/baroclinic_channel/10km/threads_test/1thread/output.nc

           /home/xylar/data/mpas/test_20210616/further_validation/ocean/baroclinic_channel/10km/threads_test/2thread/output.nc

    layerThickness       Time index: 0, 1, 2
      PASS /home/xylar/data/mpas/test_20210616/further_validation/ocean/baroclinic_channel/10km/threads_test/1thread/output.nc

           /home/xylar/data/mpas/test_20210616/further_validation/ocean/baroclinic_channel/10km/threads_test/2thread/output.nc

    normalVelocity       Time index: 0, 1, 2
      PASS /home/xylar/data/mpas/test_20210616/further_validation/ocean/baroclinic_channel/10km/threads_test/1thread/output.nc

           /home/xylar/data/mpas/test_20210616/further_validation/ocean/baroclinic_channel/10km/threads_test/2thread/output.nc

    temperature          Time index: 0, 1, 2
      PASS /home/xylar/data/mpas/test_20210616/further_validation/ocean/baroclinic_channel/10km/threads_test/1thread/output.nc

           /home/xylar/data/mpas/test_20210616/baseline/ocean/baroclinic_channel/10km/threads_test/1thread/output.nc

    salinity             Time index: 0, 1, 2
      PASS /home/xylar/data/mpas/test_20210616/further_validation/ocean/baroclinic_channel/10km/threads_test/1thread/output.nc

           /home/xylar/data/mpas/test_20210616/baseline/ocean/baroclinic_channel/10km/threads_test/1thread/output.nc

    layerThickness       Time index: 0, 1, 2
      PASS /home/xylar/data/mpas/test_20210616/further_validation/ocean/baroclinic_channel/10km/threads_test/1thread/output.nc

           /home/xylar/data/mpas/test_20210616/baseline/ocean/baroclinic_channel/10km/threads_test/1thread/output.nc

    normalVelocity       Time index: 0, 1, 2
      PASS /home/xylar/data/mpas/test_20210616/further_validation/ocean/baroclinic_channel/10km/threads_test/1thread/output.nc

           /home/xylar/data/mpas/test_20210616/baseline/ocean/baroclinic_channel/10km/threads_test/1thread/output.nc

    temperature          Time index: 0, 1, 2
      PASS /home/xylar/data/mpas/test_20210616/further_validation/ocean/baroclinic_channel/10km/threads_test/2thread/output.nc

           /home/xylar/data/mpas/test_20210616/baseline/ocean/baroclinic_channel/10km/threads_test/2thread/output.nc

    salinity             Time index: 0, 1, 2
      PASS /home/xylar/data/mpas/test_20210616/further_validation/ocean/baroclinic_channel/10km/threads_test/2thread/output.nc

           /home/xylar/data/mpas/test_20210616/baseline/ocean/baroclinic_channel/10km/threads_test/2thread/output.nc

    layerThickness       Time index: 0, 1, 2
      PASS /home/xylar/data/mpas/test_20210616/further_validation/ocean/baroclinic_channel/10km/threads_test/2thread/output.nc

           /home/xylar/data/mpas/test_20210616/baseline/ocean/baroclinic_channel/10km/threads_test/2thread/output.nc

    normalVelocity       Time index: 0, 1, 2
      PASS /home/xylar/data/mpas/test_20210616/further_validation/ocean/baroclinic_channel/10km/threads_test/2thread/output.nc

           /home/xylar/data/mpas/test_20210616/baseline/ocean/baroclinic_channel/10km/threads_test/2thread/output.nc

By default, the function checks to make sure ``filename1`` and, if provided,
``filename2`` are output from one of the steps in the test case.  In general,
validation should be performed on outputs of the steps in this test case that
are explicitly added with :py:meth:`compass.Step.add_output_file()`.  This
check can be disabled by setting ``check_outputs=False``.

Norms
~~~~~

In the unlikely circumstance that you would like to allow comparison to pass
with non-zero differences between variables, you can supply keyword arguments
``l1_norm``, ``l2_norm`` and/or ``linf_norm`` to give the desired maximum
values for these norms, above which the comparison will fail, raising a
``ValueError``.  These norms only affect the comparison between ``filename1``
and ``filename2``, not with the baseline (which always uses 0.0 for these
norms).  If you do want certain norms checked, you can pass their value as
``None``.

If you want different nonzero norm values for different variables,
the easiest solution is to call :py:func:`compass.validate.compare_variables()`
separately for each variable and  with different norm values specified.
:py:func:`compass.validate.compare_variables()` can safely be called multiple
times without clobbering a previous result.  When you specify a nonzero norm,
you may want compass to print the norm values it is using for comparison
when the results are printed.  To do so, use the optional ``quiet=False``
argument.


Validating timers
~~~~~~~~~~~~~~~~~

Timer validation is qualitatively similar to variable validation except that
no errors are raised, meaning that the user must manually look at the
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


.. _dev_provenance:

Provenance
----------

The ``compass.provenance`` module defines a function
:py:func:`compass.provenance.write()` for creating a file in the base work
directory with provenance, such as the git version, conda packages, compass
commands, and test cases.

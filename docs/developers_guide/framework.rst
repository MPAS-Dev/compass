.. _dev_framework:

Framework
=========

All of the :ref:`dev_packages` that are not in the two cores (``landice`` and
``ocean``) belong to the ``compass`` framework.  Some of these
modules and packages are used by the :ref:`dev_command_line`, while others are
meant to be called within test cases and steps to simplify tasks like adding
input and output files, downloading data sets, building up config files,
namelists and streams files, set up and run the MPAS model, and verify the
output by comparing steps with one another or against a baseline.

.. _dev_config:

Config files
------------

The ``compass.config`` module includes functions for creating and manipulating
config options and :ref:`config_files`.


The :py:func:`compass.config.add_config()` function can be used to add the
contents of a config file within a package to the current config parser.
Examples of this can be found in most test cases as well as
:py:func:`compass.setup.setup_case()`. Here is a typical example from
:py:func:`compass.landice.tests.enthalpy_benchmark.A.A.configure()`:

.. code-block:: python

    def configure(self):
        """
        Modify the configuration options for this test case
        """
        add_config(self.config, 'compass.landice.tests.enthalpy_benchmark.A',
                   'A.cfg', exception=True)
        ...

The second and third arguments are the name of a package containing the config
file and the name of the config file itself, respectively.  You can see that
the file is in the path ``compass/landice/tests/enthalpy_benchmark/A``
(replacing the ``.`` in the module name with ``/``).  In this case, we know
that the config file should always exist, so we would like the code to raise
and exception (``exception=True``) if the file is not found.  This is the
default behavior.  In some cases, you would like the code to add the config
options if the config file exists and do nothing if it does not.  This can
be useful if a common configure function is being used for all test
cases in a configuration, as in this example from
:py:func:`compass.ocean.tests.global_ocean.configure.configure_global_ocean()`:

.. code-block:: python

    add_config(config, test_case.__module__, '{}.cfg'.format(test_case.name),
               exception=False)

When this is called within the ``mesh`` test case, nothing will happend because
``compass/ocean/tests/global_ocean/mesh`` does not contain a ``mesh.cfg`` file.
The config files for meshes are handled differently, since they aren't
associated with a particular test case:

.. code-block:: python

    mesh_step = mesh.mesh_step
    add_config(config, mesh_step.package, mesh_step.mesh_config_filename,
               exception=True)

In this case, the mesh step keeps track of the package and config file in its
attributes (e.g. ``compass.ocean.tests.global_ocean.mesh.qu240`` and
``qu240.cfg`` for the ``QU240`` and ``QUwISC240`` meshes).  Since we require
each mesh to have config options (to define the vertical grid and the metadata
to be added to the mesh, at the very least), we use ``exception=True`` so an
exception will be raised if no config file is found.

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
^^^^^^^^

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
^^^^^^^^

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
^^^^^^^^^^^^

Steps that run the MPAS model should call the
:py:meth:`compass.Step.add_model_as_input()` method their ``__init__()``
method.

To run MPAS, call :py:func:`compass.model.run_model()`.  By default, this
function first updates the namelist options associated with the
`PIO library <https://ncar.github.io/ParallelIO/>`_ and partition the mesh
across MPI tasks, as we sill discuss in a moment, before running the model.
You can provide non-default names for the graph, namelist and streams files.
The number of cores and threads is determined from the ``cores``, ``min_cores``
and ``threads`` attributes of the step object, set in its
constructor or :ref:`dev_step_setup` method (i.e. before calling
:ref:`dev_step_run`) so that the ``compass`` framework can ensure that the
required resources are available.

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
comparison passed for each variable.

By default, the function checks to make sure ``filename1`` and, if provided,
``filename2`` are output of one of the steps in the test case.  In general,
validation should be performed on outputs of the steps this test case that are
explicitly added with :py:meth:`compass.Step.add_output_file()`.  This check
can be disabled by setting ``check_outputs=False``.

Norms
^^^^^

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

.. _dev_framework:

Framework
=========

All of the :ref:`dev_packages` that are not in the two MPAS cores (``landice``
and ``ocean``) belong to the ``compass`` framework.  Some of these
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
required for running the test suite are displayed.  The "target" is determined
based on the maximum product of the ``ntasks`` and ``cpus_per_task``
attributes of each step in the test suite.  This is the number of cores to run
on to complete the test suite as quickly as possible, with the
caveat that many cores may sit idle for some fraction of the runtime.  The
"minimum" number of cores is the maximum of the product of the ``min_tasks``
and ``min_cpus_per_task``` attribute for all steps in the suite, indicating the
fewest cores that the test may be run with before at least some steps in the
suite will fail.

.. _dev_run:

run.serial module
~~~~~~~~~~~~~~~~~

The function :py:func:`compass.run.serial.run_tests()` is used to run a
test suite or test case and :py:func:`compass.run.serial.run_single_step()` is
used to run a single step using ``compass run``.  ``run_tests()`` performs
setup operations like creating a log file and figuring out the number of tasks
and CPUs per task for each step, then it calls each step's ``run()`` method.

Suites run from the base work directory with a pickle file starting with the
suite name, or ``custom.pickle`` if a suite name was not given. Test cases or
steps run from their respective subdirectories with a ``testcase.pickle`` or
``step.pickle`` file in them. Both of these functions reads the local pickle
file to retrieve information about the test suite, test case and/or step that
was stored during setup.

If :py:func:`compass.run.serial.run_tests()` is used for a test suite, it will
run each test case in the test suite in the order that they are given in the
text file defining the suite (``compass/<mpas_core>/suites/<suite_name>.txt``).
Output from test cases and their steps are stored in log files in the
``case_output`` subdirectory of the base work directory. If the function is
used for a single test case, it will run the steps of that test case, writing
output for each step to a log file starting with the step's name. In either
case (suite or individual test), it displays a ``PASS`` or ``FAIL`` message for
the test execution, as well as similar messages for validation involving output
within the test case or suite and validation against a baseline (depending on
the implementation of the ``validate()`` method in the test case and whether a
baseline was provided during setup).

:py:func:`compass.run.run_single_step()` runs only the selected step from a
given test case, skipping any others, displaying the output in the terminal
window rather than a log file.

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

The primary documentation for the config parser is in
`MPAS-Tools config parser <http://mpas-dev.github.io/MPAS-Tools/stable/config.html>`_.
Here, we include some specific details relevant to using the
:py:class:`mpas_tools.config.MpasConfigParser` in compass.

Here, we provide the :py:class:`compass.config.CompassConfigParser` that has
almost the same functionality but also ensures that certain relative paths are
converted automatically to absolute paths.

The :py:meth:`mpas_tools.config.MpasConfigParser.add_from_package()` method can
be used to add the contents of a config file within a package to the config
options. Examples of this can be found in many test cases as well as
:py:func:`compass.setup.setup_case()`. Here is a typical example from
:py:func:`compass.ocean.tests.global_ocean.make_diagnostics_files.MakeDiagnosticsFiles.configure()`:

.. code-block:: python

    def configure(self):
        """
        Modify the configuration options for this test case
        """
        self.config.add_from_package(
           'compass.ocean.tests.global_ocean.make_diagnostics_files',
           'make_diagnostics_files.cfg', exception=True)

The first and second arguments are the name of a package containing the config
file and the name of the config file itself, respectively.  You can see that
the file is in the path ``compass/ocean/tests/global_ocean/make_diagnostics_files``
(replacing the ``.`` in the module name with ``/``).  In this case, we know
that the config file should always exist, so we would like the code to raise
an exception (``exception=True``) if the file is not found.  This is the
default behavior.  In some cases, you would like the code to add the config
options if the config file exists and do nothing if it does not.  This can
be useful if a common configure function is being used for all test
cases in a configuration, as in this example from
:py:func:`setup.setup_case()`:

.. code-block:: python

    # add the config options for the test group (if defined)
    test_group = test_case.test_group.name
    config.add_from_package(f'compass.{mpas_core}.tests.{test_group}',
                            f'{test_group}.cfg', exception=False)

If a test group doesn't have any config options, nothing will happen.

The ``MpasConfigParser`` class also includes methods for adding a user
config file and other config files by file name, but these are largely intended
for use by the framework rather than individual test cases.

Other methods for the ``MpasConfigParser`` are similar to those for
:py:class:`configparser.ConfigParser`.  In addition to ``get()``,
``getinteger()``, ``getfloat()`` and ``getboolean()`` methods, this class
implements :py:meth:`mpas_tools.config.MpasConfigParser.getlist()`, which
can be used to parse a config value separated by spaces and/or commas into
a list of strings, floats, integers, booleans, etc. Another useful method
is :py:meth:`mpas_tools.config.MpasConfigParser.getexpression()`, which can
be used to get python dictionaries, lists and tuples as well as a small set
of functions (``range()``, :py:meth:`numpy.linspace()`,
:py:meth:`numpy.arange()`, and :py:meth:`numpy.array()`)

Comments in config files
~~~~~~~~~~~~~~~~~~~~~~~~

One of the main advantages of :py:class:`mpas_tools.config.MpasConfigParser`
over :py:class:`configparser.ConfigParser` is that it keeps track of comments
that are associated with config sections and options.

See `comments in config files <http://mpas-dev.github.io/MPAS-Tools/stable/config.html#config_comments>`_
in MPAS-Tools for more details.


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
are attributes that belong to the step or test case.  If you run a step on its
own, no log file is created and logging happens to ``stdout``/``stderr``.  If
you run a full test case, each step gets logged to its own log file within the
test case's work directory.  If you run a test suite, each test case and its
steps get logged to a file in the ``case_output`` directory of the suite's work
directory.

Although the logger will capture ``print`` statements, anywhere with a
``run()`` function or the functions called inside that function, it is a good
idea to call ``logger.info`` instead of ``print`` to be explicit about the
expectation that the output may go to a log file.

Even more important, subprocesses that produce output should always be called
with :py:func:`mpas_tools.logging.check_call`, passing in the ``logger`` that
belongs to the step.  Otherwise, output will go to ``stdout``/``stderr`` even
when the intention is to write all output to a log file.  Whereas logging can
capture ``stdout``/``stderr`` to make sure that the ``print`` statements
actually go to log files when desired, there is no similar trick for
automatically capturing the output from direct calls to ``subprocess``
functions.  Here is a code snippet from
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
        database_root = self.config.get('paths', 'database_root')
        download_path = os.path.join(database_root, 'mpas-ocean',
                                     'bathymetry_database')

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

.. _dev_mesh:

Mesh
----

.. _dev_spherical_meshes:

Spherical Meshes
~~~~~~~~~~~~~~~~

Test cases that use global, spherical meshes can add either
:py:class:`compass.mesh.QuasiUniformSphericalMeshStep` or
:py:class:`compass.mesh.IcosahedralMeshStep` in order to creating a base mesh,
using `JIGSAW <https://github.com/dengwirda/jigsaw>`_.  Alternatively, they can
use :py:class:`compass.mesh.QuasiUniformSphericalMeshStep` as the base class
for creating a more complex mesh by overriding the
:py:meth:`compass.mesh.QuasiUniformSphericalMeshStep.build_cell_width_lat_lon()`
method.

A developer can also customize the options data structure passed on to JIGSAW
either by modifying the ``opts`` attribute of either of these classes or by
overriding the :py:meth:`compass.mesh.IcosahedralMeshStep.make_jigsaw_mesh()`
or :py:meth:`compass.mesh.QuasiUniformSphericalMeshStep.make_jigsaw_mesh()`
methods.

Icosahedral meshes will be significantly more uniform and smooth in cell size
than quasi-uniform spherical meshes.  On the other hand, icosahedral meshes are
restricted to resolutions that are an integer number of subdivisions of an
icosahedron.  The following table shows the approximate resolution of a mesh
with a given number of subdivisions:

==============  =================
 subdivisions    cell width (km)
==============  =================
5               240
6               120
7               60
8               30
9               15
10              7.5
11              3.8
12              1.9
13              0.94
==============  =================

The following config options are associated with spherical meshes:

.. code-block:: cfg

    # config options related to spherical meshes
    [spherical_mesh]

    # for icosahedral meshes, whether to use cell_width to determine the number of
    # subdivisions or to use subdivisions directly
    icosahedral_method = cell_width

    # output file names
    jigsaw_mesh_filename = mesh.msh
    jigsaw_geom_filename = geom.msh
    jigsaw_jcfg_filename = opts.jig
    jigsaw_hfun_filename = spac.msh
    triangles_filename = mesh_triangles.nc
    mpas_mesh_filename = base_mesh.nc

    # options related to writing out and plotting cell widths
    plot_cell_width = True
    cell_width_filename = cellWidthVsLatLon.nc
    cell_width_image_filename = cellWidthGlobal.png
    cell_width_colormap = '3Wbgy5'

    # whether to add the mesh density to the file
    add_mesh_density = False

    # convert the mesh to vtk format for visualization
    convert_to_vtk = True

    # the subdirectory for the vtk output
    vtk_dir = base_mesh_vtk

    # whether to extract the vtk output in lat/lon space, rather than on the sphere
    vtk_lat_lon = False

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
The number of cores and threads is determined from the ``ntasks``,
``min_tasks``, ``cpus_per_task``, `min_cpus_per_task`` and ``openmp_threads``
attributes of the step object, set in its constructor or :ref:`dev_step_setup` method (i.e. before calling
:ref:`dev_step_run`) so that the ``compass`` framework can ensure that the
required resources are available.

Partitioning the mesh
~~~~~~~~~~~~~~~~~~~~~

The function :py:func:`compass.model.partition()` calls the graph partitioning
executable (`gpmetis <https://arc.vt.edu/userguide/metis/>`_ by default) to
divide up the MPAS mesh across MPI tasks.  If you call
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

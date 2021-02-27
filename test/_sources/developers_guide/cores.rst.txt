.. _dev_cores:

Cores
=====

The test cases in compass are organized by "core", corresponding to a dynamical
core in MPAS, and then into "configurations".  Currently, there are three
cores, ``examples`` which simply houses some very basic examples (as the name
implies); ``landice``, which has test cases for MALI; and ``ocean``, which
encompasses all the test cases for MPAS-Ocean.

From a developer's perspective, a core is a package within ``compass`` that:

1. contains a ``tests`` package, which contains packages for each
   configuration, each of which contains various packages and modules for
   test cases and their steps.

2. collects all the test cases of each configuration  together in the
   ``collect()`` function in ``compass/<core>/tests/__init__.py`` (see below)

3. contains a ``<core>.cfg`` config file containing any default config options
   that are universal to all configurations of the core

The core can also contain other packages and modules besides ``tests`` as part
of its "framework".  The core's framework is a mix of shared code and other
files (config files, namelists, streams files, etc.) that is expected to be
used only by modules and packages within the core, not by other cores or the
main compass :ref:`dev_framework`.

The ``collect()`` function of a core should simply call the ``collect()``
functions for each configuration, extending the list of test cases with those
from the configuration.  This would look something like
:py:func:`compass.examples.test.collect()` from the ``examples`` core:

.. code-block:: python

    from compass.examples.tests import example_compact, example_expanded


    def collect():
        """
        Get a list of testcases in this configuration

        Returns
        -------
        testcases : list
            A dictionary of configurations within this core

        """
        testcases = list()
        # make sure you add your configuration to this list so it is included
        # in the available testcases
        for configuration in [example_compact, example_expanded]:
            testcases.extend(configuration.collect())

        return testcases

The config file for the core should, at the very least, define the
default value for the ``mpas_model`` path in the ``[paths]`` section.
Typically, it will also define the paths to the model executable and the
default namelist and streams files for "forward mode" (and, for some cores,
"init mode").  From the ``examples`` core, the MPAS dynamical core is given
the dummy name ``core`` (which does not actually exist).  This would be
replaced by the name of the dynamical core (``ocean`` or ``landice``)
throughout the config file for those cores.

The config file also contains the name of a subdirectory on the
`LCRC server <https://web.lcrc.anl.gov/public/e3sm/mpas_standalonedata/>`_
for the dynamical core in the ``core_path`` option in the ``downloads``
section:

.. code-block:: cfg

    # This config file has default config options for the "examples" core

    # The paths section points compass to external paths
    [paths]

    # the relative or absolute path to the root of a branch where MPAS core
    # has been built
    mpas_model = MPAS-Model/core/develop

    # The namelists section defines paths to example_compact namelists that will be used
    # to generate specific namelists. By default, these point to the forward and
    # init namelists in the default_inputs directory after a successful build of
    # the core model.  Change these in a custom config file if you need a different
    # example_compact.
    [namelists]
    forward = ${paths:mpas_model}/default_inputs/namelist.core.forward

    # The streams section defines paths to example_compact streams files that will be used
    # to generate specific streams files. By default, these point to the forward and
    # init streams files in the default_inputs directory after a successful build of
    # the core model. Change these in a custom config file if you need a different
    # example_compact.
    [streams]
    forward = ${paths:mpas_model}/default_inputs/streams.core.forward


    # The executables section defines paths to required executables. These
    # executables are provided for use by specific test cases.  Most tools that
    # compass needs should be in the conda environment, so this is only the path
    # to the MPAS core executable by default.
    [executables]
    model = ${paths:mpas_model}/core_model

    # Options related to downloading files
    [download]

    # the path on the server, which is the one for MPAS-Ocean since we use some of
    # its files
    core_path = mpas-ocean

.. _dev_configs:

Configurations
--------------

Configurations are the next level of test-case organization below
:ref:`dev_cores`.  Typically, the test cases within a configuration are
in some way conceptually linked, serving a similar purpose or being variants on
one another. Often, they have a common topography and initial condition,
perhaps with different mesh resolutions, parameters, or both.  It is common for
a configuration to include "framework" modules that are shared between its test
cases and steps (but typically not with other configurations).  Each core will
typically include a mix of "idealized" configurations (e.g.
:ref:`dev_ocean_baroclinic_channel` or :ref:`dev_ocean_ziso`) and "realistic"
domains (e.g. :ref:`dev_ocean_global_ocean`).

Each configuration is a python package within the core's ``tests`` package.
While it is not required, a configuration will typically include a config file
with a set of default config options that are the starting point for all its
test case, named ``<configuration>.cfg``.  As an example, here is the config
file for the ``example_compact`` configuration:

.. code-block:: cfg

    # default namelist options for the "example_compact" configuration
    [example_compact]

    # A parameter that we will use in setting up or running the test case
    parameter1 = 0.

    # Another parameter
    parameter2 = False

Some configuration options will provide defaults for config options that are
shared across the core (as is the case for the ``[vertical_grid]`` config
section in the ocean core).  But most config options for a configuration will
typically go into a section with the same name as the configuration, as in the
example above.

The ``__init__.py`` file for the configuration must define a ``collect()``
function that makes a list of test cases within the configuration.  This list
is made by calling :py:func:`compass.testcase.add_testcase()`, passing in the
module for each test case.  Returning to the ``example_compact`` configuration,
the function :py:func:`compass.examples.tests.example_compact.collect()` looks
like this:

.. code-block:: python

    def collect():
        testcases = list()
        for resolution in ['1km', '2km']:
            for test in [test1, test2]:
                add_testcase(testcases, test, resolution=resolution)

        return testcases

As in this example, it may be useful for a configuration to make several
versions of a test case by passing different parameters.  In the example, we
create versions of both ``test1`` and ``test2`` at both ``1km`` and ``2km``
resolution.  We will explore this further when we talk about
:ref:`dev_testcases` and :ref:`dev_steps` below.

It is also common for a configuration to have a ``configure()`` function that
can be shared across its tests, see :ref:`dev_testcase_configure`.

An example of a shared ``configure()`` function is
:py:func:`compass.ocean.tests.baroclinic_channel.configure()`:

.. code-block:: python


    def configure(testcase, config):
        resolution = testcase['resolution']
        res_params = {'10km': {'nx': 16,
                               'ny': 50,
                               'dc': 10e3},
                      '4km': {'nx': 40,
                              'ny': 126,
                              'dc': 4e3},
                      '1km': {'nx': 160,
                              'ny': 500,
                              'dc': 1e3}}

        if resolution not in res_params:
            raise ValueError('Unsupported resolution {}. Supported values are: '
                             '{}'.format(resolution, list(res_params)))
        res_params = res_params[resolution]
        for param in res_params:
            config.set('baroclinic_channel', param, '{}'.format(res_params[param]))

In the ``baroclinic_channel`` configuration, 3 resolutions are supported:
``1km``, ``4km`` and ``10km``.  Here, we use a dictionary to define parameters
(the size of the mesh) associated with each resolution and then to set config
options with those parameters.  This approach is appropriate if we want a user
to be able to modify these config options before running the test case (in this
case, if they would like to run on a mesh of a different size or resolution).
If these parameters should be held fixed, they should not be added to the
``config`` object but rather to the ``testcase`` or ``step`` dictionary that
the user cannot change, as we will discuss below.

As with cores and the main ``compass`` package, configurations also can have
a shared "framework" of packages, modules, config files, namelists, and streams
files that is shared among test cases and steps.

.. _dev_testcases:

Test cases
----------

In many ways, test cases are compass's fundamental building blocks, since a
user can't set up an individual step of test case (tough they can run the steps
one at a time).

A test case can be a module but is usually a python package so it can
incorporate modules for its steps and/or config files, namelists, and streams
files.  The test case must include ``collect()`` and ``run()`` functions with
the `API <https://en.wikipedia.org/wiki/API>`_ given below. Most test cases
will also have a ``configure()`` function to add to the config options, but
this is not required.

.. _dev_testcase_dict:

testcase dictionary
^^^^^^^^^^^^^^^^^^^

As discussed previously, we have opted to keep track of the data associated
with a test case using :ref:`dev_dicts_not_classes`.

The ``testcase`` dictionary will typically look like this example from the
``ocean/baroclinic_channel/10km/default`` test case at the beginning of
:py:func:`compass.ocean.tests.baroclinic_channel.default.run()`:

.. code-block:: python


    testcase = {'base_work_dir': '/home/xylar/data/mpas/test_new_run_model/nightly/new_api',
                'config': 'default.cfg',
                'configuration': 'baroclinic_channel',
                'configure': 'configure',
                'core': 'ocean',
                'description': 'baroclinic channel 10km default test',
                'module': 'compass.ocean.tests.baroclinic_channel.default',
                'name': 'default',
                'new_step_log_file': True,
                'path': 'ocean/baroclinic_channel/10km/default',
                'resolution': '10km',
                'run': 'run',
                'steps': {'forward': {...},
                          'initial_state': {...}},
                'steps_to_run': ['initial_state', 'forward'],
                'subdir': '10km/default',
                'work_dir': '/home/xylar/data/mpas/test_new_run_model/nightly/new_api/ocean/baroclinic_channel/10km/default'}

``base_work_dir``
    The base directory where the test cases or test suite have been set up.

``config``
    The config file for the test case, typically ``<name>.cfg``, where
    ``<name>`` is the name of the test case.

``configuration``
    Which of the :ref:`dev_configs` this test case belongs to.  This entry is
    added automatically by :py:func:`compass.testcase.add_testcase()` when it
    is called in the configuration's ``collect()`` function and should not be
    modified.

``configure``
    The name of the :ref:`dev_testcase_configure` function for setting config
    options, set by :py:func:`compass.testcase.add_testcase()`.  This
    entry should only be modified if you have an important reason not to name
    the function in your test case's module ``configure``.

``core``
    Which of the :ref:`dev_cores` this test case belongs to.  This entry is
    added automatically by :py:func:`compass.testcase.add_testcase()` when it
    is called in the configuration's ``collect()`` function and should not be
    modified.

``description``
    A short (one line) description of the test case.  Typically, this is
    similar to the ``path`` of the test, just put into words.  This should be
    set by the test case in :ref:`dev_testcase_collect`.

``module``
    The full name of the module or package where the test case is defined.
    This entry is added automatically by
    :py:func:`compass.testcase.add_testcase()` when it is called in the
    configuration's ``collect()`` function and should not been to be modified.

``name``
    The name of the test case.  The default is the last part of ``module`` and
    is set by :py:func:`compass.testcase.add_testcase()`.  It is not typically
    necessary to change the name of the test case, but this could be done by
    passing ``name`` as a keyword argument to ``add_testcase()`` or within the
    test case's :ref:`dev_testcase_collect` function.

``new_step_log_file``
    An entry used by the compass :ref:`dev_framework` to determine if the steps
    of this test case need their own log files or if they should perform
    :ref:`dev_logging` to the same logger as the test case itself.  This entry
    should not be altered.

``path``
    The relative path of the test case within the base work directory, the
    combination of the ``core``, ``configuration`` and ``subdir``.  This entry
    is added automatically by the :ref:`dev_framework` after
    :ref:`dev_testcase_collect` is called and should not be modified.

``run``
    The name of the :ref:`dev_testcase_run` function for running the test case,
    set by :py:func:`compass.testcase.add_testcase()`.  This entry
    should only be modified if you have an important reason not to name the
    function in your test case's module ``run``.

``steps``
    A dictionary of steps in the test case with the names of the steps as keys
    and each :ref:`dev_step_dict` as the corresponding value.  The ``steps``
    dictionary is created by :py:func:`compass.testcase.add_testcase()` and
    passed to the test case's :ref:`dev_testcase_collect`.  New steps are
    added by calling :py:func:`compass.testcase.add_step()`.

``steps_to_run``
    A list of the steps to run.  By default, this is the names of all of the
    steps in ``steps`` in the order they were added.  You can modify these
    in :ref:`dev_testcase_collect` after calling
    :py:func:`compass.testcase.add_testcase()` if some steps should not
    be run by default. If a user asks to run a single step from the test case,
    the :ref:`dev_testcase_run` function for test case is still called but with
    this list set to just the name of the step to run.

``subdir``
    The subdirectory for the test case within the configuration.  The default
    is the the last part of the ``module`` (the same as the default ``name``)
    and is set by :py:func:`compass.testcase.add_testcase()`.  Most commonly,
    this entry would be modified within the test case's
    :ref:`dev_testcase_collect` function by calling
    :py:func:`compass.testcase.set_testcase_subdir()`. You can also modify
    this entry by passing ``subdir=<subdir>`` as a keyword argument to
    :py:func:`compass.testcase.add_testcase()` in the configuration's
    ``collect()`` function.

``work_dir``
    The directory where the test case has been set up, a combination of
    ``base_work_dir`` and ``path``.

You can add other entries to the dictionary to pass information between the
:ref:`dev_testcase_collect`, :ref:`dev_testcase_configure` and
:ref:`dev_testcase_run`.  In the example above, ``resolution`` has been added
for this purpose.

.. _dev_testcase_collect:

collect()
^^^^^^^^^

The ``collect()`` function must call :py:func:`compass.testcase.add_step()` for
each step in the test case.

The argument to ``collect()`` is the dictionary ``testcase`` describing the
test case, which will include the keys and values from any keyword arguments
passed to :py:func:`compass.testcase.add_testcase()`. In the example below, the
resolution (as a string) as been passed in this way.

It is important that the ``collect()`` function doesn't perform any
time-consuming calculations, download files, or otherwise use significant
resources because this function is called quite often for every single test
case and step: when test cases are listed, set up, or cleaned up, and also when
test suites are set up or cleaned up.  However, it is okay to add input,
output, streams and namelist files to the steps in the test case by calling any
of the following functions:

* :py:func:`compass.io.add_input_file()`

* :py:func:`compass.io.add_output_file()`

* :py:func:`compass.namelist.add_namelist_file()`

* :py:func:`compass.namelist.add_namelist_options()`

* :py:func:`compass.streams.add_streams_file()`

Each of these functions just caches information about the the inputs, outputs,
namelists or streams files to be read later if the test case in question gets
set up, so each takes a negligible amount of time.

As an example, here is
:py:func:`compass.ocean.tests.baroclinic_channel.rpe_test.collect()`:

.. code-block:: python

    from compass.testcase import set_testcase_subdir, add_step
    from compass.ocean.tests.baroclinic_channel import initial_state, forward
    from compass.ocean.tests.baroclinic_channel.rpe_test import analysis
    from compass.namelist import add_namelist_file
    from compass.streams import add_streams_file


    def collect(testcase):
        """
        Update the dictionary of test case properties and add steps

        Parameters
        ----------
        testcase : dict
            A dictionary of properties of this test case, which can be updated
        """
        resolution = testcase['resolution']
        testcase['description'] = 'baroclinic channel {} reference potential '\
                                  'energy (RPE)'.format(resolution)

        nus = [1, 5, 10, 20, 200]

        res_params = {'1km': {'cores': 144, 'min_cores': 36,
                              'max_memory': 64000, 'max_disk': 64000},
                      '4km': {'cores': 36, 'min_cores': 8,
                              'max_memory': 16000, 'max_disk': 16000},
                      '10km': {'cores': 8, 'min_cores': 4,
                               'max_memory': 2000, 'max_disk': 2000}}

        if resolution not in res_params:
            raise ValueError('Unsupported resolution {}. Supported values are: '
                             '{}'.format(resolution, list(res_params)))

        defaults = res_params[resolution]

        subdir = '{}/{}'.format(resolution, testcase['name'])
        set_testcase_subdir(testcase, subdir)

        add_step(testcase, initial_state, resolution=resolution)

        for index, nu in enumerate(nus):
            name = 'rpe_test_{}_nu_{}'.format(index+1, nu)
            # we pass the defaults for the resolution on as keyword arguments
            step = add_step(testcase, forward, name=name, subdir=name, threads=1,
                            nu=float(nu), resolution=resolution, **defaults)

            # add the local namelist and streams file
            add_namelist_file(
                step, 'compass.ocean.tests.baroclinic_channel.rpe_test',
                'namelist.forward')
            add_streams_file(
                step, 'compass.ocean.tests.baroclinic_channel.rpe_test',
                'streams.forward')

        add_step(testcase, analysis, resolution=resolution, nus=nus)

We have deliberately chosen a fairly complex example to demonstrate how to make
full use of :ref:`dev_code_sharing` in a test case.

The test case imports the modules for its steps (``initial_state``,
``forward``, and ``analysis`` in this case) so it can call
:py:func:`compass.testcase.add_step()`, passing each as an argument.  In the
process, the steps are added to the ``steps`` dictionary (see
:ref:`dev_steps`).

By default, the test ase will go into a directory with the same name as the
test case (``rpe_test`` in this case).  However, ``compass`` is flexible
about the subdirectory structure and the names of the subdirectories.  This
flexibility was an important requirement in moving away from
:ref:`legacy_compass`.  Each test case and step must end up in a unique
directory, so it may be important that the name and subdirectory of each test
case or step depends in some way on the arguments passed to
:py:func:`compass.testcase.add_testcase()` or
:py:func:`compass.testcase.add_step()`.  In the example above, the
``baroclinic_channel`` configuration calls
:py:func:`compass.testcase.add_testcase()` with each of the 3 supported
resolutions.  We use :py:func:`compass.testcase.set_testcase_subdir()` to
give the test case a unique directory for each resolution: ``1km/rpe_test``,
``4km/rpe_test`` and ``10km/rpe_test``, .

In the example above, the same ``forward`` step is included in the test case
5 times with a different viscosity parameter ``nu`` for each.  The value of
``nu`` is passed to :py:func:`compass.testcase.add_step()`, along with
the unique ``name`` and ``subdir`` of the step, and several other parameters:
``resolution``, ``cores``, ``min_cores``, ``max_memory``, and ``max_disk``.
(We use a trick to pass the last 4 of these with the ``defaults`` dictionary
using the ``**defaults`` argument.)  In this example, the steps are given
rather clumsy names---``rpe_test_1_nu_1``, ``rpe_test_2_nu_5``, etc.---but
these could be any unique names.

.. _dev_testcase_configure:

configure()
^^^^^^^^^^^

The ``configure()`` function is used to set config options or build them up
from defaults stored in config files within the test case or its configuration.
The ``config`` object that is modified in this function will be written to a
config file for the test case (see :ref:`config_files`). We already discussed
the ``configure()`` function a little bit in :ref:`dev_configs` because
it is common for test cases to call a shared ``configure()`` function.

``configure()`` always takes two arguments, the ``testcase`` dictionary that
was returned by ``collect()`` and the ``config`` object with config options
to add or modify.

:py:func:`compass.ocean.tests.baroclinic_channel.rpe_test.configure()` simply
calls the shared function in its configuration,
:py:func:`compass.ocean.tests.baroclinic_channel.configure()`:

.. code-block:: python

    from compass.ocean.tests import baroclinic_channel


    def configure(testcase, config):
        """
        Modify the configuration options for this testcase.

        Parameters
        ----------
        testcase : dict
            A dictionary of properties of this testcase from the ``collect()``
            function

        config : configparser.ConfigParser
            Configuration options for this testcase, a combination of the defaults
            for the machine, core and configuration
        """
        baroclinic_channel.configure(testcase, config)


:py:func:`compass.ocean.tests.baroclinic_channel.configure()` was already
shown in :ref:`dev_configs` above.  It sets parameters for the number of
cells in the mesh in the x and y directions and the resolution of those cells.

In a pinch, the ``configure()`` function can also be used to perform other
operations at the test-case level during when a test case is being set up.
An example of this would be creating a symlink to a README file that is shared
across the whole test case, as in
:py:func:`compass.ocean.tests.global_ocean.files_for_e3sm.configure()`:


.. code-block:: python

    from importlib.resources import path

    from compass.ocean.tests import global_ocean
    from compass.io import symlink


    def configure(testcase, config):
        """
        Modify the configuration options for this testcase.

        Parameters
        ----------
        testcase : dict
            A dictionary of properties of this testcase from the ``collect()``
            function

        config : configparser.ConfigParser
            Configuration options for this testcase, a combination of the defaults
            for the machine, core and configuration
        """
        global_ocean.configure(testcase, config)
        with path('compass.ocean.tests.global_ocean.files_for_e3sm', 'README') as \
                target:
            symlink(str(target), '{}/README'.format(testcase['work_dir']))


The ``configure()`` function is not the right place for adding or altering
entries in the :ref:`dev_testcase_dict`.

.. _dev_testcase_run:

run()
^^^^^

``run()`` takes 4 arguments:

``testcase``
   a dictionary of properties of this testcase returned by ``collect()``,

``test_suite``
   a dictionary of properties of the test suite (not currently used),

``config``
   the config options for this testcase (see :ref:`config_files`),

``logger``
   a :py:class:`logging.Logger` for output from the testcase.

In its simplest form, ``run()`` just calls
:py:func:`compass.testcase.run_steps()` with the same arguments to run all of
the steps of the test case:

.. code-block:: python

    from compass.testcase import run_steps


    def run(testcase, test_suite, config, logger):
        """
        Run each step of the testcase

        Parameters
        ----------
        testcase : dict
            A dictionary of properties of this testcase from the ``collect()``
            function

        test_suite : dict
            A dictionary of properties of the test suite

        config : configparser.ConfigParser
            Configuration options for this testcase, a combination of the defaults
            for the machine, core and configuration

        logger : logging.Logger
            A logger for output from the testcase
        """
        # just run all the steps in the order they were added
        run_steps(testcase, test_suite, config, logger)


``run()`` is also the right place to perform :ref:`dev_validation` of variables
in output files and/or timers in a simulation log.

In some circumstances, it will also be appropriate to update properties of
the steps in the test case based on config options that the user may have
changed.  This should only be necessary for config options related to the
resources used by the step: the target number of cores, the minimum number of
cores, the number of threads, the maximum memory usage, and the maximum disk
usage.  Other config options can simply be read in from within the step's
``run()`` function as needed.  But these performance-related config options
affect how the step runs and must be set *before* the step can run.

In this complex example,
:py:func:`compass.ocean.tests.global_ocean.init.run()`, we see examples of both
updating the ``steps`` dictionary based on config options and of validation of
variables in the output:

.. code-block:: python

    def run(testcase, test_suite, config, logger):
        """
        Run each step of the testcase

        Parameters
        ----------
        testcase : dict
            A dictionary of properties of this testcase from the ``collect()``
            function

        test_suite : dict
            A dictionary of properties of the test suite

        config : configparser.ConfigParser
            Configuration options for this testcase, a combination of the defaults
            for the machine, core and configuration

        logger : logging.Logger
            A logger for output from the testcase
        """
        work_dir = testcase['work_dir']
        with_bgc = testcase['with_bgc']
        steps = testcase['steps_to_run']
        if 'initial_state' in steps:
            step = testcase['steps']['initial_state']
            # get the these properties from the config options
            for option in ['cores', 'min_cores', 'max_memory', 'max_disk',
                           'threads']:
                step[option] = config.getint('global_ocean',
                                             'init_{}'.format(option))

        if 'ssh_adjustment' in steps:
            step = testcase['steps']['ssh_adjustment']
            # get the these properties from the config options
            for option in ['cores', 'min_cores', 'max_memory', 'max_disk',
                           'threads']:
                step[option] = config.getint('global_ocean',
                                             'forward_{}'.format(option))

        run_steps(testcase, test_suite, config, logger)

        if 'initial_state' in steps:
            variables = ['temperature', 'salinity', 'layerThickness']
            compare_variables(variables, config, work_dir,
                              filename1='initial_state/initial_state.nc')

            if with_bgc:
                variables = ['temperature', 'salinity', 'layerThickness', 'PO4',
                             'NO3', 'SiO3', 'NH4', 'Fe', 'O2', 'DIC',
                             'DIC_ALT_CO2', 'ALK', 'DOC', 'DON', 'DOFe', 'DOP',
                             'DOPr', 'DONr', 'zooC', 'spChl', 'spC', 'spFe',
                             'spCaCO3', 'diatChl', 'diatC', 'diatFe', 'diatSi',
                             'diazChl', 'diazC', 'diazFe', 'phaeoChl', 'phaeoC',
                             'phaeoFe', 'DMS', 'DMSP', 'PROT', 'POLY', 'LIP']
                compare_variables(variables, config, work_dir,
                                  filename1='initial_state/initial_state.nc')

        if 'ssh_adjustment' in steps:
            variables = ['ssh', 'landIcePressure']
            compare_variables(variables, config, work_dir,
                              filename1='ssh_adjustment/adjusted_init.nc')


As mentioned in :ref:`dev_testcase_dict`, the ``steps_to_run`` entry may either
be the full list of steps from the test case that would typically be run to
complete the test case (the value given to it in :ref:`dev_testcase_collect`)
or it may be a single test case because the user is running the steps manually,
one at a time.  For this reason, it is always a good idea to check if a given
step is being run before altering the entries in :ref:`dev_step_dict` based on
config options, as shown in the example.  Similarly, it is important to check
if the step was run before running validation.  Otherwise, the validation may
fail merely because the user didn't ask for that particular step (yet).

.. _dev_steps:

Steps
-----

Steps are the smallest units of work that can be executed on their own in
``compass``.  All test cases are made up of 1 or more steps, and all steps
are set up into subdirectories inside of the work directory for the test case.
Typically, a user will run all steps in a test case but certain test cases may
prefer to have steps that are not run by default (e.g. a long forward
simulation or optional visualization) but which are available for a user to
manually alter and then run on their own.

A step is described by a ``step`` dictionary and has :ref:`dev_step_collect`,
:ref:`dev_step_setup`, and :ref:`dev_step_run` functions, described below.

.. _dev_step_inputs_outputs:

inputs and outputs
^^^^^^^^^^^^^^^^^^

Currently, steps run in sequence in the order they are added to the test case
(or in the order they appear in ``testcase['steps_to_run']``).  There are plans
to allow test cases and their steps to run in parallel in the future. For this
reason, we require that each step defines a list of the absolute paths to
all input files that could come from other steps (possibly in other test cases)
and all outputs from the step that might be used by other steps (again,
possibly in other test cases).  There is no harm in including inputs to the
step that do not come from other steps (e.g. files that will be downloaded
when the test case gets set up) as long as they are sure to exist before the
step runs.  Likewise, there is no harm in including outputs from the step that
aren't used by any other steps in any test cases as long as the step will be
sure to generate them.

The inputs and outputs need to be defined during :ref:`dev_step_collect` or
:ref:`dev_step_setup` because they are needed before :ref:`dev_step_run` is
called (to determine which steps depend on which other steps).  Inputs are
added with :py:func:`compass.io.add_input_file()` and outputs with
:py:func:`compass.io.add_output_file()`, see :ref:`dev_io`.  Inputs may be
symbolic links to files in ``compass``, from the various databases on the
`LCRC server <https://web.lcrc.anl.gov/public/e3sm/mpas_standalonedata/>`_,
downloaded from another source, or from another step.

Because the inputs and outputs need to be defined before the step runs, there
can be some cases to avoid.  The name of an output file should not depend on a
config option.  Otherwise, if the user changes the config option, the file
actually created may have a different name than expected, in which case the
step will fail.  This would be true even if a subsequent step would have been
able to read in the same config option and modify the name of the expected
input file.

Along the same lines, an input or output file name should not depend on data
from an input file that does not exist during :ref:`dev_step_setup`.  Since the
file does not exist, there is no way to read the file with the dependency
within :ref:`dev_step_setup` and determine the resulting input or output file
name.

Both of these issues have arisen for the
:ref:`dev_ocean_global_ocean_files_for_e3sm` test case from the
:ref:`dev_ocean_global_ocean` configuration.  Output files are named using the
"short name" of the mesh in E3SM, which depends both on config options and on
the number of vertical levels, which is read in from a mesh file created in a
previous step.  For now, the outputs of this step are not used by any other
steps so it is safe to simply omit them, but this could become problematic in
the future if new steps are added that depend on
:ref:`dev_ocean_global_ocean_files_for_e3sm`.

.. _dev_step_dict:

step dictionary
^^^^^^^^^^^^^^^

Just as a test case is described by a :ref:`dev_testcase_dict`, we use a
python dictionary ``step`` to keep track of data (other than config options)
that are needed to collect, setup and run a step.  The ``step`` dictionary will
typically look like this example from the
``ocean/baroclinic_channel/10km/default/initial_state`` step at the beginning
of :py:func:`compass.ocean.tests.baroclinic_channel.initial_state.run()`:

.. code-block:: python


    step = {'base_work_dir': '/home/xylar/data/mpas/test_new_run_model/nightly/new_api',
            'config': 'default.cfg',
            'configuration': 'baroclinic_channel',
            'core': 'ocean',
            'cores': 1,
            'inputs': [],
            'max_disk': 8000,
            'max_memory': 8000,
            'min_cores': 1,
            'module': 'compass.ocean.tests.baroclinic_channel.initial_state',
            'name': 'initial_state',
            'outputs': ['/home/xylar/data/mpas/test_new_run_model/nightly/new_api/ocean/baroclinic_channel/10km/default/initial_state/base_mesh.nc',
                        '/home/xylar/data/mpas/test_new_run_model/nightly/new_api/ocean/baroclinic_channel/10km/default/initial_state/culled_mesh.nc',
                        '/home/xylar/data/mpas/test_new_run_model/nightly/new_api/ocean/baroclinic_channel/10km/default/initial_state/culled_graph.info',
                        '/home/xylar/data/mpas/test_new_run_model/nightly/new_api/ocean/baroclinic_channel/10km/default/initial_state/ocean.nc'],
            'path': 'ocean/baroclinic_channel/10km/default/initial_state',
            'resolution': '10km',
            'run': 'run',
            'setup': 'setup',
            'subdir': 'initial_state',
            'testcase': 'default',
            'testcase_subdir': '10km/default',
            'threads': 1,
            'work_dir': '/home/xylar/data/mpas/test_new_run_model/nightly/new_api/ocean/baroclinic_channel/10km/default/initial_state'}

``base_work_dir``
    The base directory where the test cases or test suite have been set up.

``config``
    The config file for the test case, typically ``<name>.cfg``, where
    ``<name>`` is the name of the test case.

``configuration``
    Which of the :ref:`dev_configs` this test case belongs to.  This entry is
    added automatically by :py:func:`compass.testcase.add_step()` when it
    is called in the configuration's ``collect()`` function and should not be
    modified.

``core``
    Which of the :ref:`dev_cores` this test case belongs to.  This entry is
    added automatically by :py:func:`compass.testcase.add_step()` when it
    is called in the configuration's ``collect()`` function and should not be
    modified.

``cores``
    The "target" number of cores that the step would ideally run on if that
    number is available.  This entry should be set via keyword argument to
    :py:func:`compass.testcase.add_step()`, in :ref:`dev_step_collect`
    or  in :ref:`dev_step_setup` if it is known in advance, or in the test
    case's :ref:`dev_testcase_run` if it comes from a config option that a user
    might alter.

``inputs``
    A list of absolute paths of input files to the step, added with calls to
    :py:func:`compass.io.add_input_file()`.

``max_disk``
    The maximum amount of disk space the step is allowed to use.  This is a
    placeholder for the time being and is not used. This entry should be set
    via keyword argument to :py:func:`compass.testcase.add_step()`, in
    :ref:`dev_step_collect` or in :ref:`dev_step_setup` if it is known in
    advance, or in the test case's :ref:`dev_testcase_run` if it comes from a
    config option that a user might alter.

``max_memory``
    The maximum amount of memory the step is allowed to use.  This is a
    placeholder for the time being and is not used. This entry should be set
    via keyword argument to :py:func:`compass.testcase.add_step()`, in
    :ref:`dev_step_collect` or in :ref:`dev_step_setup` if it is known in
    advance, or in the test case's :ref:`dev_testcase_run` if it comes from a
    config option that a user might alter.

``min_cores``
    The minimum number of cores that the step can run on.  If fewer cores are
    available on the system, the step will fail.  This entry should be set via
    keyword argument to :py:func:`compass.testcase.add_step()`, in
    :ref:`dev_step_collect` or  in :ref:`dev_step_setup` if it is known in
    advance, or in the test case's :ref:`dev_testcase_run` if it comes from a
    config option that a user might alter.

``module``
    The full name of the module where the step is defined. This entry is added
    automatically by :py:func:`compass.testcase.add_step()` when it is called
    in the test case's :ref:`dev_testcase_collect` function and should not been
    to be modified.

``name``
    The name of the step.  The default is the last part of ``module`` and is
    set by :py:func:`compass.testcase.add_step()`.  You can modify this entry
    by passing ``name=<name>`` as a keyword argument to this function when it
    is called in the test case's :ref:`dev_testcase_collect` function.

``outputs``
    A list of absolute paths of output files to the step, added with calls to
    :py:func:`compass.io.add_output_file()`.

``path``
    The relative path of the steps within the base work directory, the
    combination of the ``core``, ``configuration``, ``testcase_subdir`` and
    ``subdir``.  This entry is added automatically by the :ref:`dev_framework`
    after :ref:`dev_step_collect` is called and should not be modified.

``run``
    The name of the :ref:`dev_step_run` function for running the step,
    set by :py:func:`compass.testcase.add_step()`.  This entry should only be
    modified if you have an important reason not to name the function in your
    test case's module ``run``.

``steup``
    The name of the :ref:`dev_step_setup` function for setting up the step,
    set by :py:func:`compass.testcase.add_step()`.  This entry should only be
    modified if you have an important reason not to name the function in your
    test case's module ``setup``.

``subdir``
    The subdirectory for the step within the test case.  The default is the the
    last part of the ``module`` (the same as the default ``name``) and is set
    by :py:func:`compass.testcase.add_step()`.  You can modify this entry by
    passing ``subdir=<subdir>`` as a keyword argument to this function when it
    is called in the configuration's ``collect()`` function.

``testcase``
    The name of the test case that this step belongs to.  This comes from the
    test case and should not be modified by the step.

``testcase_subdir``
    The subdirectory for the test case within the configuration.  This comes
    from the test case and should not be modified by the step.

``threads``
    The number of threads used by the step.  This entry should be set via
    keyword argument to :py:func:`compass.testcase.add_step()`, in
    :ref:`dev_step_collect` or  in :ref:`dev_step_setup` if it is known in
    advance, or in the test case's :ref:`dev_testcase_run` if it comes from a
    config option that a user might alter.

``work_dir``
    The directory where the step has been set up, a combination of
    ``base_work_dir`` and ``path``.

You can add other entries to the dictionary to pass information between the
:ref:`dev_step_collect`, :ref:`dev_step_setup`, and :ref:`dev_step_run`.  In
the example above, ``resolution`` has been added for this purpose.

.. _dev_step_collect:

collect()
^^^^^^^^^

The arguments to ``collect()`` are the dictionaries ``testcase`` describing the
test case and ``step`` describing the step.  The latter will include the keys
and values from any keyword arguments passed to
:py:func:`compass.testcase.add_step()`. In the example below, the resolution
(as a string) as been passed in this way.

As with the test case's :ref:`dev_testcase_collect`, it is important that the
step's ``collect()`` function doesn't perform any time-consuming calculations,
download files, or otherwise use significant resources because this function is
called quite often for every single test case and step: when test cases are
listed, set up, or cleaned up, and also when test suites are set up or cleaned
up.  However, it is okay to add input, output, streams and namelist files to
the step by calling any of the following functions:

* :py:func:`compass.io.add_input_file()`

* :py:func:`compass.io.add_output_file()`

* :py:func:`compass.namelist.add_namelist_file()`

* :py:func:`compass.namelist.add_namelist_options()`

* :py:func:`compass.streams.add_streams_file()`

Each of these functions just caches information about the the inputs, outputs,
namelists or streams files to be read later if the test case in question gets
set up, so each takes a negligible amount of time.

The following is the contents of
:py:func:`compass.ocean.tests.baroclinic_channel.forward.collect()`:

.. code-block:: python

    from compass.namelist import add_namelist_file, add_namelist_options
    from compass.streams import add_streams_file


    def collect(testcase, step):
        """
        Update the dictionary of step properties

        Parameters
        ----------
        testcase : dict
            A dictionary of properties of this test case, which should not be
            modified here

        step : dict
            A dictionary of properties of this step, which can be updated
        """
        defaults = dict(max_memory=1000, max_disk=1000, threads=1)
        for key, value in defaults.items():
            step.setdefault(key, value)

        step.setdefault('min_cores', step['cores'])

        add_namelist_file(step, 'compass.ocean.tests.baroclinic_channel',
                          'namelist.forward')
        add_namelist_file(step, 'compass.ocean.tests.baroclinic_channel',
                          'namelist.{}.forward'.format(step['resolution']))
        if 'nu' in step:
            # update the viscosity to the requested value
            options = {'config_mom_del2': '{}'.format(step['nu'])}
            add_namelist_options(step, options)

        add_streams_file(step, 'compass.ocean.tests.baroclinic_channel',
                         'streams.forward')


A set of default parameters (``max_memory``, ``max_disk`` and ``threads``) is
added to ``step`` if these parameters have not already been set.  Similarly,
``min_cores`` is set to ``cores`` if it has not already been set.

Then, two files with modifications to the namelist options are added (for
later processing), and an additional config option is set manually via
a python dictionary of namelist options.

Finally, a file with modifications to the default streams is also added (again,
for later processing).

.. _dev_step_setup:

setup()
^^^^^^^

The ``setup()`` function is called when a user is setting up each step either
as part of a call to :ref:`dev_compass_setup` or :ref:`dev_compass_suite`.
As in :ref:`dev_step_collect`, you can add input, output, streams and namelist
files to the step by calling any of the following functions:

* :py:func:`compass.io.add_input_file()`

* :py:func:`compass.io.add_output_file()`

* :py:func:`compass.namelist.add_namelist_file()`

* :py:func:`compass.namelist.add_namelist_options()`

* :py:func:`compass.streams.add_streams_file()`

You can also add the contents of one or more :ref:`config_files` to the
``config`` object, or use ``config.set()`` to set config options directly.

If namelists and streams files have been defined, you should call
:py:func:`compass.namelist.generate_namelist()` and
:py:func:`compass.streams.generate_streams()` somewhere in ``setup()``.

If you are running the MPAS model, you should call
:py:func:`compass.model.add_model_as_input()` to create a symlink to the
MPAS model's executable.

Set up should not do any major computations or any time-consuming operations
other than downloading files.

As an example, here is
:py:func:`compass.ocean.tests.baroclinic_channel.forward.setup()`:

.. code-block:: python

    from compass.io import add_input_file, add_output_file
    from compass.namelist import generate_namelist
    from compass.streams import generate_streams
    from compass.model import add_model_as_input


    def setup(step, config):
        """
        Set up the test case in the work directory, including downloading any
        dependencies

        Parameters
        ----------
        step : dict
            A dictionary of properties of this step

        config : configparser.ConfigParser
            Configuration options for this test case, a combination of the defaults
            for the machine, core, configuration and test case
        """
        # generate the namelist and streams files file from the various files and
        # replacements we have collected
        generate_namelist(step, config)
        generate_streams(step, config)

        add_model_as_input(step, config)

        add_input_file(step, filename='init.nc',
                       target='../initial_state/ocean.nc')
        add_input_file(step, filename='graph.info',
                       target='../initial_state/culled_graph.info')

        add_output_file(step, filename='output.nc')

First, the namelist and streams file are generated.  Then, the model's
executable is linked (and included among the ``inputs``).  Two other input
files are added (symlinks from the ``initial_state`` step).  Finally, an output
file is added.

.. _dev_step_run:

run()
^^^^^

Okay, we're ready to define how the step will run!

The contents of ``run()`` can vary quite a lot between steps.

In the test ``baroclinic_channel`` configuration, the ``run()`` function for
the ``initial_state`` step,
:py:func:`compass.ocean.tests.baroclinic_channel.initial_state.run()`, is quite
involved:

.. code-block:: python

    import os
    import xarray
    import numpy

    from mpas_tools.planar_hex import make_planar_hex_mesh
    from mpas_tools.io import write_netcdf
    from mpas_tools.mesh.conversion import convert, cull

    from compass.ocean.vertical import generate_grid


    def run(step, test_suite, config, logger):
        """
        Run this step of the testcase

        Parameters
        ----------
        step : dict
            A dictionary of properties of this step from the ``collect()``
            function, with modifications from the ``setup()`` function.

        test_suite : dict
            A dictionary of properties of the test suite

        config : configparser.ConfigParser
            Configuration options for this testcase, a combination of the defaults
            for the machine, core and configuration

        logger : logging.Logger
            A logger for output from the step
       """
        section = config['baroclinic_channel']
        nx = section.getint('nx')
        ny = section.getint('ny')
        dc = section.getfloat('dc')

        dsMesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc, nonperiodic_x=False,
                                      nonperiodic_y=True)
        write_netcdf(dsMesh, 'base_mesh.nc')

        dsMesh = cull(dsMesh, logger=logger)
        dsMesh = convert(dsMesh, graphInfoFileName='culled_graph.info',
                         logger=logger)
        write_netcdf(dsMesh, 'culled_mesh.nc')

        section = config['baroclinic_channel']
        use_distances = section.getboolean('use_distances')
        gradient_width_dist = section.getfloat('gradient_width_dist')
        gradient_width_frac = section.getfloat('gradient_width_frac')
        bottom_temperature = section.getfloat('bottom_temperature')
        surface_temperature = section.getfloat('surface_temperature')
        temperature_difference = section.getfloat('temperature_difference')
        salinity = section.getfloat('salinity')
        coriolis_parameter = section.getfloat('coriolis_parameter')

        ds = dsMesh.copy()

        interfaces = generate_grid(config=config)

        bottom_depth = interfaces[-1]
        vert_levels = len(interfaces) - 1

        ds['refBottomDepth'] = ('nVertLevels', interfaces[1:])
        ds['refZMid'] = ('nVertLevels', -0.5 * (interfaces[1:] + interfaces[0:-1]))
        ds['vertCoordMovementWeights'] = xarray.ones_like(ds.refBottomDepth)

        xCell = ds.xCell
        yCell = ds.yCell

        xMin = xCell.min().values
        xMax = xCell.max().values
        yMin = yCell.min().values
        yMax = yCell.max().values

        yMid = 0.5*(yMin + yMax)
        xPerturbMin = xMin + 4.0 * (xMax - xMin) / 6.0
        xPerturbMax = xMin + 5.0 * (xMax - xMin) / 6.0

        if use_distances:
            perturbationWidth = gradient_width_dist
        else:
            perturbationWidth = (yMax - yMin) * gradient_width_frac

        yOffset = perturbationWidth * numpy.sin(
            6.0 * numpy.pi * (xCell - xMin) / (xMax - xMin))

        temp_vert = (bottom_temperature +
                     (surface_temperature - bottom_temperature) *
                     ((ds.refZMid + bottom_depth) / bottom_depth))

        frac = xarray.where(yCell < yMid - yOffset, 1., 0.)

        mask = numpy.logical_and(yCell >= yMid - yOffset,
                                 yCell < yMid - yOffset + perturbationWidth)
        frac = xarray.where(mask,
                            1. - (yCell - (yMid - yOffset)) / perturbationWidth,
                            frac)

        temperature = temp_vert - temperature_difference * frac
        temperature = temperature.transpose('nCells', 'nVertLevels')

        # Determine yOffset for 3rd crest in sin wave
        yOffset = 0.5 * perturbationWidth * numpy.sin(
            numpy.pi * (xCell - xPerturbMin) / (xPerturbMax - xPerturbMin))

        mask = numpy.logical_and(
            numpy.logical_and(yCell >= yMid - yOffset - 0.5 * perturbationWidth,
                              yCell <= yMid - yOffset + 0.5 * perturbationWidth),
            numpy.logical_and(xCell >= xPerturbMin,
                              xCell <= xPerturbMax))

        temperature = (temperature +
                       mask * 0.3 * (1. - ((yCell - (yMid - yOffset)) /
                                           (0.5 * perturbationWidth))))

        temperature = temperature.expand_dims(dim='Time', axis=0)

        layerThickness = xarray.DataArray(data=interfaces[1:] - interfaces[0:-1],
                                          dims='nVertLevels')
        _, layerThickness = xarray.broadcast(xCell, layerThickness)
        layerThickness = layerThickness.transpose('nCells', 'nVertLevels')
        layerThickness = layerThickness.expand_dims(dim='Time', axis=0)

        normalVelocity = xarray.zeros_like(ds.xEdge)
        normalVelocity, _ = xarray.broadcast(normalVelocity, ds.refBottomDepth)
        normalVelocity = normalVelocity.transpose('nEdges', 'nVertLevels')
        normalVelocity = normalVelocity.expand_dims(dim='Time', axis=0)

        ds['temperature'] = temperature
        ds['salinity'] = salinity * xarray.ones_like(temperature)
        ds['normalVelocity'] = normalVelocity
        ds['layerThickness'] = layerThickness
        ds['restingThickness'] = layerThickness
        ds['bottomDepth'] = bottom_depth * xarray.ones_like(xCell)
        ds['maxLevelCell'] = vert_levels * xarray.ones_like(xCell, dtype=int)
        ds['fCell'] = coriolis_parameter * xarray.ones_like(xCell)
        ds['fEdge'] = coriolis_parameter * xarray.ones_like(ds.xEdge)
        ds['fVertex'] = coriolis_parameter * xarray.ones_like(ds.xVertex)

        write_netcdf(ds, 'ocean.nc')

Without going into all the details of this function, it creates a mesh that
is periodic in x (but not y), then adds a vertical grid and an initial
condition to an :py:class:`xarray.Dataset`, which is then written out to
the file ``ocean.nc``.

In the example step we've been using,
:py:func:`compass.ocean.tests.baroclinic_channel.forward.run()` looks like
this:

.. code-block:: python

    from compass.model import run_model


    def run(step, test_suite, config, logger):
        """
        Run this step of the test case

        Parameters
        ----------
        step : dict
            A dictionary of properties of this step

        test_suite : dict
            A dictionary of properties of the test suite

        config : configparser.ConfigParser
            Configuration options for this test case, a combination of the defaults
            for the machine, core and configuration

        logger : logging.Logger
            A logger for output from the step
        """
        run_model(step, config, logger)

the :py:func:`compass.model.run_model()` function takes care of updating the
namelist options for the test case to make sure the PIO tasks and stride are
consistent with the requested number of cores, creates a graph partition for
the requested number of cores, and runs the model.

To get a feel for different types of ``run()`` functions, it may be best to
explore different test cases.

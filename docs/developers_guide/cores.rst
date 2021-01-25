.. _dev_cores:

Cores
=====

The test cases in compass are organized by "core", corresponding to a dynamical
core in MPAS, and then into "configurations".  Currently, there are two cores,
``examples`` which simply houses some very basic examples (as the name implies)
and ``ocean``, which encompasses all the test cases for MPAS-Ocean.   Test
cases for MALI will be added to the ``landice`` core in the near future.

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
"init mode").  From the ``examples`` core, these the MPAS dynamical core
is given the dummy name ``core`` (which does not actually exist).  This would
be replaced by ``ocean`` or ``landice`` throughout the config file for those
cores:

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

.. _dev_configs:

Configurations
--------------

Configurations are the next level of test-case organization below
:ref:`dev_cores`.  Typically, the test cases within a configuration are part of
the same framework, serve a similar purpose, or are variants on one another.
Often, they have a common topography and initial condition, perhaps with
different mesh resolutions.  It is common for a configuration to include
"framework" modules that are shared between its test cases and steps (but
typically not with other configurations).  Each core will typically include a
mix of "idealized" configurations (e.g. :ref:`dev_ocean_baroclinic_channel` or
:ref:`dev_ocean_ziso`) and "realistic" domains (e.g.
:ref:`dev_ocean_global_ocean`).

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
typically go into a section with the same name as teh configuration, as in the
example above.

The ``__init__.py`` file for the configuration must define a ``collect()``
function that makes a list of test cases within the configuration.  This list
is made by calling the ``collect()`` functions of each test case.  Returning
to the ``example_compact`` configuration, the function
:py:func:`compass.examples.test.example_compact.collect()` looks like this:

.. code-block:: python

    from compass.examples.tests.example_compact import test1, test2


    def collect():
        testcases = list()
        for resolution in ['1km', '2km']:
            for test in [test1, test2]:
                testcases.append(test.collect(resolution=resolution))

        return testcases

As in this example, it may be useful for a configuration to make several
versions of a test case by passing different parameters.  In the example, we
create versions of both ``test1`` and ``test2`` at both ``1km`` and ``2km``
resolution.  We will explore this further when we talk about
:ref:`dev_testcases` and :ref:`dev_steps` below.

It is also common for a configuration to have a ``configure()`` function that
can be shared across its tests, see :ref:`dev_testcase_configure`.

An example of a shared ``configure()`` function is
:py:func:`compass.ocean.test.baroclinic_channel.configure()`:

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

In many ways, test cases are the fundamental building blocks of ``compass``,
since a user can't set up an individual step of test case (tough they can run
the steps one at a time).

A test case can be a module but is usually a python package so it can
incorporate modules for its steps and/or config files, namelists, and streams
files.  The test case must include ``collect()``, ``configure()`` and ``run()``
functions with the `API <https://en.wikipedia.org/wiki/API>`_ given below.
(Technically, you can name these functions something else but we don't suggest
doing this.)

.. _dev_testcase_dict:

testcase dictionary
^^^^^^^^^^^^^^^^^^^

As discussed previously, we have opted to keep track of the data associated
with a test case using :ref:`dev_dicts_not_classes`.

The ``testcase`` dictionary will typically look like this example from the
``ocean/baroclinic_channel/10km/default`` test case at the beginning of
:py:func:`compass.ocean.tests.baroclinic_channel.default.run()`:

.. code-block:: python


    testcase = {'module': 'compass.ocean.tests.baroclinic_channel.default',
                'description': 'baroclinic channel 10km default',
                'steps': {
                    'initial_state': {
                        'module': 'compass.ocean.tests.baroclinic_channel.initial_state',
                        'name': 'initial_state',
                        'subdir': 'initial_state',
                        'setup': 'setup',
                        'run': 'run',
                        'inputs': [],
                        'outputs': [],
                        'resolution': '10km',
                        'cores': 1,
                        'min_cores': 1,
                        'max_memory': 8000,
                        'max_disk': 8000,
                        'testcase': 'default',
                        'testcase_subdir': '10km/default'},
                    'forward': {
                        'module': 'compass.ocean.tests.baroclinic_channel.forward',
                        'name': 'forward',
                        'subdir': 'forward',
                        'setup': 'setup',
                        'run': 'run',
                        'inputs': [],
                        'outputs': [],
                        'resolution': '10km',
                        'cores': 4,
                        'max_memory': 1000,
                        'max_disk': 1000,
                        'min_cores': 4,
                        'threads': 1,
                        'testcase': 'default',
                        'testcase_subdir': '10km/default'}},
                'name': 'default',
                'core': 'ocean',
                'configuration': 'baroclinic_channel',
                'subdir': '10km/default',
                'path': 'ocean/baroclinic_channel/10km/default',
                'configure': 'configure',
                'run': 'run',
                'new_step_log_file': True,
                'steps_to_run': ['initial_state', 'forward'],
                'resolution': '10km'}

``module``
    The full name of the module or package where the test case is defined.
    This entry should be defined by passing ``module=__name__`` as an argument
    to :py:func:`compass.testcase.get_testcase_default()` in
    :ref:`dev_testcase_collect`.

``description``
    A short (one line) description of the test case.  Typically, this is
    similar to the ``path`` of the test, just put into words.  This is passed
    as the ``description`` argument to
    :py:func:`compass.testcase.get_testcase_default()` in
    :ref:`dev_testcase_collect`.

``steps``
    A dictionary of steps in the test case with the names of the steps as keys
    and each :ref:`dev_step_dict` as the corresponding value.  The ``steps``
    dictionary should created in :ref:`dev_testcase_collect` by calling each
    step's :ref:`dev_step_collect` and then passed in as the ``steps`` argument
    to :py:func:`compass.testcase.get_testcase_default()`.

``name``
    The name of the test case.  The default is the last part of ``module`` and
    is set by :py:func:`compass.testcase.get_testcase_default()`.  You can
    modify this entry in :ref:`dev_testcase_collect` anytime after calling
    ``get_testcase_default()``.

``core``
    Which of the :ref:`dev_cores` this test case belongs to.  This entry is
    added automatically by :py:func:`compass.testcase.get_testcase_default()`
    when it is called in :ref:`dev_testcase_collect` and should not be
    modified.

``configuration``
    Which of the :ref:`dev_configs` this test case belongs to.  This entry is
    added automatically by :py:func:`compass.testcase.get_testcase_default()`
    when it is called in :ref:`dev_testcase_collect` and should not be
    modified.

``subdir``
    The subdirectory for the test case within the configuration.  The default
    is the the last part of the ``module`` and is set by
    :py:func:`compass.testcase.get_testcase_default()`.  You can modify this
    entry in :ref:`dev_testcase_collect` anytime after calling
    ``get_testcase_default()``.

``path``
    The relative path of the test case within the base work directory, the
    combination of the ``core``, ``configuration`` and ``subdir``.  This entry
    is added automatically by the :ref:`dev_framework` after
    :ref:`dev_testcase_collect` is called and should not be modified.

``configure``
    The name of the :ref:`dev_testcase_configure` function for setting config
    options, set by :py:func:`compass.testcase.get_testcase_default()`.  This
    entry should only be modified if you have an important reason not to name
    the function in your test case's module ``configure``.

``run``
    The name of the :ref:`dev_testcase_run` function for running the test case,
    set by :py:func:`compass.testcase.get_testcase_default()`.  This entry
    should only be modified if you have an important reason not to name the
    function in your test case's module ``run``.

``new_step_log_file``
    An entry used by the compass :ref:`dev_framework` to determine if the steps
    of this test case need their own log files or if they should perform
    :ref:`dev_logging` to the same logger as the test case itself.  This entry
    should not be altered.

``steps_to_run``
    A list of the steps to run.  By default, this is the names of all of the
    steps in ``steps`` in the order they were added.  You can modify these
    in :ref:`dev_testcase_collect` after calling
    :py:func:`compass.testcase.get_testcase_default()` if some steps should not
    be run by default. If a user asks to run a single step from the test case,
    the :ref:`dev_testcase_run` function for test case is still called but with
    this list set to just the name of the step to run.

You can add other entries to the dictionary to pass information between the
:ref:`dev_testcase_collect`, :ref:`dev_testcase_configure` and
:ref:`dev_testcase_run`.  In the example above, ``resolution`` has been added
for this purpose.

.. _dev_testcase_collect:

collect()
^^^^^^^^^

The ``collect()`` function must include the following, each of which is
described in more detail below:

1. call the ``collect()`` functions for the steps in the test case, adding them
   to a ``steps`` dictionary,

2. call :py:func:`compass.testcase.get_testcase_default()`

3. return the resulting python dictionary ``testcase``.

You can include argument (typically parameters) to ``collect()`` as long as
the configuration's ``collect()`` function will know what these should be.  In
the example below, the argument is the resolution (as a string).

It is important that the ``collect()`` function doesn't perform any
time-consuming calculations, download files, or otherwise use significant
resources because this function is called quite often for every single test
case and step: when test cases are listed, set up, or cleaned up, and also when
test suites are set up or cleaned up.

Since the API for ``collect()`` is a bit flexible, we will provide an example,
:py:func:`compass.ocean.tests.baroclinic_channel.rpe_test.collect()`:

.. code-block:: python

    from compass.testcase import get_testcase_default
    from compass.ocean.tests.baroclinic_channel import initial_state, forward


    def collect(resolution):
        """
        Get a dictionary of testcase properties

        Parameters
        ----------
        resolution : {'1km', '4km', '10km'}
            The resolution of the mesh

        Returns
        -------
        testcase : dict
            A dict of properties of this test case, including its steps
        """
        description = 'baroclinic channel {} reference potential energy (RPE)' \
                      ''.format(resolution)
        module = __name__

        res_params = {'1km': {'core_count': 144, 'min_cores': 36,
                              'max_memory': 64000, 'max_disk': 64000},
                      '4km': {'core_count': 36, 'min_cores': 8,
                              'max_memory': 16000, 'max_disk': 16000},
                      '10km': {'core_count': 8, 'min_cores': 4,
                               'max_memory': 2000, 'max_disk': 2000}}

        if resolution not in res_params:
            raise ValueError('Unsupported resolution {}. Supported values are: '
                             '{}'.format(resolution, list(res_params)))

        res_params = res_params[resolution]
        name = module.split('.')[-1]
        subdir = '{}/{}'.format(resolution, name)
        steps = dict()
        step = initial_state.collect(resolution)
        steps[step['name']] = step

        for index, nu in enumerate([1, 5, 10, 20, 200]):
            step = forward.collect(resolution, cores=res_params['core_count'],
                                   min_cores=res_params['min_cores'],
                                   max_memory=res_params['max_memory'],
                                   max_disk=res_params['max_disk'], threads=1,
                                   testcase_module=module,
                                   namelist_file='namelist.forward',
                                   streams_file='streams.forward',
                                   nu=float(nu))
            step['name'] = 'rpe_test_{}_nu_{}'.format(index+1, nu)
            step['subdir'] = step['name']
            steps[step['name']] = step

        step = analysis.collect(resolution)
        steps[step['name']] = step

        testcase = get_testcase_default(module, description, steps, subdir=subdir)
        testcase['resolution'] = resolution

        return testcase

We have deliberately chosen a fairly complex example to demonstrate how to make
full use of :ref:`dev_code_sharing` in a test case.

The test case imports the modules for its steps (``initial_state`` and
``forward`` in this case) so it can call the ``collect()`` function for each
step.  The steps are collected in a python dictionary ``steps`` with the names
of the steps as keys and individual ``step`` dictionaries as values (so a
nested dictionary).  The ``step`` dictionary is described in :ref:`dev_steps`.

Then, :py:func:`compass.testcase.get_testcase_default()` is called.  The
required arguments are the current module, a short description of the test
case, and the ``steps`` dictionary. The name of the module is determined from
the `__name__ <https://docs.python.org/3/reference/import.html?highlight=__name__#__name__>`_
attribute of the package or module.  This will automatically detect and set the
default values for the ``name``, ``core``, ``configuration``, ``subdir``,
``configure``, ``run``, and ``steps_to_run`` entries in the
:ref:`dev_testcase_dict`.  After the call, you can update any of these
(typically just ``name`` and ``subdir``) that you need to.  None of these
entries should be altered in :ref:`dev_testcase_configure` or
:ref:`dev_testcase_run`; they are fixed properties of the test case once it
has been "collected".

By default, the test case will get set up in a subdirectory of the
configuration that is the name of the individual module or package (e.g.
``rpe_test`` for in the example above, since the package is called
``rpe_test``).  Similarly, by default each step will go into a subdirectory
with the module name of the step (e.g. ``initial_state`` or ``forward``).
However, ``compass`` is flexible about the subdirectory structure and the names
of the subdirectories.  This flexibility was an important requirement in
moving away from :ref:`legacy_compass`.  You can give the subdirectory for
the test case and steps whatever name makes sense to you. If an argument is
passed to the test case's ``collect()`` function, it would typically make sense
to have the subdirectory of the test case depend in some way on this argument.
This is because each test case must end up in a unique subdirectory.  In the
example above, the ``baroclinic_channel`` configuration will call ``collect()``
with each of the 3 supported resolutions.  Each test case will go into a
different subdirectory: ``1km/rpe_test``, ``4km/rpe_test`` and
``10km/rpe_test``.

In the example above, the same ``forward`` step is included in the test case
5 times with a different viscosity parameter ``nu`` for each.  The value of
``nu`` is passed to the step's ``collect()`` function (along with a number of
other parameters related to required resources, namelists and streams files).
The resulting ``step`` dictionary will give each step the same name and
subdirectory by default: ``forward``.  This would not work because then all
the steps would end up in the same place, so the name is changed to something
unique.  In this example, the steps are given rather clumsy
names---``rpe_test_1_nu_1``, ``rpe_test_2_nu_5``, etc.---but these could be any
unique names.

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


In general, ``configure()`` is not the right place for adding or altering
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
simulation) but which are available for a user to manually alter and then run
on their own.

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
called (to determine which steps depend on which other steps).

Because of this relationship, there can be some cases to avoid.  The name of
an output file should not depend on a config option.  Otherwise, if the user
changes the config option, the file actually created may have a different name
than expected, in which case the step will fail.  This would be true even if
a subsequent step would have been able to read in the same config option and
modify the name of the expected input file.

Along the same lines, an input or output file name should not depend on data
from an input file that does not exist during :ref:`dev_step_setup`.  Since the
file does not exist, there is no way to read the file within
:ref:`dev_step_setup` and determine the file name.

Both of these issues have arisen for the
:ref:`dev_ocean_global_ocean_files_for_e3sm` test case from the
:ref:`dev_ocean_global_ocean` configuration.  Output files are named using the
"sort name" of the mesh in E3SM, which depends both on config options and on
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
``ocean/baroclinic_channel/10km/default/initial_state`` step at the beginning of
:py:func:`compass.ocean.tests.baroclinic_channel.initial_state.run()`:

.. code-block:: python


    step = {'module': 'compass.ocean.tests.baroclinic_channel.initial_state',
            'name': 'initial_state',
            'subdir': 'initial_state',
            'inputs': [],
            'outputs': ['/home/xylar/data/mpas/test_baroclinic_channel/ocean/baroclinic_channel/10km/default/initial_state/base_mesh.nc',
                        '/home/xylar/data/mpas/test_baroclinic_channel/ocean/baroclinic_channel/10km/default/initial_state/culled_mesh.nc',
                        '/home/xylar/data/mpas/test_baroclinic_channel/ocean/baroclinic_channel/10km/default/initial_state/culled_graph.info',
                        '/home/xylar/data/mpas/test_baroclinic_channel/ocean/baroclinic_channel/10km/default/initial_state/ocean.nc'],
            'cores': 1,
            'min_cores': 1,
            'max_memory': 8000,
            'max_disk': 8000,
            'setup': 'setup',
            'run': 'run',
            'path': 'ocean/baroclinic_channel/10km/default/initial_state',
            'testcase': 'default',
            'testcase_subdir': '10km/default',
            'work_dir': '/home/xylar/data/mpas/test_baroclinic_channel/ocean/baroclinic_channel/10km/default/initial_state',
            'base_work_dir': '/home/xylar/data/mpas/test_baroclinic_channel/',
            'config': 'default.cfg',
            'resolution': '10km'}

``module``
    The full name of the module where the step case is defined. This entry
    should be defined by passing ``module=__name__`` as an argument
    to :py:func:`compass.testcase.get_step_default()` in
    :ref:`dev_step_collect`.

``name``
    The name of the step, by default the last part of the ``module`` and
    is set by :py:func:`compass.testcase.get_step_default()`.  You can
    modify this entry in :ref:`dev_step_collect` anytime after calling
    ``get_step_default()``.

``subdir``
    The subdirectory for the step within the test case.  The default is the the
    last part of the ``module`` and is set by
    :py:func:`compass.testcase.get_step_default()`.  You can modify this
    entry in :ref:`dev_step_collect` anytime after calling
    ``get_step_default()``.  The value of ``subdir`` is nearly always the
    same as for ``name``.

``inputs``
    A list of absolute paths of input files to the step, see
    :ref:`dev_step_inputs_outputs`, that should be defined in
    :ref:`dev_step_setup` (or, less commonly, in :ref:`dev_step_collect`).

``outputs``
    A list of absolute paths of outputs files from the step, see
    :ref:`dev_step_inputs_outputs`, that should be defined in
    :ref:`dev_step_setup` (or, less commonly, in :ref:`dev_step_collect`).

``cores``
    The "target" number of cores that the step would ideally run on if that
    number is available.  This entry should be set in :ref:`dev_step_collect`
    or  :ref:`dev_step_setup` if it is known in advance, or in the test case's
    :ref:`dev_testcase_run` if it comes from a config option that a user might
    alter.

``max_memory``
    The maximum amount of memory the step is allowed to use.  This is a
    placeholder for the time being and is not used. This entry should be set in
    :ref:`dev_step_collect` or  :ref:`dev_step_setup` if it is known in
    advance, or in the test case's :ref:`dev_testcase_run` if it comes from a
    config option that a user might alter.

``max_disk``
    The maximum amount of disk space the step is allowed to use.  This is a
    placeholder for the time being and is not used. This entry should be set in
    :ref:`dev_step_collect` or  :ref:`dev_step_setup` if it is known in
    advance, or in the test case's :ref:`dev_testcase_run` if it comes from a
    config option that a user might alter.

``setup``
    The name of the :ref:`dev_step_setup` function for setting up the step.
    This entry is added by :py:func:`compass.testcase.get_testcase_default()`
    and should only be modified if you have an important reason not to name
    the function in your step's module ``configure``.

``run``
    The name of the :ref:`dev_step_run` function for running the step, set
    by :py:func:`compass.testcase.get_testcase_default()`.  This entry should
    only be modified if you have an important reason not to name the function
    in your step's module ``run``.

``path``
    The relative path of the step within the base work directory, the
    combination of the ``core``, ``configuration``, test case's ``subdir``
    and the step's ``subdir``.  This entry is added automatically by the
    :ref:`dev_framework` after :ref:`dev_testcase_collect` is called and should
    not be modified.

``testcase``
    The name of the test case that this step belongs to.  This is set by the
    :ref:`dev_framework` and should not be modified.

``testcase_subdir``
    The subdirectory of the test case that this step belongs to.  This is set
    by the :ref:`dev_framework` and should not be modified.  It can be useful
    for finding the paths to other steps in the test case.

``work_dir``
    The absolute path where the step will be or has been set up.  This is set
    by the :ref:`dev_framework` before calling :ref:`dev_step_setup` and should
    not be modified.  It can be helpful for determining absolute paths for
    input and output files.

``base_work_dir``
    The absolute path to the base location where test cases are set up.  This
    is set by the :ref:`dev_framework` before calling :ref:`dev_step_setup` and
    should not be modified.  It can be helpful for determining absolute paths
    for input and output files.

``config``
    The name of the config file where config options will read in for the
    step.  This config file is shared across the test case.  This entry
    is set by the :ref:`dev_framework` before calling :ref:`dev_step_setup` and
    should not be modified.  It is used internally by the :ref:`dev_framework`
    and likely won't be useful within the step because config options are
    available in the ``config`` object.

You can add other entries to the dictionary to pass information between the
:ref:`dev_step_collect`, :ref:`dev_step_setup`, and :ref:`dev_step_run`.  In the
example above, ``resolution`` has been added for this purpose.

.. _dev_step_collect:

collect()
^^^^^^^^^

The ``collect()`` function for a step must:

1. call :py:func:`compass.testcase.get_step_default()`

3. return the resulting python dictionary ``step``.

You can include argument (typically parameters) to ``collect()`` as long as
the test case's ``collect()`` function will know what these should be.  In
the example below, there are lots of arguments.  Some, like ``resolution``,
are required while others, like the viscosity ``nu`` are not.

Typically, ``collect()`` will do little more than call
:py:func:`compass.testcase.get_step_default()` and add the parameters to
``step`` for use in :ref:`dev_step_setup` and :ref:`dev_step_run`.

The following is the contents of
:py:func:`compass.ocean.tests.baroclinic_channel.forward.collect()`, which is an
example of a ``collect()`` function with a large number of arguments:

.. code-block:: python

    from compass.testcase import get_step_default


    def collect(resolution, cores, min_cores=None, max_memory=1000,
                max_disk=1000, threads=1, testcase_module=None,
                namelist_file=None, streams_file=None, nu=None):
        """
        Get a dictionary of step properties

        Parameters
        ----------
        resolution : {'1km', '4km', '10km'}
            The name of the resolution to run at

        cores : int
            The number of cores to run on in forward runs. If this many cores are
            available on the machine or batch job, the task will run on that
            number. If fewer are available (but no fewer than min_cores), the job
            will run on all available cores instead.

        min_cores : int, optional
            The minimum allowed cores.  If that number of cores are not available
            on the machine or in the batch job, the run will fail.  By default,
            ``min_cores = cores``

        max_memory : int, optional
            The maximum amount of memory (in MB) this step is allowed to use

        max_disk : int, optional
            The maximum amount of disk space  (in MB) this step is allowed to use

        threads : int, optional
            The number of threads to run with during forward runs

        testcase_module : str, optional
            The module for the testcase

        namelist_file : str, optional
            The name of a namelist file in the testcase package directory

        streams_file : str, optional
            The name of a streams file in the testcase package directory

        nu : float, optional
            The viscosity for this step

        Returns
        -------
        step : dict
            A dictionary of properties of this step
        """
        step = get_step_default(__name__)
        step['resolution'] = resolution
        step['cores'] = cores
        step['max_memory'] = max_memory
        step['max_disk'] = max_disk
        if min_cores is None:
            min_cores = cores
        step['min_cores'] = min_cores
        step['threads'] = threads
        if testcase_module is not None:
            step['testcase_module'] = testcase_module
        else:
            if namelist_file is not None or streams_file is not None:
                raise ValueError('You must supply a testcase module for the '
                                 'namelist and/or streams file')
        if namelist_file is not None:
            step['namelist'] = namelist_file
        if streams_file is not None:
            step['streams'] = streams_file

        if nu is not None:
            step['nu'] = nu

        return step

Below, we will follow how these parameters are use later in the step.

.. _dev_step_setup:

setup()
^^^^^^^

The ``setup()`` function is called when a user is setting up each step either
as part of a call to :ref:`dev_compass_setup` or :ref:`dev_compass_suite`.
Typical activities that are involved in setting up a step include:

1. downloading files, most often from the
   `LCRC server <https://web.lcrc.anl.gov/public/e3sm/mpas_standalonedata/>`_.

2. making symlinks to files from ``compass``, in a local cache directory, or
   other steps.

3. creating a list of :ref:`dev_step_inputs_outputs`.

4. adding the contents of one or more :ref:`config_files` to the ``config``
   object, or using ``config.set()`` to set config options directly.

5. modifying namelist options from their defaults either by parsing namelist
   files or by directly adding entries to a ``replacements`` dictionary, then
   generating the namelist file that will be used by the MPAS model.

6. adding or modifying streams by parsing stream files to create an XML tree,
   then generating the streams file that will be used by the MPAS model.

Set up should not do any major computations or any time-consuming operations
other than downloading files.

As an example, here is a slightly modified version of
:py:func:`compass.ocean.tests.baroclinic_channel.forward.setup()`:

.. code-block:: python

    import os

    from compass.io import symlink
    from compass import namelist, streams


    def setup(step, config):
        """
        Set up the test case in the work directory, including downloading any
        dependencies

        Parameters
        ----------
        step : dict
            A dictionary of properties of this step from the ``collect()`` function

        config : configparser.ConfigParser
            Configuration options for this testcase, a combination of the defaults
            for the machine, core, configuration and testcase
        """
        resolution = step['resolution']
        step_dir = step['work_dir']
        if 'testcase_module' in step:
            testcase_module = step['testcase_module']
        else:
            testcase_module = None

        # generate the namelist, replacing a few default options
        replacements = dict()

        for namelist_file in ['namelist.forward',
                              'namelist.{}.forward'.format(resolution)]:
            new_replacements = namelist.parse_replacements(
                'compass.ocean.tests.baroclinic_channel', namelist_file)
            replacements.update(new_replacements)

        # see if there's one for the testcase itself
        if 'namelist' in step:
            new_replacements = namelist.parse_replacements(
                testcase_module, step['namelist'])
            replacements.update(new_replacements)

        if 'nu' in step:
            # update the viscosity to the requested value
            replacements['config_mom_del2] = '{}'.format(step['nu'])

        namelist.generate(config=config, replacements=replacements,
                          step_work_dir=step_dir, core='ocean', mode='forward')

        # generate the streams file
        streams_data = streams.read('compass.ocean.tests.baroclinic_channel',
                                    'streams.forward')

        # see if there's one for the testcase itself
        if 'streams' in step:
            streams_data = streams.read(testcase_module, step['streams'],
                                        tree=streams_data)

        streams.generate(config=config, tree=streams_data, step_work_dir=step_dir,
                         core='ocean', mode='forward')

        # make a link to the ocean_model executable
        symlink(os.path.abspath(config.get('executables', 'model')),
                os.path.join(step_dir, 'ocean_model'))

        inputs = []
        outputs = []

        links = {'../initial_state/ocean.nc': 'init.nc',
                 '../initial_state/culled_graph.info': 'graph.info'}
        for target, link in links.items():
            symlink(target, os.path.join(step_dir, link))
            inputs.append(os.path.abspath(os.path.join(step_dir, target)))

        for file in ['output.nc']:
            outputs.append(os.path.join(step_dir, file))

        step['inputs'] = inputs
        step['outputs'] = outputs

Let's go back through this a few bits at a time.

.. code-block:: python

        resolution = step['resolution']
        step_dir = step['work_dir']
        if 'testcase_module' in step:
            testcase_module = step['testcase_module']
        else:
            testcase_module = None

This just pulls the resolution, the step's work directory and (if available)
the test case's module out of the ``step`` dictionary for convenience.

.. _dev_step_namelists:

namelists
~~~~~~~~~

The next segment adds a bunch of "replacements" to a dictionary that is used to
update the default namelist from the MPAS model.

First, we create an emtpy dictionary
.. code-block:: python

        # generate the namelist, replacing a few default options
        replacements = dict()

Then, we add replacements from 2 files that are part of the
``compass.ocean.tests.baroclinic`` package.  Let's say we're setting up the
10-km version of the test case (``resolution = 10km``).  Then, the files are
``namelist.forward`` and ``namelist.10km.forward``.  The first one contains
namelist options that are appropriate for all ``baroclinic_channel`` test
cases and the second has some config options (like the time step) that are
specific to the resolution.

.. code-block:: python

        for namelist_file in ['namelist.forward',
                              'namelist.{}.forward'.format(resolution)]:
            new_replacements = namelist.parse_replacements(
                'compass.ocean.tests.baroclinic_channel', namelist_file)
            replacements.update(new_replacements)

First, we parse the new replacements from the file using
:py:func:`compass.namelist.parse_replacements()`, then we add them to our
``replacements`` dictionary (updating anything that was already in there with
the new value).

Here's what these two files look like:

.. code-block:: none

    config_write_output_on_startup = .false.
    config_run_duration = '0000_00:15:00'
    config_use_mom_del2 = .true.
    config_implicit_bottom_drag_coeff = 1.0e-2
    config_use_cvmix_background = .true.
    config_cvmix_background_diffusion = 0.0
    config_cvmix_background_viscosity = 1.0e-4

.. code-block:: none

    config_dt = '00:05:00'
    config_btr_dt = '00:00:15'
    config_mom_del2 = 10.0

Some ``baroclinic_channel`` test cases have their own namelist options as well.
These test cases will pass a namelist file and the module (or package actually
if we're being picky) for the test case as arguments to ``collect()``. If they
were included, we added them to ``step``.  So if ``namelist`` is found in the
step dictionary, we add its replacements, too:

.. code-block:: python

        # see if there's one for the testcase itself
        if 'namelist' in step:
            new_replacements = namelist.parse_replacements(
                testcase_module, step['namelist'])
            replacements.update(new_replacements)

For the ``rpe_test``, there is such a ``namelist.forward``:

.. code-block:: none

    config_run_duration = '20_00:00:00'

It changes the duration of the run from the ``baroclinic_channel`` default of
15 minutes to 20 days (yikes!).

The ``rpe_test`` test case also passes a value for the viscosity ``nu`` to
``collect()`` so we can do a parameter study with 5 different values.  If
``nu`` was passed to ``collect()``, it gets added to ``step`` and we now use it
to update the appropriate namelist option, ``config_mom_del2``:

.. code-block:: python

        if 'nu' in step:
            # update the viscosity to the requested value
            replacements['config_mom_del2] = '{}'.format(step['nu'])

Okay, now we're ready to generate a namelist file with
:py:func:`compass.namelist.generate()` by starting with the defaults for
``forward`` mode in the ``ocean`` core and substituting our replacements:

.. code-block:: python

        namelist.generate(config=config, replacements=replacements,
                          step_work_dir=step_dir, core='ocean', mode='forward')

.. _dev_step_streams:

streams
~~~~~~~

The next section is the same concept but for streams files.  Here, things get
a little more complicated because streams files are XML documents, requiring
some slightly more sophisticated parsing.

It's not as easy to start with an empty XML tree and merging XML trees isn't
quite as simple as calling their ``.update()`` method like for the replacements
dictionary we used for namelists above.  Instead, we just parse the first
streams file with :py:func:`compass.streams.read()` but without passing the
optional ``tree`` argument.  This tells the function to make a fresh XML tree
with the streams from this file:

.. code-block:: python

        # generate the streams file
        streams_data = streams.read('compass.ocean.tests.baroclinic_channel',
                                    'streams.forward')

Here's what the streams file looks like:

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
replace the default contents.  Currently, there is no way to add or remove
contents from the defaults, just keep the default contents or replace them all.

Just like for the ``namelist`` passed in from the test case, we might have a
``streams`` file passed in from the test case.  If so, we parse it as well,
this time updating ``streams_data`` by passing it in as the ``tree`` argument
to :py:func:`compass.streams.read()`:

.. code-block:: python

        # see if there's one for the testcase itself
        if 'streams' in step:
            streams_data = streams.read(testcase_module, step['streams'],
                                        tree=streams_data)

For the ``rpe_test`` test case, ``streams.forward`` is passed in and it looks
like this:

.. code-block:: xml

    <streams>

    <stream name="output"
            type="output"
            filename_template="output.nc"
            output_interval="0000-00-20_00:00:00"
            clobber_mode="truncate">

        <var_struct name="tracers"/>
        <var name="xtime"/>
        <var name="density"/>
        <var name="daysSinceStartOfSim"/>
        <var name="relativeVorticity"/>
    </stream>

    </streams>

The only stream to update ith ``output``.  Many of its attributes are updated
unnecessarily to the same values as before, but the ``output_interval`` is
increased to 20 days (the duration of the run) and the output variables are
changed.  The variables ``normalVelocity`` and ``layerThickness`` are included
in the forward runs of most ``baroclinic_channel`` test cases but they will be
dropped in the ``rpe_test`` because they are not included when the ``output``
stream gets updated.

Okay, now we're ready to generate a streams file with
:py:func:`compass.streams.generate()` by starting with the defaults for
``forward`` mode for any streams in the ``ocean`` core and updating it with
our ``streams_data``:

.. code-block:: python

        streams.generate(config=config, tree=streams_data, step_work_dir=step_dir,
                         core='ocean', mode='forward')

Any streams that are not included in ``streams_data`` will be dropped.  Any
streams where only attributes were modified will get the contents from the
default stream and the attributes will first come from the defaults and then
be replaced by values from ``streams_data`` where they were provided.


.. _dev_step_downloads:

downloads, symlinks, inputs and outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The particular example we're using in this section doesn't download any input
files.  Here's a snippet from
:py:func:`compass.ocean.tests.global_ocean.init.initial_state.setup()` that
does:

.. code-block:: python

    from compass.io import symlink, download

    def setup(step, config):

        bathymetry_database = config.get('paths', 'bathymetry_database')

        inputs = []

        remote_filename = \
            'BedMachineAntarctica_and_GEBCO_2019_0.05_degree.200128.nc'
        local_filename = 'topography.nc'

        filename = download(
            file_name=remote_filename,
            url='https://web.lcrc.anl.gov/public/e3sm/mpas_standalonedata/'
                'mpas-ocean/bathymetry_database',
            config=config, dest_path=bathymetry_database)

        inputs.append(filename)
        symlink(filename, os.path.join(step_dir, local_filename))

In this example, the remote file
`BedMachineAntarctica_and_GEBCO_2019_0.05_degree.200128.nc <https://web.lcrc.anl.gov/public/e3sm/mpas_standalonedata/mpas-ocean/bathymetry_databaseBedMachineAntarctica_and_GEBCO_2019_0.05_degree.200128.nc>`_
gets downloaded into the bathymetry database (if it's not already there).  We
take advantage of the fact that :py:func:`compass.io.download()` returns the
full path to the file, and append this file to the list of inputs to the step.
(This input didn't come from another step, so it's probably overkill but it
does no harm, see :ref:`dev_step_inputs_outputs`).  Finally, we create a local
symlink called ``topography.nc`` to the file in the bathymetry database.

Returning to our previous example,
:py:func:`compass.ocean.tests.baroclinic_channel.forward.setup()`, we make a
symlink to the ``ocean_model`` executable (which we locate using the ``model``
config option from the ``executables`` section).

.. code-block:: python

        # make a link to the ocean_model executable
        symlink(os.path.abspath(config.get('executables', 'model')),
                os.path.join(step_dir, 'ocean_model'))

Then, we make symlinks to two files, ``ocean.nc`` and ``culled_graph.info``
from the ``initial_state`` step of the same test case (giving them new names
here).  At the same time, we also add the absolute paths to these files to the
``inputs`` list, which means this step will fail to run if those files don't
exist.

.. code-block:: python

        inputs = []
        outputs = []

        links = {'../initial_state/ocean.nc': 'init.nc',
                 '../initial_state/culled_graph.info': 'graph.info'}
        for target, link in links.items():
            symlink(target, os.path.join(step_dir, link))
            inputs.append(os.path.abspath(os.path.join(step_dir, target)))

Then, we add the absolute path to the one significant output file from this
step, ``output.nc``, to the list of output files:

.. code-block:: python


        for file in ['output.nc']:
            outputs.append(os.path.join(step_dir, file))

Finally, we add the inputs and outputs as entries the step dictionary.

.. code-block:: python


        step['inputs'] = inputs
        step['outputs'] = outputs

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
:py:func:`compass.ocean.tests.baroclinic_channel.forward.run()` looks like this:

.. code-block:: python

    from compass.model import partition, run_model
    from compass.parallel import update_namelist_pio


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
        cores = step['cores']
        threads = step['threads']
        step_dir = step['work_dir']
        update_namelist_pio(config, cores, step_dir)
        partition(cores, logger)
        run_model(config, core='ocean', core_count=cores, logger=logger,
                  threads=threads)

We just get the number of cores, threads and the work directory for the step
from the ``step`` dictionary.  Then, we run
:py:func:`compass.parallel.update_namelist_pio()` with the number of cores we
want so it can update the ``namelist.ocean`` file to have one PIO task per node
(a configuration we have found to work pretty reliably).

Then, a call to :py:func:`compass.model.partition()` creates a graph partition
for the requested number of cores using
`gpmetis <http://glaros.dtc.umn.edu/gkhome/metis/metis/overview>`_.

Finally, MPAS-Ocean is run with the requested number of cores and threads using
:py:func:`compass.model.run_model()`.

This is a common approach to forward runs.

To get a feel for different types of ``run()`` functions, it may be best to
explore different test cases.

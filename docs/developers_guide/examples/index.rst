.. _dev_examples:

Examples core
=============

The ``examples`` core is by no means a fully developed compass core.  It
contains two example configurations, each with 4 test cases with 2 steps per
test case.  The test cases and steps are, themselves, trivial---they download
a couple of small files (if they aren't already in the local cache), read them
in and write them back out.  Both examples are identical in what they do but
quite different in how they are written.

The ``example_expanded`` configuration is intended to show how a configuration
*could* be written in a very verbose way with the code for each test case
and step completely independent of all the others.  We emphasize from the start
that this is *absolutely not* the way that we recommend writing test cases. It
is tedious to update as changes are made to the API for test cases or steps, or
as bugs are discovered.  It also makes future work much harder if you or
another developer wants to modify or expand the code.  But it is meant to be
simple and easy to follow.

The ``example_compact`` configuration is meant to show how the same test cases
can be written in a more compact form with more :ref:`dev_code_sharing`.  There
aren't separate packages for test cases at different resolution or separate
modules for the 2 steps within each test case.

Configurations
--------------

.. _dev_examples_example_expanded:

example_expanded
^^^^^^^^^^^^^^^^

The code for ``example_expanded`` contains the shared config options in
``example_expanded.cfg``:

.. code-block:: cfg

    # default namelist options for the "example_expanded" configuration
    [example_expanded]

    # A parameter that we will use in setting up or running the test case
    parameter1 = 0.

    # Another parameter
    parameter2 = False

There is a section with the name of the configuration and some parameters with
their default values.

The configuration also contains packages for the 2 supported resolutions:
``res1km`` and ``res2km``.  (Python packages aren't allowed to start with
numbers so we add a ``res`` prefix.)  Each resolution has packages for the 2
test cases---``test1`` and ``test2``---and each of these have 2
steps---``step1.py`` and ``step2.py``---along with a config file (e.g.
``test1.cfg``) that overrides the config options:

.. code-block:: cfg

    # default namelist options for the "example_expanded" configuration
    [example_expanded]

    # A parameter that we will use in setting up or running the test case
    parameter1 = 1.

    # Another parameter
    parameter2 = False

These test cases are pretty well commented so we won't go through the code in
detail, but will cover the basics.

Each test case is added in
:py:func:`compass.examples.tests.example_expanded.collect()` using the
framework function :py:func:`compass.testcase.add_testcase()`.  No extra
keyword arguments are passed to ``add_testcase()`` because each test case
simply has its parameters hard-coded (in contrast to
:ref:`dev_examples_example_compact`).  Each test case has the required
:ref:`dev_testcase_collect` and :ref:`dev_testcase_run` functions, and the
optional :ref:`dev_testcase_configure`.  We add the parameters to the
``testcase`` python dictionary and give the test case a description,
use :py:func:`compass.testcase.set_testcase_subdir()` to give the test cases
a unique subdirectory that includes the resolution, and call
:py:func:`compass.testcase.add_step()` for each step, passing the resolution
as a keyword parameter that will be added to the ``step`` dictionary.

In ``configure()``, we add config options from the local config file and then
add the resolution to the config file. (You will have to decide if this makes
sense for your test case---should the user be able to change this parameter or
should it remain fixed for this test case?).

In ``run()``, we simply call :py:func:`compass.testcase.run_steps` to run
each of the steps in the test case.

Each step has the required :ref:`dev_step_collect` and :ref:`dev_step_run`
functions.  ``step1`` in each test case also has the optional
:ref:`dev_step_setup` function, but ``step2`` does not need it.  The
``collect()`` function sets several parameters (``cores``, ``min_cores``,
``max_memory``, ``max_disk``, and ``threads``), adds an input file using
:py:func:`compass.io.add_input_file()`, and adds an output file using
:py:func:`compass.io.add_output_file()`.  In ``step1``, we indicate that the
input file should be downloaded from the initial-condition database, while in
``step2``, the input file points to the output from ``step1``.

In ``step1``, ``setup()`` adds some parameters to the ``step`` dictionary.

In the ``run()`` function for ``step1``, some parameters are retrieved from the
``step`` dictionary, then the input file is read in and immediately written out
to the output file.

``step2`` is even simpler.  It has no ``setup()`` and in ``run()`` it reads in
the input file and writes the results to the output file.

Obviously, these are trivial and rather dull examples.

.. _dev_examples_example_compact:

example_compact
^^^^^^^^^^^^^^^

The main purpose of this example is to show how to do
:ref:`dev_examples_example_expanded` "right", with :ref:`dev_code_sharing`.

Instead of packages for each resolution, these become keyword arguments to
:py:func:`compass.testcase.add_testcase()` in
:py:func:`compass.examples.tests.example_compact.collect()` that are
automatically added to the ``testcase`` dictionary by the :ref:`dev_framework`.
Similarly, parameters are passed on to the steps in each test case's
:ref:`dev_testcase_collect` function by passing them as keyword argument to
:py:func:`compass.testcase.add_step()`.
There is only one package for each test case (``test1`` and ``test2``), and
they both share the modules ``step1.py`` and ``step2.py`` (there are 4 copies
of each in :ref:`dev_examples_example_expanded`).

Rather than going through each test case and step, we will focus on the
differences compared with :ref:`dev_examples_example_expanded`.

Each test case's :ref:`dev_testcase_collect` now retrieves the resolution from
the ``testcase`` dictionary.  The resolution is then used in the description
of the test case and is passed to the steps via
:py:func:`compass.testcase.add_step()`:

.. code-block:: python

    def collect(testcase):
        """
        Update the dictionary of test case properties and add steps

        Parameters
        ----------
        testcase : dict
            A dictionary of properties of this test case, which can be updated
        """
        # you can get any information out of the "testcase" dictionary, e.g. to
        # pass them on to steps.  Some of the entries will be from the framework
        # while others are passed in as keyword arguments to "add_testcase" in the
        # configuration's "collect()"
        resolution = testcase['resolution']

        # you must add a description
        testcase['description'] = 'Template {} test1'.format(resolution)

        # You can change the subdirectory from the default, the name of the test
        # case.  In this case, we add a directory for the resolution.
        subdir = '{}/{}'.format(resolution, testcase['name'])
        set_testcase_subdir(testcase, subdir)

        # we can pass keyword argument to the step so they get added to the "step"
        # dictionary and can be used throughout the step
        add_step(testcase, step1, resolution=resolution)
        add_step(testcase, step2, resolution=resolution)

The other functions in the test case are not changed from
:ref:`dev_examples_example_expanded`.

The steps now retrieve the resolution from the ``step`` dictionary:

.. code-block:: python

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
        # the "testcase" and "step" dictionaries will contain some information that
        # is either added by the framework or passed in to "add_step" as a keyword
        # argument.  In this case, we get the name of the test case that was added
        # by the framework and the resolution, which was passed as a keyword
        # argument to "add_step".
        testcase_name = testcase['name']
        resolution = step['resolution']
        ...

the input file for ``step1`` depends on the test case.  This is handled by
having a dictionary that can be used to select the appropriate input file
based on the name of the test case:

.. code-block:: python

        ...
        targets = {'test1': 'particle_regions.151113.nc',
                   'test2': 'layer_depth.80Layer.180619.nc'}

        if testcase_name not in targets:
            raise ValueError('Unsupported test case name {}. Supported test cases '
                             'are: {}'.format(testcase, list(targets)))
        target = targets[testcase_name]

        ...
        add_input_file(step, filename='input_file.nc', target=target,
                       database='initial_condition_database')

The appropriate file for the test case is then added as an input file that
should be downloaded from the initial-condition database.

Similarly, in ``setup``, there are several parameters that differ depending on
which resolution the step is run with.  These are handled with a nested
dictionary of possible parameters and selecting which parameters
are appropriate for the given resolution:


.. code-block:: python

    def setup(step, config):
        """
        Set up the test case in the work directory, including downloading any
        dependencies

        Parameters
        ----------
        step : dict
            A dictionary of properties of this step

        config : configparser.ConfigParser
            Configuration options for this step, a combination of the defaults for
            the machine, core, configuration and test case
        """
        resolution = step['resolution']
        # This is a way to handle a few parameters that are specific to different
        # test cases or resolutions, all of which can be handled by this function
        res_params = {'1km': {'parameter4': 1.0,
                              'parameter5': 500},
                      '2km': {'parameter4': 2.0,
                              'parameter5': 250}}

        # copy the appropriate parameters into the step dict for use in run
        if resolution not in res_params:
            raise ValueError('Unsupported resolution {}. Supported values are: '
                             '{}'.format(resolution, list(res_params)))
        res_params = res_params[resolution]

        # add the parameters for this resolution to the step dictionary so they
        # are available to the run() function
        for param in res_params:
            step[param] = res_params[param]

We get the resolution from ``step``, then we select the appropriate dictionary
from the nested dictionary ``res_params`` for our resolution,
``res_params = res_params[resolution]``, and we add these parameters
to ``step``.

The ``run()`` function is unchanged, as is ``run()`` functions for ``step2``.

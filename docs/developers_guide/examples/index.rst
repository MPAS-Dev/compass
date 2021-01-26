.. _dev_examples:

Examples core
=============

The ``examples`` core is, by no means, a fully developed compass core.  It
contains two example configurations, each with 4 test cases with 2 steps per
test case.  The test cases and steps are, themselves, trivial---they download
a couple of small files (if they aren't already in the local cache), read them
in and write them back out.  Both examples are identical in what they do but
quite different in how they are written.

The ``example_expanded`` configuration is intended to show how a configuration
*could* be written in a very verbose way way with the code for each test case
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

The code for `example_expanded` contains the shared config options in
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

Each test case has the required :ref:`dev_testcase_collect`,
:ref:`dev_testcase_configure`, and :ref:`dev_testcase_run` functions.  In
these examples, the ``collect()`` function doesn't take any arguments because
we will just hard-code the parameters (in contrast to
:ref:`dev_examples_example_compact`).  We given the test case a description,
call :ref:`dev_step_collect` on each step, collect the steps into a dictionary,
call :py:func:`compass.testcase.get_testcase_default()`, add the resolution
as a parameter, and return the ``testcase`` dictionary.

In ``configure()``, we add config options from the local config file and then
add the resolution to the config file. (You will have to decide if this makes
sense for your test case---should the user be able to change this parameter or
should it remain fixed for this test case?).

In ``run()``, we simply call :py:func:`compass.testcase run_steps` to run
each of the steps in the test case.

Each step has the required :ref:`dev_step_collect`, :ref:`dev_step_setup`, and
:ref:`dev_step_run` functions. The ``collect()`` function calls
:py:func:`compass.testcase.get_step_default()`, as it must, and then adds the
resolution (again hard-coded) to the ``step`` dictionary and retruns it.

In ``step1``, ``setup()`` retrieves some parameters from the ``step``
dictionary and adds others to it.  Then, it downloads a file to the initial
condition database (if it's not already there) and adds that file to the list
of inputs.  Which file is downloaded depends on the test case, but all files
are hard-coded.  Finally, an output file is added to the list of outputs, and
the lists of inputs and outputs are added to the ``step`` dictionary.

In the ``run()`` function for ``step1``, some parameters are again retrieved
from the ``step`` dictionary, then the input file is read in from the input
file and and immediately written out to the output file.

``step2`` is even simpler.  In ``setup()``, it creates a symlink to the
output file from ``step1`` and adds that file to its list of inputs.  It, too,
adds an output file to its list of outputs and adds the inputs and outputs to
the ``step`` dictionary.  Then, in ``run()`` it reads in the input file and
writes the results to the output file.

Obviously, these are trivial and rather dull examples.

.. _dev_examples_example_compact:

example_compact
^^^^^^^^^^^^^^^

The main purpose of this example is to show how to do
:ref:`dev_examples_example_expanded` "right", with :ref:`dev_code_sharing`.

Instead of packages for each resolution, these become parameters to the
test case's :ref:`dev_testcase_collect` and the step's :ref:`dev_step_collect`.
There is only one package for each test case (``test1`` and ``test2``), and
they both share the modules ``step1.py`` and ``step2.py`` (there are 4 copies
of each in :ref:`dev_examples_example_expanded`).

Rather than going through each test case and step, we will focus on the
differences compared with :ref:`dev_examples_example_expanded`.

Each test case's :ref:`dev_testcase_collect` now takes resolution as an
argument.  The resolution is then used in the description of the test case:

.. code-block:: python

    def collect(resolution):
        """
        Get a dictionary of testcase properties

        Parameters
        ----------
        resolution : {'1km', '2km'}
            The resolution of the mesh

        Returns
        -------
        testcase : dict
            A dict of properties of this test case, including its steps
        """
        # fill in a useful description of the test case
        description = 'Template {} test1'.format(resolution)
        ...

Otherwise, the test case is not changed from
:ref:`dev_examples_example_expanded`.

The steps now also take the resolution as a an argument:

.. code-block:: python

    def collect(resolution):
        """
        Get a dictionary of step properties

        Parameters
        ----------
        resolution : {'1km', '2km'}
            The name of the resolution to run at

        Returns
        -------
        step : dict
            A dictionary of properties of this step
        """
        ...

``step1`` had several parameters that differ depending on which test case and
resolution it is run with.  These are handled in
:py:func:`compass.examples.tests.example_compact.step1.setup()` by having
nested dictionaries of of possible parameters and selecting which parameters
are appropriate for a given test case or resolution:

.. code-block:: python

    def setup(step, config):
        """
        Set up the test case in the work directory, including downloading any
        dependencies

        Parameters
        ----------
        step : dict
            A dictionary of properties of this step from the ``collect()`` function

        config : configparser.ConfigParser
            Configuration options for this step, a combination of the defaults for
            the machine, core, configuration and testcase
        """
        resolution = step['resolution']
        testcase = step['testcase']
        # This is a way to handle a few parameters that are specific to different
        # testcases or resolutions, all of which can be handled by this function
        res_params = {'1km': {'parameter4': 1.0,
                              'parameter5': 500},
                      '2km': {'parameter4': 2.0,
                              'parameter5': 250}}

        test_params = {'test1': {'filename': 'particle_regions.151113.nc'},
                       'test2': {'filename': 'layer_depth.80Layer.180619.nc'}}

        # copy the appropriate parameters into the step dict for use in run
        if resolution not in res_params:
            raise ValueError('Unsupported resolution {}. Supported values are: '
                             '{}'.format(resolution, list(res_params)))
        res_params = res_params[resolution]

        # add the parameters for this resolution to the step dictionary so they
        # are available to the run() function
        for param in res_params:
            step[param] = res_params[param]

        if testcase not in test_params:
            raise ValueError('Unsupported testcase name {}. Supported testcases '
                             'are: {}'.format(testcase, list(test_params)))
        test_params = test_params[testcase]

        # add the parameters for this testcase to the step dictionary so they
        # are available to the run() function
        for param in test_params:
            step[param] = test_params[param]

We get the resolution and test case name from ``step``, then we select the
appropriate dictionary from the nested dictionary ``res_params`` for our
resolution, ``res_params = res_params[resolution]``, and we add these parameters
to ``step``.  Then, we do the same for the parameters associated with the
test case (the name of the input file to download).

From there, ``setup()`` proceeds as it would in
:ref:`dev_examples_example_expanded`.  The ``run()`` function is unchanged,
as are the ``setup()`` and ``run()`` functions for ``step2``.

.. _dev_overview:

Overview
========

``compass`` is a `python package <https://docs.python.org/3/tutorial/modules.html#packages>`_.
All of the code in the package can be accessed in one of two ways.  The first
is the command-line interface with commands like :ref:`dev_compass_list` and
:ref:`dev_compass_setup`.  The second way is through import commands like:

.. code-block:: python

    from compass.io import symlink


    symlink('../initial_condition/initial_condition.nc', 'init.nc')

Before we dig into the details of now to develop new test cases and other
infrastructure for ``compass``, we first give a little bit of background on
the design philosophy behind the package.

.. _dev_packages:

Packages and Modules
--------------------

Why a python package?  That sounds complicated.

Some of the main advantages of ``compass`` being a package instead of a group
of scripts (as was the case for :ref:`legacy_compass`) are that:

1) it is a lot easier to share code between test cases;

2) there is no need to create symlinks to individual scripts or use
   `subprocess <https://docs.python.org/3/library/subprocess.html>`_ calls to
   run one python script from within another;

3) functions within ``compass`` modules and subpackages have relatively simple
   interfaces that are easier to document and understand that the arguments
   passed in to a script; and

4) releases of the ``compass`` package will make it easy for developers of
   other python packages and scripts to use our code.

This documentation won't try to provide a whole tutorial on python packages and
modules but we know most developers won't be too clued in on these concepts so
here's a short intro.

Packages
~~~~~~~~

A python package is a directory that has a file called ``__init__.py``.  That
file can be empty or it can have code in it.  If it has functions inside of
it, those functions act like they're directly in the package.  As an example,
the compass file
`compass/testcase/__init__.py <https://github.com/MPAS-Dev/compass/tree/master/compass/testcase/__init__.py>`_
has a function :py:func:`compass.testcase.get_step_default()` that looks like
this:

.. code-block:: python

    def get_step_default(module):
        name = module.split('.')[-1]
        step = {'module': module,
                'name': name,
                'subdir': name,
                'setup': 'setup',
                'run': 'run',
                'inputs': [],
                'outputs': []}
        return step

The details aren't important.  The point is that the function can be imported
like so:

.. code-block:: python

    from compass.testcase import get_step_default


    step = get_step_default(__name__)

So you don't ever refer to ``__init__.py``, it's like a hidden shortcut so the
its contents can be referenced with just the subdirectory (package) name.

A package can contain other packages and modules (we'll discuss these in just
a second).  For example, the ``testcase`` package mentioned above is inside the
``compass`` package.  The sequence of dots in the import is how you find your
way from the root (``compass`` for this package) into subpackages and modules.
It's similar to the ``/`` characters in a unix directory.

Modules
~~~~~~~

Modules are just python files that aren't scripts.  Since you can often treat
scripts like modules, even that distinction isn't that exact.  But for the
purposes of the ``compass`` package, every single file ending in ``.py`` in the
``compass`` package is a module (except maybe the ``__init__.py``, not sure
about those...).

As an exmaple, the ``compass`` package contains a module ``testcases.py`` (a
little confusing, since there's a ``testcase`` package too, but trust us for
now that there's a good reason for this).  There's a function
:py:func:`compass.testcases.collect()` in that module:

.. code-block:: python

    def collect():

    testcase_list = list()

    for tests in [example_tests, ocean_tests]:
        testcase_list.extend(tests.collect())

    validate(testcase_list)

    ...

    return testcases

Don't worry about the details, the point is that you would import this function
just like in the package example above:

.. code-block:: python

    from compass.testcases import collect


    testcases = collect()

So a module named ``foo.py`` and a package in a directory named ``foo`` with
and ``__init__.py`` file look exactly the same when you import them.

So why choose one over the other?

The main reason to go with a package over a module is if you need to include
other files (such as other modules and packages, but also other things like
:ref:`config_files`, :ref:`dev_namelist`, or :ref:`dev_streams` files).  It's
always pretty easy to make a module into a package (by making a directory with
the name of the package, moving the module in, an renaming it ``__init__.py``)
or visa versa (by renaming ``__init__.py`` to the module name, moving it up
a directory, and deleting the subdirectory).

.. _dev_code_sharing:

Code sharing
------------

Very nearly all of the code in :ref:`legacy_compass` was in the form of python
scripts.  A significant amount of external code was also in this form.  A test
case was composed of XML files, and python scripts parsed these XML files to
produce other python scripts to run the test case.  These scripts were dense.
The XML files had a unique syntax that made the learning curve for
:ref:`legacy_compass` pretty high.  Errors in syntax were often hard to
understand because the script-generating scripts were difficult to read and
understand.

The ``compass`` package endeavors to increase code readability and code sharing
in a number of ways.

In compass framework
~~~~~~~~~~~~~~~~~~~~

The ``compass`` framework (modules and packages not in the core-specific
packages) has a lot of code that is shared across existing test cases and could
be very useful for future ones.

Most of the framework currently has roughly the same functionality as
:ref:`legacy_compass`, but it has been broken into more modules that make it
clear what functionality each contains, e.g. ``compass.namelists`` and
``compass.streams`` are for manipulating :ref:`dev_namelist` and
:ref:`dev_streams` files, respectively; ``compass.io`` has functionality for
downloading files from the
`LCRC server <https://web.lcrc.anl.gov/public/e3sm/mpas_standalonedata/>`_
and creating symlinks; and ``compass.validation`` can be used to ensure that
variables are bit-for-bit identical between steps or when compared with a
baseline, and to compare timers with a baseline.  This functionality was all
included in 4 very long scripts in :ref:`legacy_compass`.

One example that doesn't have a clear analog in :ref:`legacy_compass` is the
``compass.parallel`` module.  It contains two functions:
:py:func:`compass.parallel.get_available_cores_and_nodes()`, which can find out
the number of total cores and nodes available for running steps, and
:py:func:`compass.parallel.update_namelist_pio()`, which updates the number of
PIO tasks and the stride between tasks based on the number of cores that a step
is actually running with.

Within a core
~~~~~~~~~~~~~

:ref:`legacy_compass` shares functionality with a core by having scripts at the
core level that are linked within test cases and which take command-line
arguments that function roughly the same way as function arguments.  But these
scripts are not able to share any code between them unless it is from
``mpas_tools`` or another external package.

A core in ``compass`` could, theoretically, build out functionality as complex
as in MPAS-Model if desired.  This has already been accomplished for the 3
idealized test cases included in ``compass``.

The shared functionality in the :ref:`dev_ocean` is described in
:ref:`dev_ocean_framework`.

Within a configuration
~~~~~~~~~~~~~~~~~~~~~~

So far, the most common type of shared code within configurations are modules
defining steps that are used in multiple test cases.  For example, the
:ref:`dev_ocean_baroclinic_channel` configuration uses shared modules to define
the ``initial_state`` and ``forward`` steps of each test case.  Configurations
also often include namelist and streams files with replacements to use across
test cases.

In addition to shared steps, the :ref:`dev_ocean_global_ocean` configuration
includes some additional shared framework described in
:ref:`dev_ocean_global_ocean_framework`.

The shared code in ``global_ocean`` has made it easy to define 138 different
test cases using the QU240 (or QUwISC240) mesh.  This is possible because
the same conceptual test (e.g. restart) can be defined:

  * with or without ice-shelf cavities

  * with the PHC or EN4 1900 initial conditions

  * with or without BGC support

  * with the RK4 or split-explicit time integrators

Within a test case
~~~~~~~~~~~~~~~~~~

The main way code is currently reused with a test case is when the same module
for a step gets used multiple times within a test case.  For example,
the :ref:`dev_ocean_baroclinic_channel_rpe_test` test case uses the same
forward run with 5 different values of the viscosity.

.. _dev_dicts_not_classes:

Dictionaries not classes
------------------------

In the process of developing
`MPAS-Analysis <https://github.com/MPAS-Dev/MPAS-Analysis/>`_, we found that
many of our developers were not very comfortable with
`classes <https://docs.python.org/3/tutorial/classes.html>`_, methods,
`inheritance <https://docs.python.org/3/tutorial/classes.html#inheritance>`_
and other concepts related to
`object-oriented programming <https://en.wikipedia.org/wiki/Object-oriented_programming>`_.
In MPAS-Analysis, tasks are implemented as classes to make it easier to use
python's `multiprocessing <https://docs.python.org/3/library/multiprocessing.html>`_
capability.  In practice, this led to code that was complex enough that only
a handful of developers felt comfortable contributing directly to the code.

Since we would like developers to feel comfortable contributing new test cases
to ``compass`` even if they are relatively new to python, we decided not to
use classes here.  But we did need a data structure to represent all of the
data associated with a test case and another for a step.  We settled on
`python dictionaries <https://docs.python.org/3/tutorial/datastructures.html#dictionaries>`_
as the simplest data structure for storing this data.

You can take a look at the :ref:`dev_testcase_dict` and :ref:`dev_step_dict`
for a full listing of the typical entries in these dictionaries. Some entries
are required for the compass :ref:`dev_framework` to work properly.  Others
are used to keep track of parameters of a test case or step that the user
should not alter and therefore that should not be in the :ref:`config_files`.


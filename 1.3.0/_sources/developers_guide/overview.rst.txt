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

Before we dig into the details of how to develop new test cases and other
infrastructure for ``compass``, we first give a little bit of background on
the design philosophy behind the package.

.. _dev_style:

Code Style
----------

All code is required to adhere fairly strictly to the
`PEP8 style guide <https://www.python.org/dev/peps/pep-0008/>`_.  A bot will
flag any PEP8 violations as part of each pull request to
https://github.com/MPAS-Dev/compass.  Please consider using an editor that
automatically flags PEP8 violations during code development, such as
`pycharm <https://www.jetbrains.com/pycharm/>`_ or
`spyder <https://www.spyder-ide.org/>`_, or a linter, such as
`flake8 <https://flake8.pycqa.org/en/latest/>`_ or
`pep8 <https://pep8.readthedocs.io/>`_.  We discourage you from automatically
reformatting your code (e.g. with `autopep8 <https://github.com/hhatto/autopep8>`_)
because this can often produce undesirable and confusing results.

The `flake8 <https://flake8.pycqa.org/en/latest/>`_ utility for linting python
files to the PEP8 standard is included in the COMPASS conda environment. To use
flake8, just run ``flake8`` from any directory and it will return lint results
for all files recursively through all subdirectories.  You can also run it for a
single file or using wildcards (e.g., ``flake8 *.py``).  There also is a
`vim plugin <https://github.com/nvie/vim-flake8>`_ that runs the flake8 linter
from within vim.  If you are not using an IDE that lints automatically, it is
recommended you run flake8 from the command line or the vim plugin before
committing your code changes.

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
   interfaces that are easier to document and understand than the arguments
   passed into a script; and

4) releases of the ``compass`` package would make it easy for developers of
   other python packages and scripts to use our code (though there are not yet
   any "downstream" packages that use ``compass``).

This documentation won't try to provide a whole tutorial on python packages,
modules and classes but we know most developers won't be too clued in on these
concepts so here's a short intro.

Packages
~~~~~~~~

A python package is a directory that has a file called ``__init__.py``.  That
file can be empty or it can have code in it.  If it has functions or classes
inside of it, they act like they're directly in the package.  As an example,
the compass file
`compass/ocean/__init__.py <https://github.com/MPAS-Dev/compass/tree/main/compass/ocean/__init__.py>`_
has a class :py:class:`compass.ocean.Ocean()` that looks like this (with the
`docstrings <https://www.python.org/dev/peps/pep-0257/>`_ stripped out):

.. code-block:: python

    class Ocean(MpasCore):
        def __init__(self):
            super().__init__(name='ocean')

            self.add_test_group(BaroclinicChannel(mpas_core=self))
            self.add_test_group(GlobalOcean(mpas_core=self))
            self.add_test_group(IceShelf2d(mpas_core=self))
            self.add_test_group(Ziso(mpas_core=self))

This class contains all of the ocean test groups, which contain all the ocean
test cases and their steps.  The details aren't important.  The point is that
the class can be imported like so:

.. code-block:: python

    from compass.ocean import Ocean


    ocean = Ocean()

So you don't ever refer to ``__init__.py``, it's like a hidden shortcut so the
its contents can be referenced with just the subdirectory (package) name.

A package can contain other packages and modules (we'll discuss these in just
a second).  For example, the ``ocean`` package mentioned above is inside the
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

As an example, the ``compass`` package contains a module ``list.py``.
There's a function :py:func:`compass.list.list_machines` in that module:

.. code-block:: python

    def list_machines():
        machine_configs = contents('compass.machines')
        print('Machines:')
        for config in machine_configs:
            if config.endswith('.cfg'):
                print('   {}'.format(os.path.splitext(config)[0]))

It lists the supported machines.  You would import this function just like in
the package example above:

.. code-block:: python

    from compass.list import list_machines


    list_machines()

So a module named ``foo.py`` and a package in a directory named ``foo`` with
an ``__init__.py`` file look exactly the same when you import them.

So why choose one over the other?

The main reason to go with a package over a module is if you need to include
other files (such as other modules and packages, but also other things like
:ref:`config_files`, namelists and streams files).  It's
always pretty easy to make a module into a package (by making a directory with
the name of the package, moving the module in, an renaming it ``__init__.py``)
or visa versa (by renaming ``__init__.py`` to the module name, moving it up
a directory, and deleting the subdirectory).

Classes
~~~~~~~

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

Based on this experience, we were hesitant to use classes in ``compass`` and
tried an implementation without them.  This led to a clumsy set of functions
and `python dictionaries <https://docs.python.org/3/tutorial/datastructures.html#dictionaries>`_
that was equally complex but harder to understand and document than classes.

The outcome of this experience is that we have used classes to define
MPAS cores, test groups, test cases and steps.  Each MPAS core will "descend"
from the :py:class:`compass.MpasCore` base class; each test groups descends
from :py:class:`compass.TestGroup`; each test case descends from
:py:class:`compass.TestCase`; and each steps descends from
:py:class:`compass.Step`.  These base classes contain functionality that can
be shared with the "child" classes that descend from them and also define
a few "methods" (functions that belong to a class) that the child class is
meant to "override" (replace with their own version of the function, or augment
by replacing the function and then calling the base class's version of the
same function).

We will provide a tutorial on how to add new MPAS cores, test groups, test
cases and steps in the near future that will explain the main features of
classes that developers need to know about.  Until that is available, we hope
that the examples currently in the package can provide a starting point.

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

The ``compass`` package is also dense and will have a learning curve.  We hope
the python package approach is worth it because the skills learned to work with
it will be more broadly applicable than those required for
:ref:`legacy_compass`. In developing ``compass`` we endeavor to increase code
readability and code sharing in a number of ways.

In compass framework
~~~~~~~~~~~~~~~~~~~~

The ``compass`` framework (modules and packages not in the MPAS-core packages)
has a lot of code that is shared across existing test cases and could be very
useful for future ones.

Most of the framework currently has roughly the same functionality as
:ref:`legacy_compass`, but it has been broken into more modules that make it
clear what functionality each contains, e.g. ``compass.namelists`` and
``compass.streams`` are for manipulating namelist and
streams files, respectively; ``compass.io`` has functionality for
downloading files from the
`LCRC server <https://web.lcrc.anl.gov/public/e3sm/mpas_standalonedata/>`_
and creating symlinks; and ``compass.validation`` can be used to ensure that
variables are bit-for-bit identical between steps or when compared with a
baseline, and to compare timers with a baseline.  This functionality was all
included in 4 very long scripts in :ref:`legacy_compass`.

One example that doesn't have a clear analog in :ref:`legacy_compass` is the
``compass.parallel`` module.  It contains a function
:py:func:`compass.parallel.get_available_cores_and_nodes()` that can find out
the number of total cores and nodes available for running steps.

Within an MPAS core
~~~~~~~~~~~~~~~~~~~

:ref:`legacy_compass` shared functionality within a MPAS core by having scripts
at the core level that were linked within test cases and which took
command-line arguments that function roughly the same way as function
arguments.  But these scripts were not able to share any code between them
unless it is from ``mpas_tools`` or another external python package.

An MPAS core in ``compass`` could, theoretically, build out functionality as
complex as in the MPAS components themselves.  This has already been
accomplished for several of the idealized test cases included in ``compass``.

The shared functionality in the :ref:`dev_ocean` is described in
:ref:`dev_ocean_framework`.

Within a test group
~~~~~~~~~~~~~~~~~~~

So far, the most common type of shared code within test group are modules
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

  * with the WOA23, PHC or EN4 1900 initial conditions

  * with the RK4 or split-explicit time integrators

Within a test case
~~~~~~~~~~~~~~~~~~

The main way code is currently reused with a test case is when the same module
for a step gets used multiple times within a test case.  For example,
the :ref:`dev_ocean_baroclinic_channel_rpe_test` test case uses the same
forward run with 5 different values of the viscosity.

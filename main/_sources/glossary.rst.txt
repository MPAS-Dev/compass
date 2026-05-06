.. _glossary:

Glossary
========

``compass``
    The python package containing framework for listing, setting up and running
    test cases, as well as the cores, test groups, test cases and steps.

Legacy COMPASS
    The old version of COMPASS that used XML files and python scripts to define
    cores, test groups, test cases and steps.

MPAS core
    This term refers to the collection of tests cases associated with an
    MPAS dynamical core, ``ocean`` for MPAS-Ocean and ``landice`` for MALI.


Step
    A step is the smallest units of work in ``compass`` that you can run on
    its own.  Each test case is made up of a sequence of steps.  Currently,
    these steps run in sequence, but there are plans to allow them to run in
    parallel in the near future.  Common types of steps in ``compass`` are
    those for creating meshes and/or initial conditions; for running the MPAS
    model; and for performing visualization of the results.


Test case
    A test case is the smallest unit of work that ``compass`` will let a user
    set up on its own.  It is a sequence of one or more steps that a user can
    either run all together or one at a time.  A test case is often independent
    of other test cases.  For example, many test groups in ``compass`` include
    test cases for checking bit-for-bit restart and decomposition across
    different numbers of cores and threads.  Sometimes, it is convenient to
    have test cases that depend on other test cases.  The most common example
    of this in ``compass`` are test cases that create meshes and/or initial
    conditions that are used in several subsequent test cases.

Test group
    A test group is a collection of test cases with a common concept or
    purpose.  Each MPAS core defines several test groups.  Examples in the
    :ref:`landice` are :ref:`landice_dome` and :ref:`landice_greenland`, while
    examples from :ref:`ocean` include :ref:`ocean_baroclinic_channel` and
    :ref:`ocean_global_ocean`.

Test suite
    A collection of test cases that are from the same MPAS core (but not
    necessarily the same test group) that can be run together with a single
    command (``compass run <suite>``, where ``<suite>`` is the suite's name).
    Sometimes, test suites are used to perform regression testing to make sure
    new MPAS-model features do not add unexpected changes to simulation
    results.  Typically, this involves providing a "baseline" run of the same
    test suite to compare with.  Other types of test suites are used to run
    a sequence of test cases that depend on one another (e.g. creating a mesh,
    setting up an initial condition, running the model, and processing the
    output, as in :ref:`ocean_global_ocean`).

Work directory
   The location where test cases will be set up and run.  The "base work
   directory" is the root for all MPAS cores (and therefore test groups, test
   cases and steps), and is the path passed with the ``-w`` or ``-b`` flags
   to :ref:`dev_compass_setup` and :ref:`dev_compass_suite`.  The work
   directory for a test case or step its location within base work directory
   (the base work directory plus its relative path)


Package
    A python package is a directory that has a file called ``__init__.py``.
    That file can be empty or it can have code in it.  If there is code in
    ``__init__.py``, it gets imported as if it were directly in the package
    (you never include ``__init__`` in an ``import`` statement).

Module
    Python modules are python files that can be imported by other python files
    (so they're not just scripts).  Nearly every single file ending in ``.py``
    in the ``compass`` package is a module.  The ``__init__.py`` files are a
    special case, that may define a module with the name of the package
    (directory) that ``__init__.py`` is in.

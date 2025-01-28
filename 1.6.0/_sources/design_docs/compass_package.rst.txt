
compass python package
======================

Author: Xylar Asay-Davis

date: 2020/11/16


Summary
-------

While the existing COMPASS infrastructure has served us well in providing a
framework for setting up MPAS test cases and test suites, several shortcomings
have emerged over the years.  First, new users have not found the current
system of creating XML files that are parsed into python scripts, namelists and
streams files very intuitive or easy to modify.  Second, the current scripts and
XML files do not lend themselves to code reuse, leading to a cumbersome system
of symlinked scripts and XML config files.  Third, the only way that users
currently have of modifying test cases is to edit namelists, streams files and run
scripts for each step individually after the test case has been set up.  Fourth
and related, there is not a way for users to easily constrain or modify
how many cores a given test case uses, making it hard to configure test cases
in a way that is appropriate for multiple machines.  Fifth and also related,
COMPASS does not currently have a way to provide machine-specific paths and
other information that could allow for better automation on supported machines.
Sixth, the directory structure imposed by COMPASS
(``mpas_core/test_group/resoltuion/test_case/step``) is too rigid for many
applications. Finally, COMPASS is not well documented and the documentation that
does exist is not very helpful either for new users or for new developers
interested in creating new test cases.

The proposed ``compass`` python package should address these challenges with
the hope of making the MPAS test cases significantly easier to develop and run.

Requirements
------------

.. _req_easy:

Requirement: Make test cases easy to understand, modify and create
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2020/12/04

Contributors: Xylar Asay-Davis, Luke Van Roekel


Currently, test cases are written primarily in XML files that are then used to
generate a python script along with namelist and streams files for MPAS-Model.
We have found that this system is not very intuitive for new users or very easy
to get started with.  New users would likely have an easier time if test cases
were written in a more direct way, using a common language rather than custom
XML tags.

Importantly, creating a test case should also be as easy as possible.  There is a
need to balance readability and reusability. There is a risk that the compass
redesign, as it becomes heavily pythonic, may make it difficult for developers to
contribute.  But we can't go too far the other way either. We want the best
balance possible between readability and reusibility.


.. _req_shared_code:

Requirement: Shared code
^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2020/11/16

Contributors: Xylar Asay-Davis

Currently, there are two approaches to sharing code between COMPASS test cases.
In some cases, shared code is part of an external package, often ``mpas_tools``,
which is appropriate for code that may be used outside of COMPASS.  However,
this approach is cumbersome for testing, so it is not the preferred approach for
COMPASS-specific code.  In other cases, scripts are symlinked in test cases and
run with test-case-specific flags.  This approach is also cumbersome and does
not lend itself to code reuse between scripts.  Finally, many test cases attempt
to share XML files using symlinks, a practice that has led to frequent
unintended consequences when a linked file is modified with changes appropriate
for only one of the test cases that uses it.  A more sophisticate method
for code reuse should be developed beyond symlinks to isolated scripts and
shared XML files.


.. _req_shared_options:

Requirement: Shared configuration options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2020/11/16

Contributors: Xylar Asay-Davis

Currently, COMPASS reads a configuration file as part of setting up test cases,
but these configuration options are largely unavailable to test cases themselves.
Some steps of some test cases (e.g. ``files_for_e3sm`` in some
``ocean/global_ocean`` test cases) have their own dedicated config files, but
these are again separate from the config file used to set up test cases, are
awkward to modify (requiring editing after test case generation).


.. _req_core_count:

Requirement: Ability specify/modify core counts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2020/11/16

Contributors: Xylar Asay-Davis

Some test cases involve multiple steps of running the MPAS model, each with a
hard-coded number of cores (and often with a corresponding hard-coded number of
PIO tasks), which makes it tedious to modify the number of cores or nodes that
a given test case uses.  This problem is exacerbated in test suites, where it is
even more difficult and tedious to modify processor counts for individual test
cases.  A system is needed where the user can more easily override the default
number of cores used in one or more steps of a test case.  The number of PIO
tasks and the stride between them should be updated automatically to accommodate
the new core count.


.. _req_machine_data:

Requirement: Machine-specific data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2020/11/16

Contributors: Xylar Asay-Davis

Currently, many COMPASS test cases have hard-coded processor counts and related
information that are likely only appropriate for one machine.  Users must
specify the paths to shared datasets such as meshes and initial conditions.
Users must also know where to load the ``compass`` conda environment appropriate
for running test cases.  If information were available on the system being used,
such as the number of cores per node and the locations of shared paths,
test cases and the COMPASS infrastructure could take advantage of this to
automate many aspects of setting up and running test cases that are currently
unnecessarily redundant and tedious.


.. _req_dir_struct:

Requirement: Looser, more flexible directory structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2020/11/16

Contributors: Xylar Asay-Davis

The directory structure currently imposed by COMPASS
(``mpas_core/test_group/resoltuion/test_case/step``) is too rigid for many
applications.  Some test cases (e.g. convergence tests) require multiple
resolutions within the test case.  Some test groups would prefer to sort
test cases based on another parameter or property besides resolution.  It would
be convenient if the directory structure could be more flexible, depending on
the needs of a given test group and test case.  Even so, it is important that
the subdirectory of each test case and step is unique, they do not overwrite one
another.


.. _req_docs:

Requirement: User- and developer-friendly documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2020/11/16

Contributors: Xylar Asay-Davis

We need a set of user-friendly documentation on how to setup and activate an
appropriate conda environment; build the appropriate MPAS core; list and setup
a test case; and run the test case in via a batch queuing system.

Similarly, we need a set of developer-friendly documentation to describe how to
create a new "test group" with one or more "test cases", each made up of
one or more "steps".


.. _req_parallel:

Requirement: Considerations related to running test cases in parallel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2020/12/10

Contributors: Xylar Asay-Davis, Matt Hoffman

In the longer term, we would like ot add the capability of running multiple
test cases within a test suite in parallel with one another for reduced
wall-clock time.  Similarly, we would also like to support multiple steps within
a test case running in parallel with one another (e.g. the forward runs with
different viscosities in the baroclinic channel RPE test case).  Full support for
this capability will not be included in this design, but design choices should
be mindful of this future addition in the hopes of minimizing future
modifications, particularly to individual test cases.


.. _req_res:

Requirement: Resolution can be a test case parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2020/12/04

Contributors: Xylar Asay-Davis, Mark Petersen

Currently, resolution is hard-coded in the directory structure and in scripts
for individual test groups like ``build_base_mesh.py``. This works for more
complex meshes but for convergence tests, it is not useful to have a directory
per resolution.  Instead, it could be helpful to have a list of resolutions that
can easily be altered (e.g. ``dx = {min, max, step}`` with a linear or log step)
with either configuration options or within the code. For convergence tests,
resolution is a parameter, rather than something fundamental.  This could also
reduce the number of test cases in the full list.


.. _req_alter_code:

Requirement: Test case code is easy to alter and rerun
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2020/12/04

Contributors: Xylar Asay-Davis, Mark Petersen

In the current ``compass``, the created directories include soft links to
scripts like ``build_base_mesh.py`` and ``add_initial_condition.py``. It is
easy to edit that file and rerun it, and quickly iterate until one gets the
desired result. New people also understand this workflow. The new design should
still be easy to work with.


.. _req_premade_ic:

Requirement: Support for pre-made initial condition files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2020/12/04

Contributors: Xylar Asay-Davis, Mark Petersen

Ideally, it should be possible for a given test case to either generate an
initial condition or read a pre-made initial condition from a file (possibly
downloading this file if it has not been cached).  Alternatively, two different
versions of a test case could exists, one with the generated and one with the
pre-made initial condition.


.. _req_batch:

Requirement: Easy batch submission
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2020/12/04

Contributors: Xylar Asay-Davis, Mark Petersen

There should be an easy way for users to submit batch jobs without having to
create their own batch script or modify an example.


Algorithm Design
----------------

.. _alg_easy:

Algorithm design: Make test cases easy to understand, modify and  and create
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/04/13

Contributors: Xylar Asay-Davis


The proposed solution would be to write test cases as Python packages made up
of modules, functions and classes within a larger ``compass`` package.  A test
case will descend from a base ``TestCase`` class with a constructor for
adding steps to the test case (the equivalent of parsing ``config_driver.xml``
in the current implementation), a ``configure()`` method for adding
test-case-specific config options, and a ``run()`` method to run the steps
and perform validation.  Each step of a test case (equivalent to the other
``config_*.xml`` files) will descend from the ``Step`` base class.  Each
step class will include a constructor to add input, output, namelist and
streams files and collect other information on the step (equivalent to parsing
``config_*.xml``); a ``setup()`` method that downloads files, makes symlinks,
creates namelist and streams files; and a ``run()`` method that runs the step.
Steps may be shared between test cases.  A balance will have to be struck
between code reusability and readability within each test group (a set of test
cases).

Readability would be improved by using Jinja2 templates for code generation,
rather than via string manipulation within python scripts as is the case in the
current COMPASS implementation:

.. code-block:: python

    #!/usr/bin/env python
    import pickle
    import configparser

    from mpas_tools.logging import LoggingContext


    def main():
        with open('test_case_{{ test_case.name }}.pickle', 'rb') as handle:
            test_case = pickle.load(handle)
        test_case.steps_to_run = ['{{ step.name }}']
        test_case.new_step_log_file = False

        with open('{{ step.name }}.pickle', 'rb') as handle:
            step = pickle.load(handle)

        config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())
        config.read('{{ step.config_filename }}')
        test_case.config = config

        # start logging to stdout/stderr
        test_name = step.path.replace('/', '_')
        with LoggingContext(name=test_name) as logger:
            test_case.logger = logger
            test_case.run()


    if __name__ == '__main__':
        main()


A Jinja2 template uses curly braces (e.g. ``{{ test_case.name }}``) to indicate
where an element of the template will be replaced by a python variable or
dictionary value.  In this example, ``{{ test_case.name }}`` will be replaced
with the contents of ``test_case['name']`` in the python code, and similarly
for other replacements in the template.  Other than the replacements, the code
can be read as normal, in contrast to the existing approach of python scripts
that define other python scripts via a series of string formatting statements.

The only XML files that would be used would be templates for streams files,
written in the same syntax as the resulting streams files.

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


Templates for namelist files would have the same basic syntax as the resulting
namelist files:

.. code-block:: ini

    config_write_output_on_startup = .false.
    config_run_duration = '0000_00:15:00'
    config_use_mom_del2 = .true.
    config_implicit_bottom_drag_coeff = 1.0e-2
    config_use_cvmix_background = .true.
    config_cvmix_background_diffusion = 0.0
    config_cvmix_background_viscosity = 1.0e-4

Regarding the balance between reusability and readability, it is difficult to
generalize this to the whole redesign.  To some degree this will be a choice
left to each test case.  It will be difficult to reuse code across test cases
and steps within a test group without some degree of increased complexity.
The redesign will attempt to include simpler examples, perhaps with less code
reuse, that can serve as starting points for the creation of new test cases.
These "prototype" test cases will include additional documentation and commenting
to help new developers follow them and use them to design their own test cases.

Even without the compass redesign, a certain familiarity with use of python
packages is somewhere between recommended and required to add new test cases to
COMPASS.  With the redesign, it will become essentially inevitable that
developers have a certain minimum level of familiarity with python.  While there
may be a learning curve, it is hoped that these skills will pay off far beyond
COMPASS in a way that learning the existing XML-based approach cannot be.


.. _alg_shared_code:

Algorithm design: Shared code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/04/13

Contributors: Xylar Asay-Davis


By organizing both the test cases themselves and shared framework code into a
``compass`` Python package, code reuse and organization should be greatly
simplified.

The organization of the package will be as follows:

.. code-block:: none

  - compass/
    - <mpas_core>/
      - <mpas_core>.cfg
      - <mpas_core_framework_module>.py
      - <mpas_core_framework_package>/
      - tests/
        - <test_group>/
          - <test_case>/
            - <step>.py
            - <test_case>.cfg
            - namelist.<step>
            - streams.<step>
          - <shared_step>.py
          - <test_group_shared_module>.py
          - <test_group>.cfg
          - namelist.<step>
          - streams.<step>
    - <framework_module>.py
    - <framework_package>/

The proposed solution would slightly modify the naming conventions currently
used in COMPASS. An MPAS core (descended from the ``MpasCore`` base class)
would be the same as it now is -- corresponding to an MPAS dynamical core such
as ``ocean`` or ``landice``.  A test group (descended from ``TestGroup``) would
would be the equivalent of a "configuration" in legacy COMPASS -- a group of
test cases such as ``global_ocean`` or ``MISMIP3D``.  For at least two reasons
described in :ref:`req_dir_struct`, we do not include ``resolution`` as the
next level of hierarchy.  Instead, a test group contains test cases
(descended from ``TestCase``), which can be given any convenient name and
relative path to distinguish it from other test cases within that test group.
Several variants of a test case can define by varying a parameter or other
characteristic (including resolution) but there need not be separate packages
or modules for each.  This is an important aspect of the code reuse provided by
this approach.  Each test case is made up of several steps (e.g. ``base_mesh``,
``initial_state``, ``forward``).  Legacy COMPASS' documentation referred to
a test case as a "test" and a step as a "case", but users have found this
naming convention to be confusing so the proposed solution tries to make a
clearer distinction between a test case and a step within a test case.

In addition to defining test cases and steps, MPAS cores and test groups
can also include "framework" python code that could be more general (e.g. for
creating meshes or initial conditions).  The main ``compass`` package would
also include several framework modules and package, some for infrastructure
related to listing, setting up and cleaning up test cases, and others for tasks
common to many test cases.  The methods of the base classes, particularly of
the ``Step`` class, are also an important part of the framework that can be
used to indicate what the input and output files for the step are and how to
create the namelist and streams files.  Here is an example of a step that
that is defined using a combination of methods from ``Step`` (e.g.
``self.add_input_file()``) and framework functions (e.g. ``run_model()``):

.. code-block:: python

    from compass.model import run_model
    from compass.step import Step


    class Forward(Step):
        """
        A step for performing forward MPAS-Ocean runs as part of baroclinic
        channel test cases.

        Attributes
        ----------
        resolution : str
            The resolution of the test case
        """
        def __init__(self, test_case, resolution, name='forward', subdir=None,
                     cores=1, min_cores=None, threads=1, nu=None):
            """
            Create a new test case

            Parameters
            ----------
            test_case : compass.TestCase
                The test case this step belongs to

            resolution : str
                The resolution of the test case

            name : str
                the name of the test case

            subdir : str, optional
                the subdirectory for the step.  The default is ``name``

            cores : int, optional
                the number of cores the step would ideally use.  If fewer cores
                are available on the system, the step will run on all available
                cores as long as this is not below ``min_cores``

            min_cores : int, optional
                the number of cores the step requires.  If the system has fewer
                than this number of cores, the step will fail

            threads : int, optional
                the number of threads the step will use

            nu : float, optional
                the viscosity (if different from the default for the test group)
            """
            self.resolution = resolution
            if min_cores is None:
                min_cores = cores
            super().__init__(test_case=test_case, name=name, subdir=subdir,
                             cores=cores, min_cores=min_cores, threads=threads)
            self.add_namelist_file('compass.ocean.tests.baroclinic_channel',
                                   'namelist.forward')
            self.add_namelist_file('compass.ocean.tests.baroclinic_channel',
                                   'namelist.{}.forward'.format(resolution))
            if nu is not None:
                # update the viscosity to the requested value
                options = {'config_mom_del2': '{}'.format(nu)}
                self.add_namelist_options(options)

            self.add_streams_file('compass.ocean.tests.baroclinic_channel',
                                  'streams.forward')

            self.add_input_file(filename='init.nc',
                                target='../initial_state/ocean.nc')
            self.add_input_file(filename='graph.info',
                                target='../initial_state/culled_graph.info')

            self.add_output_file(filename='output.nc')

        def setup(self):
            """
            Set up the test case in the work directory, including downloading any
            dependencies
            """
            self.add_model_as_input()

        def run(self):
            """
            Run this step of the test case
            """
            run_model(self)


.. _alg_shared_config:

Algorithm design: Shared configuration options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/04/13

Contributors: Xylar Asay-Davis


In the work directory, each test case will have a single config file that is
populated during the setup phase and which is symlinked within each step of the
test case.  The idea of having a single config file per test case, rather than
one for each step, is to make it easier for users to modify config options in
one place at runtime before running all the steps in a test case.  This will
hopefully avoid the tedium of altering redundant namelist or config options in
each step.

The config files will be populated from default config options provided in
several config files within the ``compass`` package.  Any config options read in
from a later config file will override the same option from an earlier config
file, so the order in which the files are loaded is important.  The proposed
loading order is:

* A top level default config file related downloading files and partitioning
  meshes for parallel execution

* machine config file (found in ``compass/machines/<machine>.cfg``, with
  ``default`` being the machine name if none is specified)

* MPAS core config file (found in ``compass/<mpas_core>/<mpas_core>.cfg``)

* test group config file (found in
  ``compass/<mpas_core>/tests/<test_group>/<test_group>.cfg``)

* any additions or modifications made within the test case's ``configure()``
  method.

* the config file passed in by the user at the command line (if any).

The ``configure()`` method allows each test case to load one or more config
files specific to the test case (e.g. ``<test_case>.cfg`` within the test
case's package) and would also allow calls to ``config.set()`` that define
config options directly.

The resulting config file would be written to ``<test_case>.cfg`` within the
test case directory and symlinked to each step subdirectory as stated above.


Algorithm design: Ability specify/modify core counts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/04/13

Contributors: Xylar Asay-Davis


Each step will specify the "target" number of cores, the minimum possible
number of cores, a number of treads, the maximum memory it will be allowed to
use, and the maximum amount of disk space it can use.  These specifications are
with the ``WorkerQueue`` approach in mind for future parallelism, as explained
in :ref:`alg_parallel`.

The total number of available cores will be determined via python or slurm
commands.  An error will be raised if too few cores are available for a
particular step.  Otherwise, the step will run on the minimum of the target
number of cores or the total available.

Some test cases (e.g. those within the ``global_ocean`` test group) will
allow the user to specify the target and minimum number of cores as config
options, meaning they can be set to non-default values before running the test
case.  Config options are common to all steps within a test case, but
the target and minimum cores are a property of each step that must be known
before it is run (again for reasons related to a likely strategy for
future parallelism in :ref:`alg_parallel`).  This means that a test case will
need to parse the config options and use them to determine the number of cores
each step needs to run with as part of its ``run()`` method before calling
``super().run()`` from the base class to run the steps.

Parsing config options and updating the target and minimum cores in a step will
need to happen in each test cases that supports this capability.  From there,
shared infrastructure will take care of determining if sufficient cores are
available and how many to run each step with if so.  Developers of individual
test cases will not need to worry about this.  Here is an example from the
``Init`` test case from the ``GlobalOcean`` test group:

.. code-block:: python

    def run(self):
        """
        Run each step of the testcase
        """
        config = self.config
        steps = self.steps_to_run
        if 'initial_state' in steps:
            step = self.steps['initial_state']
            # get the these properties from the config options
            step.cores = config.getint('global_ocean', 'init_cores')
            step.min_cores = config.getint('global_ocean', 'init_min_cores')
            step.threads = config.getint('global_ocean', 'init_threads')

        if 'ssh_adjustment' in steps:
            step = self.steps['ssh_adjustment']
            # get the these properties from the config options
            step.cores = config.getint('global_ocean', 'forward_cores')
            step.min_cores = config.getint('global_ocean', 'forward_min_cores')
            step.threads = config.getint('global_ocean', 'forward_threads')

        # run the steps
        super().run()
        ...


Shared infrastructure can also be used to set the number of PIO tasks to one
per node, using the number of cores for a given step and the number of cores
per node from the machine config file (see :ref_`alg_machine_data`).


.. _alg_machine:

Algorithm design: Machine-specific data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/04/13

Contributors: Xylar Asay-Davis


The machine config file mentioned in :ref:`alg_shared_config` would have
the following config options:

.. code-block:: cfg

    # The paths section describes paths that are used within the ocean core test
    # cases.
    [paths]

    # The root to a location where the mesh_database, initial_condition_database,
    # and bathymetry_database for MPAS-Ocean will be cached
    ocean_database_root = /usr/projects/regionalclimate/COMMON_MPAS/ocean/grids/

    # The root to a location where the mesh_database and initial_condition_database
    # for MALI will be cached
    landice_database_root = /usr/projects/regionalclimate/COMMON_MPAS/mpas_standalonedata/mpas-albany-landice

    # the path to the base conda environment where compass environments have
    # been created
    compass_envs = /usr/projects/climate/SHARED_CLIMATE/anaconda_envs/base


    # The parallel section describes options related to running tests in parallel
    [parallel]

    # parallel system of execution: slurm or single_node
    system = slurm

    # whether to use mpirun or srun to run the model
    parallel_executable = srun

    # cores per node on the machine
    cores_per_node = 36

    # the slurm account
    account = e3sm

    # the number of multiprocessing or dask threads to use
    threads = 18

The various ``paths`` would help with finding mesh or initial condition files.
The database root paths depend on the MPAS core, so new paths would need to be
added for new cores.

A strategy for setting environment variables, activating the appropriate conda
environment, and loading compiler and MPI modules for each machine will be
explored as a follow-up project and is not part of this design.

The ``parallel`` options are intended to contain all of the machine-specific
information needed to determine how many cores a given step would require.  The
use of python thread parallelism will not be part of the first version of the
``compass`` package described in this design document but is expected to be
incorporated in the coming year.  An appropriate value for ``threads`` for
each machine will likely need determined as that capability gets more
exploration but is left as a placeholder for the time being.


.. _alg_dir_struct:

Algorithm design: Looser, more flexible directory structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/04/13

Contributors: Xylar Asay-Davis


Each test case and step will be defined by a unique subdirectory within the
work directory.  Within the base work directory, the first two levels of
subdirectories will be conceptually the same as in the current implementation:
``mpas_core/test_group``.  However, test cases will be free to determine the
(unique) subdirectory structure beyond this top-most level.  Many existing
test cases will likely stick with the ``resolution/test_case/step`` organization
structure imposed in the legacy COMPASS framework, but others may choose a
different way of organizing (and, indeed, many test cases already have given the
``resolution`` subdirectory a name that is seemingly unrelated to the mesh
resolution).  A unique subdirectory for each test case and step will be provided
as the ``subdir`` argument to the base class's constructor (i.e.
``super().__init__()`` or will be taken from the ``name`` argument if
``subdir`` is not provided.

.. code-block:: python

    name = 'restart_test'
    self.resolution = resolution
    subdir = '{}/{}'.format(resolution, name)
    super().__init__(test_group=test_group, name=name,
                     subdir=subdir)

COMPASS will list test cases based on their full paths within the work directory,
since this is the way that they can be uniquely identified.


.. _alg_docs:

Algorithm design: User- and developer-friendly documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2020/04/13

Contributors: Xylar Asay-Davis


Documentation using ``sphinx`` and the ``ReadTheDocs`` template will be built
out in a manner similar to what has already been done for:

* `geometric_features <https://mpas-dev.github.io/geometric_features/stable/>`_

* `pyremap <https://mpas-dev.github.io/pyremap/stable/>`_

* `MPAS-Tools <https://mpas-dev.github.io/MPAS-Tools/stable/>`_

* `MPAS-Analysis <https://mpas-dev.github.io/MPAS-Analysis/latest/>`_

The documentation will include:

* A user's guide for

  * setting up the conda environment

  * listing, setting up, and cleaning up test case

  * regression suites

  * creating and modifying config files

  * more details on each MPAS core, test group, test case and step

  * machine-specific instructions

* A developer's guide:

  * A quick start

  * An overview (e.g. the design philosophy)

  * A section for each MPAS core

    * A subsection describing the test groups

      * A sub-subsection for each test case and its steps

    * A subsection for the MPAS core's framework code

  * A description of the ``compass`` framework code:

    * for use within test cases

    * for listing, setting up and cleaning up test cases

    * for managing regression test suites

  * An automated documentation of the API pulled from docstrings

Eventually, but probably not as part of the current design, the documentation
will also include:

* A developer's guide for creating new test cases

  * MPAS-core-specific details for developing new test cases

* More detailed tutorials:

  * Running a test case

  * Running the regression suite


.. _alg_parallel:

Algorithm design: Considerations related to running test cases in parallel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/04/13

Contributors: Xylar Asay-Davis

I plan to use `parsl <https://parsl.readthedocs.io/en/stable/>`_ to support
parallelism between both test cases and steps within a test case.  After reading
documentation, running tutorials, and beginning prototyping, it seems that the
relatively new
`WorkQueueExecutor <https://parsl.readthedocs.io/en/stable/stubs/parsl.executors.WorkQueueExecutor.html#parsl.executors.WorkQueueExecutor>`_
is likely to be the approach within Parsl that allows the level of flexibility
and control that we would need.  However, this is a new enough feature
that it is still considered to be "beta" and is not available in the latest
release (v1.0.0).  So it seems premature to settle on this design choice or to
begin to incorporate it into code (except perhaps as a separate prototype).

Even so, some design choices can be made with future support for Parsl in mind.
Each step of a test case will be required to provide full paths to its input and
output files so that, in the future, Parsl can determine dependencies between
test cases and their steps using these files and control execution accordingly.
This will be the only method for determining dependencies, so steps will have to
be accurate in providing their inputs and outputs to avoid errors,
race conditions, or unnecessary blocking.  Test cases with an test suite and
steps within a test case will also need to be ordered in such a way that outputs
of a "prerequisite" step are always defined before the inputs of any subsequent
steps that need them as inputs.  In the future, this should allow ``compass``
to associate each input file with a so-called Parsl ``DataFuture``, which will
allow each step of a test case to run only when all of its input files are
available.

Also with Parsl in mind, the ``Step`` base class includes a specified maximum
memory and disk usage.  Currently, these are set to an arbitrary
reference value of 1GB each but will be calibrated to the actual approximate
usage of each step once this can be determined using debugging output from
Parsl.

This design solution will be fleshed out further in a separate document at a
later date.


.. _alg_res:

Algorithm design: Resolution can be a test case parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/04/13

Contributors: Xylar Asay-Davis

As mentioned in :ref:`alg_shared_code` and :ref:`alg_dir_struct`, resolution
will no longer be part of the directory structure for test cases and
no restrictions will be placed on how individual test cases handle resolution
or mesh generation.  To facilitate shared code, a test group can use the
same code for a step that generates a mesh and/or initial condition for
different resolutions, e.g. passing in the resolution or mesh name as an
argument to the step's constructor.


.. _alg_alter_code:

Algorithm design: Test case code is easy to alter and rerun
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/04/13

Contributors: Xylar Asay-Davis

When ``python -m compass setup`` or ``python -m compass suite`` is run from a
local ``compass`` repo as opposed to the conda package, the package creates
a local symlink within each test case and step's work directory to the
``compass`` package.  A developer can edit any files within the package either
using the symlink or in the original local repo and then simply rerun the test
case or step without having to rerun setup. Changes do not require a test build
of a conda package or anything like that.  After some discussion about adding
symlinks to individual python files within the ``compass`` package, it was
decided that this has too many risks of being misunderstood, having unintended
consequences, and could be difficult to implement.


.. _alg_premade_ic:

Algorithm design: Support for pre-made initial condition files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/04/13

Contributors: Xylar Asay-Davis, Mark Petersen

To a large degree, the implementation of this requirement will be left up to
individual test cases.  It should not be difficult to add a config option
to a given test case selecting whether to generate an initial condition or
read it from a file (and skipping initialization steps if the latter).

The suggested approach would be to put an initial condition in the
``initial_condition_database`` under a directory structure similar to the
``compass`` work directory.  The initial condition would have a date stamp so
new initial conditions could be added over time without breaking backwards
compatibility.

However, this work will be considered outside the scope of this design document
and is only discussed to ensure that the proposed design does not hinder a
future effort in this direction.

.. _alg_batch:

Algorithm design: Easy batch submission
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/01/14

Contributors: Xylar Asay-Davis, Mark Petersen

Rather than having users create their own batch scripts from scratch, a simper
solution would be to generate a job script appropriate for a given
machine from a template.  This has been done for performance tests,
`see example <https://github.com/MPAS-Dev/MPAS-Tools/blob/main/ocean/performance_testing/submit_performance_test_to_queue.py#L96>`_
for single line command. An alternative will be to use ``parsl`` to handle the
SLURM (or other) submission.

Prototyping that is currently underway will help to decide which approach we
use for individual test cases.  ``parsl`` will most likely be used for test
suites.  This work will not be part of the current implementation but an effort
will be made to ensure that the design doesn't hinder later automatic
generation of batch scripts.  Additional information such as a default account
name could be added to machine-specific config files to aid in this process.


Implementation
--------------

The implementation of this design can be found in the branch:
`xylar/compass/compass_1.0 <https://github.com/xylar/compass/tree/compass_1.0>`_
and on the pull request at:
https://github.com/MPAS-Dev/compass/pull/28


.. _imp_easy:

Implementation: Make test cases easy to understand, modify and  and create
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/04/13

Contributors: Xylar Asay-Davis


As already discussed, this requirement is somewhat in conflict with
:ref:`req_shared_code`, in that shared code within a test case tends to lead
to a bit more complexity but considerably less redundancy.

In addition to the constructor (``__init__()``), the ``TestCase`` base class
has 2 other methods, ``configure()`` and ``run()``, that child classes are
expected to override to set config options and perform additional tasks beyond
just running the steps that belong to the test case. Similarly, in addition to
the constructor, the ``Step`` base class has 2 other methods, ``setup()`` and
``run()`` for setting up and running the step.  Each of these is described
below.

constructors
~~~~~~~~~~~~

When test cases and steps are instantiated, the constructor method
``__init__()`` is called. I will not go into the details of what happens in
the ``TestCase`` and ``Step`` base classes when this happens because the idea
is that developers of new test cases would not need to know these details.
The constructors always need to take the parent (a ``TestGroup`` or
``TestCase`` object, respectively) and they can have additional arguments (such
as the resolution or other parameters). The constructors must always call the
base class' constructor ``super().__init__()`` with, at a minimum, the parent
and the name of the test case or step as arguments.

As an example, here is the constructor for the ``Default`` test case in the
``BaroclinicChannel`` test group in the ``Ocean`` MPAS core:

.. code-block:: python

    class Default(TestCase):
        """
        The default test case for the baroclinic channel test group simply creates
        the mesh and initial condition, then performs a short forward run on 4
        cores.

        Attributes
        ----------
        resolution : str
            The resolution of the test case
        """

        def __init__(self, test_group, resolution):
            """
            Create the test case

            Parameters
            ----------
            test_group : compass.ocean.tests.baroclinic_channel.BaroclinicChannel
                The test group that this test case belongs to

            resolution : str
                The resolution of the test case
            """
            name = 'default'
            self.resolution = resolution
            subdir = '{}/{}'.format(resolution, name)
            super().__init__(test_group=test_group, name=name,
                             subdir=subdir)

            self.add_step(
                InitialState(test_case=self, resolution=resolution))
            self.add_step(
                Forward(test_case=self, cores=4, threads=1, resolution=resolution))

And here is the constructor of the ``InitialState`` step:

.. code-block:: python

    class InitialState(Step):
        """
        A step for creating a mesh and initial condition for baroclinic channel
        test cases

        Attributes
        ----------
        resolution : str
            The resolution of the test case
        """
        def __init__(self, test_case, resolution):
            """
            Update the dictionary of step properties

            Parameters
            ----------
            test_case : compass.TestCase
                The test case this step belongs to

            resolution : str
                The resolution of the test case
            """
            super().__init__(test_case=test_case, name='initial_state')
            self.resolution = resolution

            for file in ['base_mesh.nc', 'culled_mesh.nc', 'culled_graph.info',
                         'ocean.nc']:
                self.add_output_file(file)

In this case, the argument ``resolution`` is passed in when the test case is
created, and is passed on to the step when it is created within the test case's
constructor.  Both the test case and the step save the resolution in an
attribute ``self.resolution`` of the class.  The developer of a test case can
add any number of parameters as attributes of each class in this way for later
use in the test case or step.  For example, the ``Default`` test case later
uses the resolution to call a shared ``configure()`` function:

.. code-block:: python

    def configure(self):
        """
        Modify the configuration options for this test case.
        """
        baroclinic_channel.configure(self.resolution, self.config)

The shared function uses the resolution to determine other config options:

.. code-block:: python

    def configure(resolution, config):
        """
        Modify the configuration options for one of the baroclinic test cases

        Parameters
        ----------
        resolution : str
            The resolution of the test case

        config : configparser.ConfigParser
            Configuration options for this test case
        """
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

Since all MPAS cores, test groups, test cases and steps are constructed as part
of listing, setting up, and cleaning up test cases and test suites, it is
important that these methods only perform a minimum of work to describe the
test case and should not directly download or read files, or perform any
complex computations.

Because of these considerations, the ``Step`` base class includes
infrastructure for identifying input and output files, and creating a "recipe"
for setting up namelist and streams files within ``__init__()`` without
actually downloading files, creating symlinks, parsing templates, or writing
files.  A step is allowed to:

  * call ``self.add_input_file()`` indicate files that should be symlinked from
    the ``compass`` package or from another step (in this or another test case)

  * call ``self.add_input_file()`` indicate files that should be downloaded
    from the LCRC server (or elsewhere)

  * call ``self.add_output_file()`` to indicate output files that will be
    produced by running the step

  * call ``self.add_namelist_file()`` or ``self.add_namelist_options()`` to
    add to the "recipe" for updating namelist options

  * call ``self.add_streams_file()`` to update the "recipe" for defining
    streams file during setup

These functions can also be called on the step from the test case's
constructor, e.g. ``step.add_namelist_file()``.  This might be convenient when
adding namelist options that are specific to the test case when you are using
the same step class for many test cases.

Namelist options always begin with a template produced when the MPAS model is
compiled.  Replacements are stored as keys and values in a python dictionary.
For convenience, they can be read from easy-to-read files similar to the
namelist files themselves but without sections:

.. code-block:: none

    config_time_integrator = 'split_explicit'
    config_dt = '02:00:00'
    config_btr_dt = '00:06:00'
    config_run_duration = '0000_06:00:00'
    config_hmix_use_ref_cell_width = .true.
    config_write_output_on_startup = .false.
    config_use_debugTracers = .true.

Such a file can be added within ``__init__()`` like this:

.. code-block:: python

    class ForwardStep(Step):
        def __init__(self, test_case, mesh, init, time_integrator, name='forward',
                     subdir=None, cores=None, min_cores=None, threads=None):
            ...

            self.add_namelist_file(
                'compass.ocean.tests.global_ocean', 'namelist.forward')
            if mesh.with_ice_shelf_cavities:
                self.add_namelist_file(
                    'compass.ocean.tests.global_ocean', 'namelist.wisc')

        self.add_streams_file(
            'compass.ocean.tests.global_ocean', 'streams.forward')

The namelist recipe can be updated with multiple calls to
``self.add_namelist_file()`` as in this example, or it can be altered with a
python dictionary of options by calling ``self.add_namelist_options()``.

Streams files are in XML format and are therefore a little bit trickier to
define.  The recipe is always defined by adding a streams file with
``self.add_streams_file()`` as in the example above.

A typical streams file might look like:

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

    <stream name="forcing_data"
            filename_template="forcing_data.nc"/>

    <stream name="mixedLayerDepthsOutput"/>

    </streams>

The file only has to provide attributes of a ``<stream>`` or
``<immutable_stream>`` tag if they differ from the defaults in the MPAS-model
template.  If ``<var>``, ``<var_struct>`` and/or ``<var_array>`` tags are
included in a stream, these will always replace the default contents of the
stream.  If none are provided, the default constants will be used.  There is
currently no mechanism for adding or removing ``vars``, etc. from a stream
because that seemed to be a feature that was rarely used or found to be useful
in the legacy COMPASS implementation.

configure()
~~~~~~~~~~~

Test cases do not have very many options for customization.  The main one is
customizing the config file that is shared between all steps in the test
case.  The framework sets up a ``self.config`` attribute for each test case,
and a test case can override the ``configure()`` method to modify these config
options. One way to update ``config`` is by calling
``compass.config.add_config()`` to add options from  a config file, typically
found in the package (directory) for the test case:

.. code-block:: python

    from compass.config import add_config
    from compass.io import symlink

    ...

    def configure(self):
        """
        Modify the configuration options for this test case
        """
        add_config(self.config, 'compass.landice.tests.enthalpy_benchmark.A',
                   'A.cfg', exception=True)

        with path('compass.landice.tests.enthalpy_benchmark', 'README') as \
                target:
            symlink(str(target), '{}/README'.format(self.work_dir))

Another way is to call the ``config.set()`` method:

.. code-block:: python

    def configure(self):
        """
        Modify the configuration options for this test case
        """
        # We want to visualize all test cases by default
        self.config.set('eismint2_viz', 'experiment', 'a, b, c, d, f, g')

Config options in ``config`` will be written to a config file in the work
directory called ``<test_case>.cfg``, where ``<test_case>`` is the name of the
test case.  These config options differ from parameters (such as ``resolution``
in the example above) that are attributes of test case's class in that config
options could be changed by a user before running the test case. Attributes of
the test case are not available in a format where users could easily alter them
and are unchanged between when the test case was set up and when it is run.

Typically, config options that are specific to a test case will go into a
config section with the same name as the test group.  In the example above,
we used a special section for visualization within the ``eismint2`` test group
called ``eismint2_viz``.  Developers can use whichever section name makes
sense as long as the section names are different from those used by the
framework such as ``[paths]`` and ``[parallel]``.

It is also possible to create symlinks within ``configure()``, e.g. to a README
file that applies to all steps in a test case, as shown above.

Steps do not have a ``configure()`` method because they share the same
``config`` with the other steps in the test case.  The idea is that it should
be relatively easy to change config options for all the steps in the test case
in one place.

setup()
~~~~~~~

Test cases do not have a ``setup()`` method because the only setting up they
typically include is to update config options in ``configure()``.  The step
may call ``self.add_input_file()``, ``self.add_output_file()``,
`self.add_namelist_file()``, ``self.add_namelist_options()`` or
``self.add_streams_file()`` to add inputs, outputs, and update the recipes for
namelist and streams files.  Any operations that require explicit references to
the work directory (i.e. ``self.work_dir``) or make use of config options
(``self.config``) have to happen in ``setup()`` rather than ``__init__()``
because neither of these attributes are defined within ``__init__()``.
Calls to ``self.add_model_as_input()``, which adds a symlink to the MPAS
model's executable, must also happen in ``setup()`` because the path to the
executable is a config option.

run
~~~

The ``run()`` method of a test case should, at a minimum, call the
base class' ``super().run()`` to run all the steps in the test case.
It can also:

  * read config options and use them to update the number of cores and threads
    that a step can use
  * perform validation of variables and timers

Here is a relatively complex example:

.. code-block:: python

    from compass.validate import compare_variables

    ...

    def run(self):
        """
        Run each step of the testcase
        """
        step = self.mesh_step
        config = self.config
        # get the these properties from the config options
        step.cores = config.getint('global_ocean', 'mesh_cores')
        step.min_cores = config.getint('global_ocean', 'mesh_min_cores')

        # run the step
        super().run()

        variables = ['xCell', 'yCell', 'zCell']
        compare_variables(variables, config, self.work_dir,
                          filename1='mesh/culled_mesh.nc')

The ``run()`` method of a step does the main "job" of the step so the
contents will very much depend on the purpose of the step.  Many steps will
use Metis to split the domain across processors and then call the MPAS model,
which can be done trivially with a call to ``run_model()``:

.. code-block:: python

    from compass.model import run_model

    ...

    def run(self):
        """
        Run this step of the testcase
        """
        run_model(self)

global ocean test group
~~~~~~~~~~~~~~~~~~~~~~~

The global ocean test group includes many other test cases and steps, and
is quite complex compared to idealized test cases, so may need the most
discussion.

The ``global_ocean`` test group works with variable resolution meshes,
requiring more significant numbers of parameters and even a function for
defining the resolution.  For this reason, it turned out to be more practical
to define each mesh as its own python package:

.. code-block:: none

  - compass/
    - ocean/
      - ocean.cfg
      - __init__.py
      - tests/
        - global_ocean
          ...
          - mesh
            - ec30to60
              - dynamic_adjustment
                - __init__.py
                - streams.template
              - __init__.py
              - ec30to60.cfg
              - namelist.split_explicit
            - qu240
              - dynamic_adjustment
                - __init__.py
                - streams.template
              - __init__.py
              - namelist.rk4
              - namelist.split_explicit
              - qu240.cfg
          ...

The ``mesh`` module includes an intermediate step class ``MeshStep`` for
defining meshes.  ``MeshStep`` includes a method ``build_cell_width_lat_lon()``
that child classes must override to define the mesh resolution.

To implement a new global mesh, one would need to define the resolution
in the ``__init__.py`` file:

.. code-block:: python

    import numpy as np

    from compass.ocean.tests.global_ocean.mesh.mesh import MeshStep


    class QU240Mesh(MeshStep):
        """
        A step for creating QU240 and QUwISC240 meshes
        """
        def __init__(self, test_case, mesh_name, with_ice_shelf_cavities):
            """
            Create a new step

            Parameters
            ----------
            test_case : compass.TestCase
                The test case this step belongs to

            mesh_name : str
                The name of the mesh

            with_ice_shelf_cavities : bool
                Whether the mesh includes ice-shelf cavities
            """

            super().__init__(test_case, mesh_name, with_ice_shelf_cavities,
                             package=self.__module__,
                             mesh_config_filename='qu240.cfg')

        def build_cell_width_lat_lon(self):
            """
            Create cell width array for this mesh on a regular latitude-longitude
            grid

            Returns
            -------
            cellWidth : numpy.array
                m x n array of cell width in km

            lon : numpy.array
                longitude in degrees (length n and between -180 and 180)

            lat : numpy.array
                longitude in degrees (length m and between -90 and 90)
            """
            dlon = 10.
            dlat = dlon
            constantCellWidth = 240.

            nlat = int(180/dlat) + 1
            nlon = int(360/dlon) + 1
            lat = np.linspace(-90., 90., nlat)
            lon = np.linspace(-180., 180., nlon)

            cellWidth = constantCellWidth * np.ones((lat.size, lon.size))
            return cellWidth, lon, lat

A developer would also need to define any namelist options for forward runs that
are specific to this mesh (once for RK4 and once for split-explicit if both
time integrators are supported):

.. code-block:: none

    config_time_integrator = 'split_explicit'
    config_dt = '00:30:00'
    config_btr_dt = '00:01:00'
    config_run_duration = '0000_01:30:00'
    config_mom_del2 = 1000.0
    config_mom_del4 = 1.2e11
    config_hmix_scaleWithMesh = .true.
    config_use_GM = .true.

The developer would define config options to do with the number of cores and
vertical layers (both of which the user could change at runtime) as well as
metadata to include in the output files:

.. code-block:: cfg

    # Options related to the vertical grid
    [vertical_grid]

    # the type of vertical grid
    grid_type = 60layerPHC


    # options for global ocean testcases
    [global_ocean]

    ## config options related to the initial_state step
    # number of cores to use
    init_cores = 36
    # minimum of cores, below which the step fails
    init_min_cores = 8
    # maximum memory usage allowed (in MB)
    init_max_memory = 1000
    # maximum disk usage allowed (in MB)
    init_max_disk = 1000

    ## config options related to the forward steps
    # number of cores to use
    forward_cores = 128
    # minimum of cores, below which the step fails
    forward_min_cores = 36
    # maximum memory usage allowed (in MB)
    forward_max_memory = 1000
    # maximum disk usage allowed (in MB)
    forward_max_disk = 1000

    ## metadata related to the mesh
    # the prefix (e.g. QU, EC, WC, SO)
    prefix = EC
    # a description of the mesh and initial condition
    mesh_description = MPAS Eddy Closure mesh for E3SM version ${e3sm_version} with
                       enhanced resolution around the equator (30 km), South pole
                       (35 km), Greenland (${min_res} km), ${max_res}-km resolution
                       at mid latitudes, and ${levels} vertical levels
    # E3SM version that the mesh is intended for
    e3sm_version = 2
    # The revision number of the mesh, which should be incremented each time the
    # mesh is revised
    mesh_revision = 3
    # the minimum (finest) resolution in the mesh
    min_res = 30
    # the maximum (coarsest) resolution in the mesh, can be the same as min_res
    max_res = 60
    # The URL of the pull request documenting the creation of the mesh
    pull_request = <<<Missing>>>

Finally, the developer would implement the ``dynamical_adjustment`` test case,
using one of the existing spin-up test cases as a kind of a template.  These
test cases descend from the ``DynamicalAdjustment`` class, which itself
descends from ``TestCase``.

.. code-block:: python

    from compass.ocean.tests.global_ocean.dynamic_adjustment import \
        DynamicAdjustment
    from compass.ocean.tests.global_ocean.forward import ForwardStep


    class QU240DynamicAdjustment(DynamicAdjustment):
        """
        A test case performing dynamic adjustment (dissipating fast-moving waves)
        from an initial condition on the QU240 MPAS-Ocean mesh
        """

        def __init__(self, test_group, mesh, init, time_integrator):
            """
            Create the test case

            Parameters
            ----------
            test_group : compass.ocean.test.global_ocean.GlobalOcean
                The global ocean test group that this test case belongs to

            mesh : compass.ocean.tests.global_ocean.mesh.Mesh
                The test case that produces the mesh for this run

            init : compass.ocean.tests.global_ocean.init.Init
                The test case that produces the initial condition for this run

            time_integrator : {'split_explicit', 'RK4'}
                The time integrator to use for the forward run
            """
            restart_times = ['0001-01-02_00:00:00', '0001-01-03_00:00:00']
            restart_filenames = [
                'restarts/rst.{}.nc'.format(restart_time.replace(':', '.'))
                for restart_time in restart_times]

            super().__init__(test_group=test_group, mesh=mesh, init=init,
                             time_integrator=time_integrator,
                             restart_filenames=restart_filenames)

            module = self.__module__

            # first step
            step_name = 'damped_adjustment_1'
            step = ForwardStep(test_case=self, mesh=mesh, init=init,
                               time_integrator=time_integrator, name=step_name,
                               subdir=step_name)

            namelist_options = {
                'config_run_duration': "'00-00-01_00:00:00'",
                'config_Rayleigh_friction': '.true.',
                'config_Rayleigh_damping_coeff': '1.0e-4'}
            step.add_namelist_options(namelist_options)

            stream_replacements = {
                'output_interval': '00-00-01_00:00:00',
                'restart_interval': '00-00-01_00:00:00'}
            step.add_streams_file(module, 'streams.template',
                                  template_replacements=stream_replacements)

            step.add_output_file(filename='../{}'.format(restart_filenames[0]))
            self.add_step(step)

            # final step
            step_name = 'simulation'
            step = ForwardStep(test_case=self, mesh=mesh, init=init,
                               time_integrator=time_integrator, name=step_name,
                               subdir=step_name)

            namelist_options = {
                'config_run_duration': "'00-00-01_00:00:00'",
                'config_do_restart': '.true.',
                'config_start_time': "'{}'".format(restart_times[0])}
            step.add_namelist_options(namelist_options)

            stream_replacements = {
                'output_interval': '00-00-01_00:00:00',
                'restart_interval': '00-00-01_00:00:00'}
            step.add_streams_file(module, 'streams.template',
                                  template_replacements=stream_replacements)

            step.add_input_file(filename='../{}'.format(restart_filenames[0]))
            step.add_output_file(filename='../{}'.format(restart_filenames[1]))
            self.add_step(step)

Whew! That was a lot, thanks for bearing with me.

.. _imp_shared_code:

Implementation: Shared code
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/04/13

Contributors: Xylar Asay-Davis

The package includes myriad examples of code sharing so I will highlight a few.

compass framework
~~~~~~~~~~~~~~~~~

The ``compass`` framework (classes, modules and packages not in the
MPAS-core-specific packages) has a lot of code that is shared across existing
test cases and could be very useful for future ones.

Most of the framework currently has roughly the same functionality as legacy
COMPASS, but it has been broken into more modules that make it clear what
functionality each contains, e.g. ``compass.namelists`` and ``compass.streams``
are for manipulating namelists and streams files, respectively;
``compass.io`` has functionality for downloading files from LCRC and creating
symlinks; and ``compass.validation`` can be used to ensure that variables are
bit-for-bit identical between steps or when compared with a baseline, and to
compare timers with a baseline.  This functionality was all included in 4 very
long scripts in legacy COMPASS.

One example that doesn't have a clear analog in legacy COMPASS is the
``compass.parallel`` module.  It contains two functions:
``get_available_cores_and_nodes()``, which can find out the number of total
cores and nodes available for running steps.

within an MPAS core
~~~~~~~~~~~~~~~~~~~

Legacy COMPASS shares functionality with an MPAS core by having scripts at the MPAS core
level that are linked within test cases and which take command-line arguments
that function roughly the same way as function arguments.  But these scripts
are not able to share any code between them unless it is from ``mpas_tools``
or another external package.

Am MPAS core in ``compass`` could, theoretically, build out functionality as complex
as in MPAS-Model if desired.  Indeed, it is my ambition to gradually replace
"init mode" in MPAS-Ocean with equivalent python functionality, starting with
simpler test cases.  This has already been accomplished for the 3 idealized
ocean test cases included in the proposed design.

The current shared functionality in the ``ocean`` MPAS core includes:

  * ``compass.ocean.namelists`` and ``compass.ocean.streams``: namelist
    replacements and streams that are similar to MPAS-core-level templates in legacy
    COMPASS.  Current templates are for adjusting sea surface height in
    ice-shelf cavities, and outputting variables related to frazil and
    land-ice fluxes,

  * ``compass.ocean.suites``: the ocean test suites

  * ``compass.ocean.vertical``: supports for 1D vertical coordinates and the 3D
    z* coordinate.

  * ``compass.ocean.iceshelf``: computes sea-surface height and
    land-ice pressure, and adjusts them to match one another

  * ``compass.ocean.particles``: initialization of particles

  * ``compass.ocean.plot``: plots initial state and 1D vertical grid


within a test group
~~~~~~~~~~~~~~~~~~~

So far, the most common type of shared code within test groups are modules
defining steps that are used in multiple test cases.  For example, the
``BaroclinicChannel`` test group uses shared modules to define the
``InitialState`` and ``Forward`` steps of each test case.  Configurations
also often include namelist and streams files with replacements to use across
test cases.

In addition to shared steps, the ``GlobalOcean`` test group includes
some additional shared modules:

  * ``compass.ocean.tests.global_ocean.mesh``: defines properties of each
    global mesh (as well as a ``Mesh`` test case)

  * ``compass.ocean.tests.global_ocean.metadata``: determines the values of a
    set of metadata related to the E3SM mesh name, initial condition, conda
    environment, etc. that are added to nearly all ``global_ocean`` NetCDF
    output

  * ``compass.ocean.tests.global_ocean.subdir``: helps with maintaining the
    slightly complex subdirectory structure within ``global_ocean`` test cases.

The shared code in ``global_ocean`` could easily define hundreds of different
test cases using the QU240 (or QUwISC240) mesh.  This is possible because
the same conceptual test (e.g. restart) can be defined:

  * with or without ice-shelf cavities

  * with the PHC or EN4 1900 initial conditions

  * with the RK4 or split-explicit time integrators

In practice, this is overkill and many of these variants will never be used so
they are not currently made available.

Also, I want to note that it is because of this flexibility that I added an
RK4 restart test, which failed and showed us that there was a recent problem
with RK4 restarts (https://github.com/MPAS-Dev/MPAS-Model/issues/777).

within a test case
~~~~~~~~~~~~~~~~~~

There aren't too many cases so far where reuse of code within a test case is
particularly useful.  The main way this currently occurs is when the same
module for a step gets used multiple times within a test case.  For example,
the `baroclinic_channel.rpe_test` test case uses the same forward run with
5 different viscosities.

.. _imp_shared_config:

Implementation: Shared configuration options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/04/13

Contributors: Xylar Asay-Davis

As discussed in :ref:`alg_shared_config`, the proposed design builds up the
config file for a given test from several sources.  Some of the config options
are related to setting up the test case (e.g. the locations of cached data
files) but the majority are related to running the steps of the test case.

During setup of a test case and its steps, The config file is assembled from
a number of sources.  Before the ``configure()`` method of the test case is
called, config options come from:

* the default config file, ``default.cfg``, which sets a few options related to
  downloading files during setup (whether to download and whether to check the
  size of files already downloaded)

* the machine config file (using ``machines/default.cfg`` if none was
  specified) with information on the parallel system and (typically) the paths
  to cached data files

* the MPAS core's config file.  For the MPAS-Ocean core, this sets default paths to
  the MPAS model build (including the namelist templates).  It uses "extended
  interpolation" in the config file to use config opitons within other config
  options, e.g. ``model = ${paths:mpas_model}/ocean_model``.

* the test group's config file if one is found.  For idealized
  test groups, these include config options that were previously init-mode
  namelist options.  For ``global_ocean``, these include defaults for mesh
  metadata (again using "extended interpolation"); the default number of cores
  and other resource usage for mesh, init and forward steps; and options
  related to files created for E3SM initial conditions.

Then, the ``configure()`` method is called on the test case itself.  All of
the current ocean test cases first call a shared ``configure()`` function at
the test group level, e.g.:

.. code-block:: python

    from compass.ocean.tests.global_ocean.configure import configure_global_ocean

    ...

    def configure(self):
        """
        Modify the configuration options for this test case
        """
        configure_global_ocean(test_case=self, mesh=self.mesh)

where ``configure_global_ocean()`` is:

.. code-block:: python

    from compass.config import add_config


    def configure_global_ocean(test_case, mesh, init=None):
        """
        Modify the configuration options for this test case

        Parameters
        ----------
        test_case : compass.TestCase
            The test case to configure

        mesh : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case that produces the mesh for this run

        init : compass.ocean.tests.global_ocean.init.Init, optional
            The test case that produces the initial condition for this run
        """
        config = test_case.config
        mesh_step = mesh.mesh_step
        add_config(config, mesh_step.package, mesh_step.mesh_config_filename,
                   exception=True)

        if mesh.with_ice_shelf_cavities:
            config.set('global_ocean', 'prefix', '{}wISC'.format(
                config.get('global_ocean', 'prefix')))

        add_config(config, test_case.__module__, '{}.cfg'.format(test_case.name),
                   exception=False)

        # add a description of the initial condition
        if init is not None:
            initial_condition = init.initial_condition
            descriptions = {'PHC': 'Polar science center Hydrographic '
                                   'Climatology (PHC)',
                            'EN4_1900':
                                "Met Office Hadley Centre's EN4 dataset from 1900"}
            config.set('global_ocean', 'init_description',
                       descriptions[initial_condition])

        # a description of the bathymetry
        config.set('global_ocean', 'bathy_description',
                   'Bathymetry is from GEBCO 2019, combined with BedMachine '
                   'Antarctica around Antarctica.')

        if mesh.with_ice_shelf_cavities:
            config.set('global_ocean', 'wisc_description',
                       'Includes cavities under the ice shelves around Antarctica')

In this case, a config options related to the mesh are loaded, then those
related to ice-shelf cavities (if they are included in the mesh), then those
specific to the test case itself.

Although none of the existing ocean test cases do so, further changes could be
made to the config file beyond those at the test group level.  Indeed,
there is no reason a test case cannot just set its config options or read them
from a file without calling a test-group-level function, this is just a
convenience.

Finally, config options are taken from the user's config file if one was passed
in with the ``-f`` or ``--config_file`` commandline flag:

.. code-block:: bash

    python -m compass setup -n 10 11 12 13 14 \
        -w ~/scratch/mpas/test_baroclinic_channel -m anvil -f ocean.cfg

    python -m compass suite -s -c ocean -t nightly -m anvil -f ocean.cfg \
        -w ~/scratch/mpas/test_nightly

A typical config file resulting from all of this looks like:

.. code-block:: cfg

    [download]
    download = True
    check_size = False
    verify = True

    [parallel]
    system = single_node
    parallel_executable = mpirun
    cores_per_node = 8
    threads = 8

    [paths]
    mpas_model = /home/xylar/code/mpas-work/compass/compass_1.0/MPAS-Model/ocean/develop
    mesh_database = /home/xylar/data/mpas/meshes
    initial_condition_database = /home/xylar/data/mpas/initial_conditions
    bathymetry_database = /home/xylar/data/mpas/bathymetry_database

    [namelists]
    forward = /home/xylar/code/mpas-work/compass/compass_1.0/MPAS-Model/ocean/develop/default_inputs/namelist.ocean.forward
    init = /home/xylar/code/mpas-work/compass/compass_1.0/MPAS-Model/ocean/develop/default_inputs/namelist.ocean.init

    [streams]
    forward = /home/xylar/code/mpas-work/compass/compass_1.0/MPAS-Model/ocean/develop/default_inputs/streams.ocean.forward
    init = /home/xylar/code/mpas-work/compass/compass_1.0/MPAS-Model/ocean/develop/default_inputs/streams.ocean.init

    [executables]
    model = /home/xylar/code/mpas-work/compass/compass_1.0/MPAS-Model/ocean/develop/ocean_model

    [ssh_adjustment]
    iterations = 10

    [global_ocean]
    mesh_cores = 1
    mesh_min_cores = 1
    mesh_max_memory = 1000
    mesh_max_disk = 1000
    init_cores = 4
    init_min_cores = 1
    init_max_memory = 1000
    init_max_disk = 1000
    init_threads = 1
    forward_cores = 4
    forward_min_cores = 1
    forward_threads = 1
    forward_max_memory = 1000
    forward_max_disk = 1000
    add_metadata = True
    prefix = QU
    mesh_description = MPAS quasi-uniform mesh for E3SM version ${e3sm_version} at
        ${min_res}-km global resolution with ${levels} vertical
        level
    bathy_description = Bathymetry is from GEBCO 2019, combined with BedMachine Antarctica around Antarctica.
    init_description = <<<Missing>>>
    e3sm_version = 2
    mesh_revision = 1
    min_res = 240
    max_res = 240
    max_depth = autodetect
    levels = autodetect
    creation_date = autodetect
    author = Xylar Asay-Davis
    email = xylar@lanl.gov
    pull_request = https://github.com/MPAS-Dev/compass/pull/28

    [files_for_e3sm]
    enable_ocean_initial_condition = true
    enable_ocean_graph_partition = true
    enable_seaice_initial_condition = true
    enable_scrip = true
    enable_diagnostics_files = true
    comparisonlatresolution = 0.5
    comparisonlonresolution = 0.5
    comparisonantarcticstereowidth = 6000.
    comparisonantarcticstereoresolution = 10.
    comparisonarcticstereowidth = 6000.
    comparisonarcticstereoresolution = 10.

    [vertical_grid]
    grid_type = tanh_dz
    vert_levels = 16
    bottom_depth = 3000.0
    min_layer_thickness = 3.0
    max_layer_thickness = 500.0

Unfortunately, all comments are lost in the process of combining config
options.  Comments are not parsed by ``ConfigParser``, and there is not a
standard for which comments are associated with which options.  So users
would need to search through the code for the original config or look through
the documentation to know what the config options are used for.  In the future,
we could consider implementing our own customized version of ``ConfigParser``
that preserves comments.

Implementation: Ability specify/modify core counts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/01/16

Contributors: Xylar Asay-Davis

The ``Step`` class includes two attributes, ``cores`` and ``min_cores``,
which should be set by the time the ``run()`` method gets called.
``cores`` is the target number of cores for the step and ``min_cores`` is the
minimum number of cores, below which the test case would probably fail. Before
a step is run, ``compass`` finds out how many total cores are available to run
the test. If the number is below ``self.min_cores``, an error is raised.
Otherwise, the test case will run with ``self.cores`` or the number of
available cores, whichever is lower.

The idea is that the same test case could be run efficiently on one or more
nodes of an HPC machine but could also be run on a laptop or desktop if the
minimum number of required cores is reasonable.

There are a variety of ways that the ``cores`` and ``min_cores`` attributes can
be set.  The most straightforward is to set them by calling the base class'
``__init__()``.  They could be passed through from calls to the child class'
``__init__()``:

.. code-block:: python

    def __init__(self, test_case, cores=1, min_cores=None):
        """
        Create a new test case

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        cores : int, optional
            the number of cores the step would ideally use.  If fewer cores
            are available on the system, the step will run on all available
            cores as long as this is not below ``min_cores``

        min_cores : int, optional
            the number of cores the step requires.  If the system has fewer
            than this number of cores, the step will fail
        """
        if min_cores is None:
            min_cores = cores
        super().__init__(test_case=test_case, name='forward', cores=cores,
                         min_cores=min_cores)

or just hard coded:

.. code-block:: python

    def __init__(self, test_case):
        """
        Create a new test case

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to
        """
        if min_cores is None:
            min_cores = cores
        super().__init__(test_case=test_case, name='forward', cores=4,
                         min_cores=1)

Or they could be defined later in the process, at setup or in the test
case's ``run()`` method.  (Defining them in the step's ``run()`` is too late,
since the number of cores to actually use is determined before this call is
made.)  In ``global_ocean``, the number of cores and minimum cores are set
using config options.  Since users could modify these before calling the
``run.py`` script, they are parsed in the test case's ``run()`` function
before ``run_steps()`` is called:

.. code-block:: python

    def run(self):
        """
        Run each step of the testcase
        """
        config = self.config
        # get the these properties from the config options
        for step_name in self.steps_to_run:
            step = self.steps[step_name]
            # get the these properties from the config options
            step.cores = config.getint('global_ocean', 'forward_cores')
            step.min_cores = config.getint('global_ocean', 'forward_min_cores')
            step.threads = config.getint('global_ocean', 'forward_threads')

        # run the steps
        super().run()

The ``steps_to_run`` attribute of the test case is a list of the subset of the
steps that were actually requested to run from the test case.  For example,
if you run a step on its own, it still actually runs the test case but only
requesting that one step.  Some test cases include steps that are not run by
default, and this is specified by passing ``run_by_default=False`` as an
argument to ``self.add_step()`` when adding the step in the test case's
constructor:

.. code-block:: python

    def __init__(self, test_group, mesh_type):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.dome.Dome
            The test group that this test case belongs to

        mesh_type : str
            The resolution or tye of mesh of the test case
        """
        name = 'smoke_test'
        self.mesh_type = mesh_type
        subdir = '{}/{}'.format(mesh_type, name)
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        self.add_step(
            SetupMesh(test_case=self, mesh_type=mesh_type))
        self.add_step(
            RunModel(test_case=self, cores=4, threads=1, mesh_type=mesh_type))
        step = Visualize(test_case=self, mesh_type=mesh_type)
        self.add_step(step, run_by_default=False)


.. _imp_machine:

Implementation: Machine-specific data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/04/13

Contributors: Xylar Asay-Davis


Machine-specific configuration options are in a set of config files under
``compass/machines``.  As an example, the config file for Anvil looks like:

.. code-block:: cfg

    # The paths section describes paths that are used within the ocean core test
    # cases.
    [paths]

    # The root to a location where the mesh_database, initial_condition_database,
    # and bathymetry_database for MPAS-Ocean will be cached
    ocean_database_root = /lcrc/group/e3sm/public_html/mpas_standalonedata/mpas-ocean

    # The root to a location where the mesh_database and initial_condition_database
    # for MALI will be cached
    landice_database_root = /lcrc/group/e3sm/public_html/mpas_standalonedata/mpas-albany-landice

    # the path to the base conda environment where compass environments have
    # been created
    compass_envs = /lcrc/soft/climate/e3sm-unified/base


    # The parallel section describes options related to running tests in parallel
    [parallel]

    # parallel system of execution: slurm or single_node
    system = slurm

    # whether to use mpirun or srun to run the model
    parallel_executable = srun

    # cores per node on the machine
    cores_per_node = 36

    # the number of multiprocessing or dask threads to use
    threads = 18

It is likely that ``cores_per_node`` can be detected using a Slurm command and
doesn't need to be supplied.  This is something I have not fully explored yet.

The ``threads`` option is not currently used and would also need to be
explored.

Additional config options are needed to support automatically generating
job scripts, but this will be left for future work.

The available machines are listed with:

.. code-block:: bash

    python -m compass list --machine

.. code-block:: none

    Machines:
       anvil
       default
       chrysalis
       compy

When setting up a test case or test suite, the ``--machine`` or ``-m`` flag
is used to specify the machine.


.. _imp_dir_struct:

Implementation: Looser, more flexible directory structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/04/13

Contributors: Xylar Asay-Davis

Test cases (and steps) in ``compass`` are uniquely defined by their relative
paths within the work directory.  The first two subdirectories in this path
must be the name of the MPAS core and of the test group.  The names and
organization beyond that are quite flexible.  Steps are expected to be nested
somewhere within test cases but there is no restriction on the number of levels
of subdirectories or their meaning beyond that of the test group.

The idealized ocean test groups that have been implemented so far and the
example test groups use the same organization as in legacy COMPASS:

.. code-block:: none

    mpas_core/test_group/resolution/testcase/step

For example:

.. code-block:: none

    ocean/baroclinic_channel/10km/default/initial_state

But the ``global_ocean`` test group takes advantage of the new
flexibility.  Here are the directories for test cases using the QU240 mesh:

.. code-block:: none

  27: ocean/global_ocean/QU240/mesh
  28: ocean/global_ocean/QU240/PHC/init
  29: ocean/global_ocean/QU240/PHC/performance_test
  30: ocean/global_ocean/QU240/PHC/restart_test
  31: ocean/global_ocean/QU240/PHC/decomp_test
  32: ocean/global_ocean/QU240/PHC/threads_test
  33: ocean/global_ocean/QU240/PHC/analysis_test
  34: ocean/global_ocean/QU240/PHC/daily_output_test
  35: ocean/global_ocean/QU240/PHC/dynamic_adjustment
  36: ocean/global_ocean/QU240/PHC/files_for_e3sm
  37: ocean/global_ocean/QU240/PHC/RK4/performance_test
  38: ocean/global_ocean/QU240/PHC/RK4/restart_test
  39: ocean/global_ocean/QU240/PHC/RK4/decomp_test
  40: ocean/global_ocean/QU240/PHC/RK4/threads_test
  41: ocean/global_ocean/QU240/EN4_1900/init
  42: ocean/global_ocean/QU240/EN4_1900/performance_test
  43: ocean/global_ocean/QU240/EN4_1900/dynamic_adjustment
  44: ocean/global_ocean/QU240/EN4_1900/files_for_e3sm

As in legacy COMPASS, there is a subdirectory for the mesh.  In the proposed
design, there is a ``mesh`` test case with a single ``mesh`` step within that
subdirectory.  The mesh constructed and culled within that test case serves
as the starting point for all other test cases using the mesh.

Then, there are 3 different subdirectories for variants of the initial
condition: WOA23, PHC or EN4_1900.  Each of these subdirectories has own
``init`` test case that creates the initial condition.  The results of this
test case are then used in all other steps within the subdirectory for that
initial condition.

Each remaining test case includes one or more forward model runs, or uses the
results of such a run.  Since the forward model can be run with either the
split-explicit or the RK4 time integrator, variants of many test cases are
supported with each time integrator.  It is important that these are
conceptually separate test cases because we use both the split-explicit and
the RK4 versions of many test cases in our test suites.  Each requires a set of
corresponding namelist options and modifications to streams, so it is also not
trivial for a user to switch between the two time integrators simply by
manually modifying the test case at runtime.  We treat the split-explicit
time integrator as the default and put tests with RK4 in an additional ``RK4``
subdirectory.


.. _imp_docs:

Implementation: User- and developer-friendly documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/04/13

Contributors: Xylar Asay-Davis


The documentation is still very much a work in progress and may be added with
a separate pull request so that commits related to the infrastructure don't get
intermixed with those for documentation.

Documentation will continue to be generated automatically with Azure Pipelines
using sphinx, as is the case for this design doc.

The legacy COMPASS documentation will be renamed with "legacy" added to its
titles (e.g. "Legacy User's Guide") and will be included at the end of the
table of contents.

The latest version of the test documentation is available in the branch:
https://github.com/xylar/compass/tree/compass_1.0_docs
and for browsing at the URL:
https://mpas-dev.github.io/compass/test/index.html


.. _imp_parallel:

Implementation: Considerations related to running test cases in parallel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/04/13

Contributors: Xylar Asay-Davis


While an implementation of test-case parallelism will be left to a future
design document and implementation, several parts of the current ``compass``
design and implementation were made with this work in mind.


cores, max_memory and max_disk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``Step`` base class keeps track of not only the number of cores (and
threads) used but also the maximum allowed memory and disk.  While the latter
two are not currently used for anything and the values are just placeholders,
they are expected to be useful for Parsl ``WorkerQueues``.

inputs and outputs
~~~~~~~~~~~~~~~~~~

An effort has been made to be thorough about providing an absolute path to the
inputs and outputs of each step.  These are currently verified to make sure
inputs are present before running a step and outputs are present after. We
expect them to also be useful when we use Parsl to determine dependencies
between steps and to figure out which can run in parallel with one another.


.. _imp_res:

Implementation: Resolution can be a test case parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/04/13

Contributors: Xylar Asay-Davis


This was discussed in :ref:`imp_dir_struct`.  For all of the ocean and many
of the land-ice test groups, either the resolution or the name of the mesh
(which implicitly includes the resolution) is an argument to test case's and
step's constructors. Nearly all test cases use that resolution or mesh name as
a subdirectory within the relative path of the test case.  So far, no
convergence tests have been added where resolution is a parameter that varies
across steps in a test case but the ``rpe_test`` test case of the
``baroclinic_channel`` includes viscosity as a parameter that varies across
steps, and resolution is expected to be easy to use in the same way for future
test cases.


.. _imp_alter_code:

Implementation: Test case code is easy to alter and rerun
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/04/13

Contributors: Xylar Asay-Davis


When test cases and suites are set up from a local repository (and not the
conda package from a conda environment), local symlinks to the ``compass``
directory are created.  These links seem to provide and easy method for
altering code and having it affect test cases and steps immediately without
the need to build a conda package or a conda environment, or even to rerun
``python -m compass setup`` in most cases.


.. _imp_premade_ic:

Implementation: Support for pre-made initial condition files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/01/16

Contributors: Xylar Asay-Davis, Mark Petersen


This work has not been included in any of the test cases that are part of the
current implementation.  Nothing in the implementation should preclude adding
this capability later on.


.. _imp_batch:

Implementation: Easy batch submission
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/01/16

Contributors: Xylar Asay-Davis, Mark Petersen


Batch scripts are not yet generated automatically as part of setting up a
test case.  Additional machine-specific config options will be needed to make
this possible. This capability will be part of a future design.  Nothing in the
current implementation should preclude adding this capability later on.
Indeed, it likely wouldn't be to much work.


Testing
-------

.. _test_easy:

Testing: Make test cases easy to understand, modify and create
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/04/13

Contributors: Xylar Asay-Davis, Luke Van Roekel


Given limited time, the reviewers will not attempt to implement any new MPAS
cores or test groups as part of there reviews.  However, in the near future,
Luke Van Roekel has agreed to attempt to implement a test group
("single-column") and its test cases and steps as a test of the ease of
understanding, modifying and creating test cases.  Mark Petersen will add a
new shallow-water MPAS core.  Matt Hoffman will add new test cases as he has
time and interest down the road.

.. _test_shared_code:

Testing: Shared code
^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/04/13

Contributors: Xylar Asay-Davis


All of the test cases in the proposed implementation use shared code.  Nearly
all of the 86 test cases have been tested including those in all ocean and
land-ice suites except the EC30to60 and ECwISC30to60.  The 10 km RPE test for
the ``baroclinic_channel`` test group has also been run successfully.
The higher resolution versions of that test case have not yet been tested.

So far, there is no indication of problems with shared code, but this is
something of a subjective thing to test, beyond the proof of concept that code
can, indeed, be shared.


.. _test_shared_options:

Testing: Shared configuration options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/01/16

Contributors: Xylar Asay-Davis


All test cases include a config file and most test cases make use of config
options from that file.

I verified that altering the following config options in the
``ocean/global_ocean/QU240/PHC/init`` test case:

.. code-block:: cfg

    [global_ocean]
    init_cores = 2

    [vertical_grid]
    vert_levels = 32
    max_layer_thickness = 250.0

Did indeed use 2 MPI tasks to produce an initial condition with 32 vertical
levels, and a target maximum layer thickness of 250.0 m (actual was 245.35 m).

It would be nearly impossible to test altering all parameters to see if they
have the intended effect, so this will not be part of this testing.


.. _test_core_count:

Testing: Ability specify/modify core counts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/01/16

Contributors: Xylar Asay-Davis


This was included in :ref:`test_shared_options`.


.. _test_machine_data:

Testing: Machine-specific data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/01/14

Contributors: Xylar Asay-Davis


I ran the ocean nightly test suite on Anvil, providing ``-m anvil`` and no
user config file.  This worked successfully and no cached files were
downloaded, meaning the cache directories were found successfully via Anvil's
config file.  I verified that the number of available cores and nodes in my job
were successfully detected via Slurm commands.


.. _test_dir_struct:

Testing: Looser, more flexible directory structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/04/13

Contributors: Xylar Asay-Davis


Testing of the ocean nightly suite includes tests of the flexible directory
structure because it uses the ``global_ocean`` test group.  More to the
point, this capability has been tested by showing that test cases can be
implemented using the flexible directory structure.


.. _test_docs:

Testing: User- and developer-friendly documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/04/13

Contributors: Xylar Asay-Davis


Reviewers have been asked to run test cases and suites with the documentation,
which is still evolving.  Users and developers will be asked to run test cases
and suites with the documentation and to add new test cases.  In the near
future, the documentation will be declared "good enough for now" and will be
merged with the intention to update it on an ongoing basis.


.. _test_parallel:

Testing: Considerations related to running test cases in parallel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/01/16

Contributors: Xylar Asay-Davis, Matt Hoffman


This has not yet been implemented so will be tested as part of a later design.


.. _test_res:

Testing: Resolution can be a test case parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/01/16

Contributors: Xylar Asay-Davis, Mark Petersen


Resolution is a parameter in many existing test cases.  No test case has yet
been implemented that includes multiple steps with different resolutions so
no testing of such a test case is possible at this time.


.. _test_alter_code:

Testing: Test case code is easy to alter and rerun
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/01/16

Contributors: Xylar Asay-Davis, Mark Petersen


Xylar and Mark have both demonstrated that it is easy to modify code and rerun
test cases without additional work because of the symlinks to the ``compass``
directory.


.. _test_premade_ic:

Testing: Support for pre-made initial condition files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/01/16

Contributors: Xylar Asay-Davis, Mark Petersen


This was not yet implemented so no testing was performed.


.. _test_batch:

Testing: Easy batch submission
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/01/16

Contributors: Xylar Asay-Davis, Mark Petersen


This was not yet implemented so no testing was performed.

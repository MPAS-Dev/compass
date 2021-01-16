
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
(``core/configuration/resoltuion/testcase/step``) is too rigid for many
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
(``core/configuration/resoltuion/testcase/step``) is too rigid for many
applications.  Some test cases (e.g. convergence tests) require multiple
resolutions within the test case.  Some configurations would prefer to sort
test cases based on another parameter or property besides resolution.  It would
be convenient if the directory structure could be more flexible, depending on
the needs of a given configuration and test case.  Even so, it is important that
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
create a new "configuration" with one or more "test cases", each made up of
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
for individual configurations like ``build_base_mesh.py``. This works for more
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

Date last modified: 2021/01/14

Contributors: Xylar Asay-Davis


The proposed solution would be to write test cases as Python packages made up
of modules and functions within a larger ``compass`` package.  A test case will
have separate functions to collect information on the test case (the equivalent
of parsing ``config_driver.xml`` in the current implementation), configure it
by adding test-case-specific config options, and run the default steps.  Each
step of a test case (equivalent to the other ``config_*.xml`` files) will be
a module (possibly shared between test cases) that implements functions for
collecting information on the step (equivalent to parsing ``config_*.xml``),
setting up the step (downloading files, making symlinks, creating namelist and
streams files), and running the step.  A balance will have to be struck between
code reusability and readability within each configuration (a set of test cases).

Readability would be improved by using Jinja2 templates for code generation,
rather than via string manipulation within python scripts as is the case in the
current COMPASS implementation:

.. code-block:: python

    #!/usr/bin/env python
    import pickle
    import configparser

    from {{ testcase.module }} import {{ testcase.run }} as run
    from mpas_tools.logging import LoggingContext


    def main():
        with open('testcase_{{ testcase.name }}.pickle', 'rb') as handle:
            testcase = pickle.load(handle)
        testcase['steps_to_run'] = ['{{ step.name }}']
        testcase['new_step_log_file'] = False

        with open('{{ step.name }}.pickle', 'rb') as handle:
            step = pickle.load(handle)

        config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())
        config.read('{{ step.config }}')

        # start logging to stdout/stderr
        test_name = step['path'].replace('/', '_')
        with LoggingContext(name=test_name) as logger:
            test_suite = dict()
            run(testcase, test_suite, config, logger)


    if __name__ == '__main__':
        main()


A Jinja2 template uses curly braces (e.g. ``{{ testcase.module }}``) to indicate
where an element of the template will be replaced by a python variable or
dictionary value.  In this example, ``{{ testcase.module }}`` will be replaced with
the contents of ``testcase['module']`` in the python code, and similarly for other
replacements in the template.  Other than the replacements, the code can be read
as normal, in contrast to the existing approach of python scripts that define
other python scripts via a series of string formatting statements.

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
and steps within a configuration without some degree of increased complexity.
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

Date last modified: 2021/01/14

Contributors: Xylar Asay-Davis


By organizing both the test cases themselves and shared framework code into a
``compass`` Python package, code reuse and organization should be greatly
simplified.

The organization of the package will be as follows:

.. code-block:: none

  - compass/
    - <core>/
      - <core>.cfg
      - <core_framework_module>.py
      - <core_framework_package>/
      - tests/
        - <configuration>/
          - <testcase>/
            - <step>.py
            - <testcase>.cfg
            - namelist.<step>
            - streams.<step>
          - <shared_step>.py
          - <configuration_shared_module>.py
          - <configuration>.cfg
          - namelist.<step>
          - streams.<step>
    - <framework_module>.py
    - <framework_package>/

The proposed solution would slightly modify the naming conventions currently
used in COMPASS. A ``core`` would be the same as it now is -- corresponding to
an MPAS dynamical core such as ``ocean`` or ``landice``.  A ``configuration``
would also retain its current meaning -- a group of test cases such as
``global_ocean`` or ``MISMIP3D``.  For at least two reasons described
in :ref:`req_dir_struct`, we do not include
``resolution`` as the next level of hierarchy.  Instead, a ``configuration``
contains ``testcases`` which can be given any convenient name to distinguish it
from other test cases within that ``configuration``.  Several variants of a
``testcase`` can define by varying a parameter or other characteristic
(including resolution) but there need not be defined with separate packages
or modules.  This is an important aspect of the code reuse provided by this
approach.  Each ``testcase`` is made up of several steps (e.g. ``base_mesh``,
``initial_state``, ``test``).  Previously, COMPASS documentation referred to
a ``testcase`` as a ``test`` and a ``step`` as a ``case``, but users have found
this naming convention to be confusing so the proposed solution tries to make
a clearer distinction between a ``testcase`` and a ``step`` within a
``testcase``.

In addition to defining ``testcases`` and ``steps``, ``cores`` and
``configurations`` can also include "framework" python code that could be
more general (e.g. for creating meshes or initial conditions).  The main
``compass`` package would also include several framework modules and package,
some for infrastructure related to listing, setting up and cleaning up
test cases, and others for tasks common to many test cases.  As an example of the
latter, ``io.py`` defines functions for downloading files and creating symlinks.
Here's how it would be used in the ``setup()`` function of a step:

.. code-block:: python

    from compass.io import download, symlink


    def setup(step, config):

        initial_condition_database = config.get('paths',
                                                'initial_condition_database')

        filename = download(
            dest_path=initial_condition_database,
            file_name='particle_regions.151113.nc',
            url='https://web.lcrc.anl.gov/public/e3sm/mpas_standalonedata/'
                'mpas-ocean/initial_condition_database')

        symlink(filename, os.path.join(step['work_dir'], 'input_file.nc'))


.. _alg_shared_config:

Algorithm design: Shared configuration options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/01/14

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

* machine config file (found in ``compass/machines/<machine>.cfg``, with
  ``default`` being the machine name if none is specified)

* core config file (found in ``compass/<core>/<core>.cfg``)

* configuration config file (found in
  ``compass/<core>/tests/<configuration>/<configuration>.cfg``)

* any additions or modifications made within the test case's ``configure()``
  function.

* the config file passed in by the user at the command line (if any).

The ``configure()`` function allows each test case to load one or more config
files specific to the test case (e.g. ``<testcase>.cfg`` within the test case's
package) and would also allow calls to ``config.set()`` that define config
options directly.

The resulting config file would be written to ``<testcase>.cfg`` within the
test case directory and symlinked to each step subdirectory as stated above.


Algorithm design: Ability specify/modify core counts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/01/14

Contributors: Xylar Asay-Davis


Each step will specify the "target" number of cores, the minimum possible
number of cores, a number of treads, the maximum memory it will be allowed to
use, and the maximum amount of disk space it can use.  These specifications are
with the ``WorkerQueue`` approach in mind for future parallelism, as explained
in :ref:`alg_parallel`.

The total number of available cores will be determined via python or slurm
commands.  An error will be raised if too few cores are available for a
particular test.  Otherwise, the test will run on the minimum of the target
number of cores or the total available.

Some test cases (e.g. those within the ``global_ocean`` configuration) will
allow the user to specify the target and minimum number of cores as config
options, meaning they can be set to non-default values before running the test
case.  Config options are common to all steps within a test case, but
the target and minimum cores are a property of each step that must be known
before it is run (again for reasons related to a likely strategy for
future parallelism in :ref:`alg_parallel`).  This means that a test case will
need to parse the config options and use them to determine the number of cores
to run with before calling each step.  Parsing config options and updating the
target and minimum cores in a step will need to happen in each test cases that
supports this capability.  From there, shared infrastructure will take care of
determining if sufficient cores are available and how many to run each step
with if so.  Developers of individual test cases will not need to worry about
this.

Shared infrastructure can also be used to set the number of PIO tasks to one
per node, using the number of cores for a given step and the number of cores
per node from the machine config file (see :ref_`alg_machine_data`).


.. _alg_machine:

Algorithm design: Machine-specific data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/01/14

Contributors: Xylar Asay-Davis


The machine config file mentioned in :ref:`alg_shared_config` would have
the following config options:

.. code-block:: ini

    # The paths section describes paths that are used within the ocean core test
    # cases.
    [paths]

    # The mesh_database and the initial_condition_database are locations where
    # meshes / initial conditions might be found on a specific machine. They can be
    # the same directory, or different directory. Additionally, if they are empty
    # some test cases might download data into them, which will then be reused if
    # the test case is run again later.
    mesh_database = /usr/projects/regionalclimate/COMMON_MPAS/ocean/grids/mesh_database
    initial_condition_database = /usr/projects/regionalclimate/COMMON_MPAS/ocean/grids/initial_condition_database
    bathymetry_database = /usr/projects/regionalclimate/COMMON_MPAS/ocean/grids/bathymetry_database
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
These paths are currently assumed to be core independent, so would need to be
renamed or moved to core-specific sections if different cores wish to have their
own versions of these paths.  The ``compass_envs`` path is not yet used but
will be part of a future strategy for automatically generating job scripts that
include loading of the appropriate conda environment.

The ``parallel`` options are intended to contain all of the machine-specific
information needed to determine how many cores a given ``step`` would require
and to create a job script for each ``testcase`` and ``step``.  The use of
python thread parallelism will not be part of the first version of the
``compass`` package described in this design document but is expected to be
incorporated in the coming year.  An appropriate value for ``threads`` for
each machine will likely need determined as that capability gets more
exploration but is left as a placeholder for the time being.


.. _alg_dir_struct:

Algorithm design: Looser, more flexible directory structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2020/11/18

Contributors: Xylar Asay-Davis


Each test case and step will be defined by a unique subdirectory within the
work directory.  Within the base work directory, the first two levels of
subdirectories will be the same as in the current implementation:
``core/configuration``.  However, test cases will be free to determine the
(unique) subdirectory structure beyond this top-most level.  Many existing
test cases will likely stick with the ``resolution/testcase/step`` organization
structure imposed in the existing COMPASS framework, but others may choose a
different way of organizing (and, indeed, many test cases already have given the
``resolution`` subdirectory a name that is seemingly unrelated to the mesh
resolution).  A unique subdirectory for each test case and step will be provided
as a value in ``testcase['subdir']`` or ``step['subdir']`` within the python
dictionary that describes each test case or step.  The default ``subdir`` will
be the name of the test case or step, but each test case or step can modify this
as appropriate in the ``collect()`` function.

COMPASS will list test cases based on their full paths within the work directory,
since this is they way that they can be uniquely identified.


.. _alg_docs:

Algorithm design: User- and developer-friendly documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2020/11/16

Contributors: Xylar Asay-Davis


Documentation using ``sphinx`` and the ``ReadTheDocs`` template will be built
out in a manner similar to what has already been done for:

* `geometric_features <https://mpas-dev.github.io/geometric_features/stable/>`_

* `pyremap <https://mpas-dev.github.io/pyremap/stable/>`_

* `MPAS-Tools <https://mpas-dev.github.io/MPAS-Tools/stable/>`_

* `MPAS-Analysis <https://mpas-dev.github.io/MPAS-Analysis/latest/>`_

The documentation will include:

* A user's guide for listing, setting up, and cleaning up test case

* A user's guide for regression suites

* More detailed tutorials:

  * Running a test case

  * Running the regression suite

* A section for each core

  * A subsection describing the configurations

    * A sub-subsection for each test case and its steps

  * A subsection for the core's framework code

* A description of the ``compass`` framework code:

  * for use within test cases

  * for listing, setting up and cleaning up test cases

  * for managing regression test suites

* An automated documentation of the API pulled from docstrings

* A developer's guide for creating new test cases

  * core-specific details for developing new test cases


.. _alg_parallel:

Algorithm design: Considerations related to running test cases in parallel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/01/14

Contributors: Xylar Asay-Davis

I plan to use `parsl <https://parsl.readthedocs.io/en/stable/>`_ to support
parallelism between both test cases and steps within a test case.  After reading
documentation, running tutorials, and beginning prototyping, it seems that the
relatively new
`WorkQueueExecutor <https://parsl.readthedocs.io/en/stable/stubs/parsl.executors.WorkQueueExecutor.html#parsl.executors.WorkQueueExecutor>`_
is likely to be the approach within Parsl that allows the level of flexibility
and control that we would likely need.  However, this is a new enough feature
that it is still considered to be "beta" and is not available in the latest
release (v1.0.0).  So it seems premature to settle on this design choice or to
begin to incorporate it into code (except perhaps as a separate prototype).

Even so, some design choices can be made with future support for Parsl in mind.
Each step of a test case will be required to provide full paths to its input and
output files so that, in the future, Parsl can determine dependencies between
test cases and their steps using these files and control execution accordingly.
This will be the only method for determining dependencies, so steps will have to
be accurate in providing their inputs and outputs to avoid errors,
race conditions, or unnecessary blocking.  Test cases with an testing suite and
steps within a test case will also need to be ordered in such a way that outputs
of a "prerequisite" step are always defined before the inputs of any subsequent
steps that need them as inputs.  In the future, this should allow ``compass``
to associate each input file with a so-called Parsl ``DataFuture``, which will
allow each step of a test case to run only all of its input files are available.

Also with Parsl in mind, each step within a test case includes a specified
maximum memory and disk usage.  Currently, these are set to an arbitrary
reference value of 1GB each but will be calibrated to the actual approximate
usage of each step once this can be determined using debugging output from
Parsl.

This design solution will be fleshed out further in a separate document at a
later date.


.. _alg_res:

Algorithm design: Resolution can be a test case parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/01/14

Contributors: Xylar Asay-Davis

As mentioned in :ref:`alg_shared_code` and :ref:`alg_dir_struct`, resolution
will no longer be part of the directory structure for test cases and
no restrictions will be placed on how individual test cases handle resolution
or mesh generation.  To facilitate shared code, a configuration can use the
same code for a step that generates a mesh and/or initial condition for
different resolutions, e.g. passing in the resolution or mesh name as an
argument to the function.


.. _alg_alter_code:

Algorithm design: Test case code is easy to alter and rerun
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2020/12/04

Contributors: Xylar Asay-Davis

There is a local link to the ``compass`` and one could edit any files within
the package either using the link or in the original location and then simply
rerun the test case or step.  Changes do not require a test build of a conda
package or anything like that.  After some discussion about adding symlinks to
individual python files within the ``compass`` package, it was decided that this
has too many risks of being misunderstood, having unintended consequences, and
could be difficult to implement.


.. _alg_premade_ic:

Algorithm design: Support for pre-made initial condition files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/01/14

Contributors: Xylar Asay-Davis, Mark Petersen

To some degree, the implementation of this requirement will be left up to
individual test cases.  It should not be difficult to add a configuration option
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
`see example <https://github.com/MPAS-Dev/MPAS-Tools/blob/master/ocean/performance_testing/submit_performance_test_to_queue.py#L96>`_
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

Date last modified: 2021/01/15

Contributors: Xylar Asay-Davis


As already discussed, this requirement is somewhat in conflict with
:ref:`req_shared_code`, in that shared code within a test case tends to lead
to a bit more complexity but considerably less redundancy.

Test cases have 3 required functions: ``collect()``, ``configure()``, and
``run()``.  Steps of a test case have 3 required functions as well:
``collect()``, ``setup()`` and ``run()``.  Each of these is described below.

collect
~~~~~~~

Test cases and steps are "created" by calling the ``collect()`` function. I
will not go into the details of how these work under the hood because the idea
is that developers of new test cases would not need to know these details.
These ``collect()`` functions don't need to take any arguments but they can
have some (such as the resolution or other parameters) if it helps with code
reuse (see :ref:`imp_shared_code`).  ``collect()`` returns a python dictionary
that describes properties of the test case or step.  These dictionaries are
a little bit of a crutch that allows me to avoid using python classes.  It
isn't clear that they will be a whole lot more intuitive for developers who are
new to python or object-oriented programming but that was the hope.  Here is
the dictionary associated with an example test case:

.. code-block:: python

    testcase = {'module': 'compass.ocean.tests.baroclinic_channel.default',
                'name': 'default',
                'path': 'ocean/baroclinic_channel/10km/default',
                'core': 'ocean',
                'configuration': 'baroclinic_channel',
                'subdir': '10km/default',
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
                'configure': 'configure',
                'run': 'run',
                'new_step_log_file': True,
                'steps_to_run': ['initial_state', 'forward'],
                'resolution': '10km'}

and here is the same for a step, after the ``setup`` phase described below:

.. code-block:: python


    step = {'module': 'compass.ocean.tests.baroclinic_channel.initial_state',
            'name': 'initial_state',
            'subdir': 'initial_state',
            'setup': 'setup',
            'run': 'run',
            'inputs': [],
            'outputs': ['/home/xylar/data/mpas/test_baroclinic_channel/ocean/baroclinic_channel/10km/default/initial_state/base_mesh.nc',
                        '/home/xylar/data/mpas/test_baroclinic_channel/ocean/baroclinic_channel/10km/default/initial_state/culled_mesh.nc',
                        '/home/xylar/data/mpas/test_baroclinic_channel/ocean/baroclinic_channel/10km/default/initial_state/culled_graph.info',
                        '/home/xylar/data/mpas/test_baroclinic_channel/ocean/baroclinic_channel/10km/default/initial_state/ocean.nc'],
            'resolution': '10km',
            'cores': 1,
            'min_cores': 1,
            'max_memory': 8000,
            'max_disk': 8000,
            'testcase': 'default',
            'testcase_subdir': '10km/default',
            'path': 'ocean/baroclinic_channel/10km/default/initial_state',
            'work_dir': '/home/xylar/data/mpas/test_baroclinic_channel/ocean/baroclinic_channel/10km/default/initial_state',
            'base_work_dir': '/home/xylar/data/mpas/test_baroclinic_channel/',
            'config': 'default.cfg'}

Many of these entries are required for the internal operation of ``compass``.
In this case, the key ``resolution`` is only used internally to this particular
test case.  The developer of a test case can add any number of parameters and
values like this to the test case or step dictionary, for later use in
configuring, setting up or running the test case or step.

Since ``collect()`` is used as part of listing, setting up, cleaning up and
running test cases and steps, this function should only perform a minimum of
work to describe the test case and should not download or even read files or
perform any complex computations.  It is important to keep in mind that every
test case's ``collect()`` function gets called to list the test cases.

The ``collect()`` function for a test case should always call the ``collect()``
functions for each of its steps, then call the
``compass.testcase.get_testcase_default()`` function.  After that, further
alterations can be made to ``testcase`` such as modifying the ``name`` and
``subdir`` for the test case:

.. code-block:: python

    from compass.testcase import get_testcase_default
    from compass.ocean.tests.global_ocean.mesh import mesh


    def collect(mesh_name, with_ice_shelf_cavities):
        description = 'global ocean {} - mesh creation'.format(mesh_name)
        module = __name__

        name = module.split('.')[-1]
        subdir = '{}/{}'.format(mesh_name, name)
        steps = dict()
        step = mesh.collect(mesh_name, cores=4, min_cores=2,
                            max_memory=1000, max_disk=1000, threads=1,
                            with_ice_shelf_cavities=with_ice_shelf_cavities)
        steps[step['name']] = step

        testcase = get_testcase_default(module, description, steps, subdir=subdir)
        testcase['mesh_name'] = mesh_name
        testcase['with_ice_shelf_cavities'] = with_ice_shelf_cavities

        return testcase

For a step, the ``collect()`` function needs to call
``compass.testcase.get_step_default()`` and then make any further changes
to the ``step`` dictionary, typically based on arguments to the ``collect()``
function:

.. code-block:: python

    from compass.testcase import get_step_default


    def collect(mesh_name, cores, min_cores=None, max_memory=1000,
                max_disk=1000, threads=1, with_ice_shelf_cavities=False):

        step = get_step_default(__name__)
        step['mesh_name'] = mesh_name
        step['cores'] = cores
        step['max_memory'] = max_memory
        step['max_disk'] = max_disk
        if min_cores is None:
            min_cores = cores
        step['min_cores'] = min_cores
        step['threads'] = threads
        step['with_ice_shelf_cavities'] = with_ice_shelf_cavities

        return step


configure
~~~~~~~~~

The only customization for a test case that can be performed as part of setup
is customizing the config file that is shared between all steps in the test
case.  The ``configure()`` function takes ``config`` options and the
``testcase`` dictionary as arguments, and can either do nothing (``pass``) or
add to or modify ``config``.  One way to update ``config`` is by calling
``compass.config.add_config()`` to add options from  a config file, typically
found in the package (directory) for the test case:

.. code-block:: python

    from compass.config import add_config


    def configure(testcase, config):
        add_config(config, 'compass.examples.tests.example_compact.test1',
                   'test1.cfg')

Another way is to call the ``config.set()`` method:

.. code-block:: python

    config.set('example_compact', 'resolution', testcase['resolution'])

Config options in ``config`` will be written to a config file in the work
directory called ``<testcase>.cfg``, where ``<testcase>`` is the name of the
test case.  These config options differ from parameters (such as ``resolution``
in the example above) that are stored in the ``testcase`` dictionary in that
these options could be changed by a user before running the test case.
Parameters in ``testcase`` are not available in a format where users could
easily alter them and are unchanged between when the test case was set up and
when it is run.

Typically, config options that are specific to a test case will go into a
config section with the same name as the configuration (``example_compact`` in
this example) but this is a convention and developers can use whichever section
name makes sense.  It is best to avoid putting config options that are specific
to a configuration, test case or step in the config sections such as
``[paths]`` and ``[parallel]`` that are meant to be used in the shared
infrastructure of ``compass``.

It is also possible to create symlinks within ``configure()``, e.g. to a README
file that applies to all steps in a test case:

.. code-block:: python

    from importlib.resources import path

    from compass.io import symlink


    def configure(testcase, config):
        with path('compass.ocean.tests.global_ocean.files_for_e3sm', 'README') as \
                target:
            symlink(str(target), '{}/README'.format(testcase['work_dir']))

Steps do not have a ``configure()`` function because they share the same
``config`` with the other steps in the test case.  The idea is that it should
be relatively easy to change config options for all the steps in the test in
one place.

setup
~~~~~

Test cases do not have a ``setup()`` function because the only setting up they
typically include is to update config options in ``configure()``.  The
``setup()`` function in a step can be used to:

  * add parameters to the ``step`` dictionary

  * download and cache any files from the LCRC server (or elsewhere) that the
    step requires

  * make symlinks to cached data files from the server, small data files within
    ``compass`` itself, or output of other steps in this or another test case

  * update namelist options (if the MPAS model will be called)

  * update the streams file (if the MPAS model will be called)

  * add absolute paths of all require input files to the ``step['inputs']`` list

  * add absolute paths of all output files that are available to other test cases
    and steps to the ``step['outputs']`` list

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

Such a file can be parsed within ``setup()`` like this:

.. code-block:: python

    from compass import namelist


    def setup(step, config):
        with_ice_shelf_cavities = step['with_ice_shelf_cavities']

        replacements = namelist.parse_replacements(
            'compass.ocean.tests.global_ocean', 'namelist.forward')

        if with_ice_shelf_cavities:
            replacements.update(namelist.parse_replacements(
                'compass.ocean.tests.global_ocean', 'namelist.wisc'))

        namelist.generate(config=config, replacements=replacements,
                          step_work_dir=step_dir, core='ocean', mode='forward')

The ``replacements`` dictionary can be updated with multiple calls to
``compass.namelist.parse_replacements`` as in this example, or it can be
altered directly, e.g. ``replacements['config_dt'] = "'02:00:00'"``.

Streams files are in XML format and are therefore a little bit trickier to
define.  A typical workflow might be:

.. code-block:: python

    from compass import streams


    def setup(step, config):
        with_bgc = step['with_bgc']
        streams_data = streams.read('compass.ocean.tests.global_ocean',
                                    'streams.forward')

        if with_bgc:
            streams_data = streams.read('compass.ocean.tests.global_ocean',
                                        'streams.bgc', tree=streams_data)

        streams.generate(config=config, tree=streams_data, step_work_dir=step_dir,
                         core='ocean', mode='forward')

In this example, ``streams_data`` is an XML tree.  The initial call to
``streams.read()`` creates a new XML tree from scratch, and subsequent calls
update that tree (by passing it as the ``tree`` keyword argument).  Then, a
streams file is generated from the MPAS-model template using these streams.

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

``inputs`` and ``outputs`` are not currently used but are expected to become an
essential part of the parallelization strategy in the future (see
:ref:`imp_parallel`).

run
~~~

The ``run()`` function of a test case should, at a minimum, call the
function ``compass.testcase.run_steps()`` to run the steps of the test case.
It can also:

  * read config options and use them to update the number of cores and threads
    that a step can use
  * perform validation that variables and timers

Here is a relatively complex example:

.. code-block:: python

    from compass.testcase import run_steps
    from compass.validate import compare_variables


    def run(testcase, test_suite, config, logger):
        # get the these properties from the config options
        for step_name in testcase['steps_to_run']:
            step = testcase['steps'][step_name]
            for option in ['cores', 'min_cores', 'max_memory', 'max_disk',
                           'threads']:
                step[option] = config.getint('global_ocean',
                                             'forward_{}'.format(option))

        run_steps(testcase, test_suite, config, logger)

        variables = ['temperature', 'salinity', 'layerThickness', 'normalVelocity']

        compare_variables(variables, config, work_dir=testcase['work_dir'],
                          filename1='simulation/output.nc')

The ``run()`` function of a step does the main "job" of the step so the
contents will very much depend on the purpose of the step.  Many steps will
use Metis to split the domain across processors and then call the MPAS model.
An example like that is as follows:

.. code-block:: python

    from compass.model import partition, run_model
    from compass.parallel import update_namelist_pio

    def run(step, test_suite, config, logger):
        cores = step['cores']
        threads = step['threads']
        step_dir = step['work_dir']
        update_namelist_pio(config, cores, step_dir)
        partition(cores, logger)

        run_model(config, core='ocean', core_count=cores, logger=logger,
                  threads=threads)

More examples
~~~~~~~~~~~~~

Below are some further examples of configurations and test cases to give a
sense of what the implementation looks like.  Whether it is easy enough to
understand, modify and use as a starting pont for new test cases is subjective
and is going to need some discussion.

The implementation includes two example test cases, one "expanded" and one
"compact"  The expanded one shows how a test case looks without code reuse.
This version might be especially easy for a new developers to follow, but is
frustrating to modify because of the redundancy.  Here is what the file
structure looks like:

.. code-block:: none

  - compass/
    - example/
      - examples.cfg
      - __init__.py
      - tests/
        - example_expanded/
          - res1km/
            - test1/
              - __init__.py
              - step1.py
              - step2.py
              - test1.cfg
            - test2/
              - __init__.py
              - step1.py
              - step2.py
              - test2.cfg
            - __init__.py
          - res2km/
            - test1/
              - __init__.py
              - step1.py
              - step2.py
              - test1.cfg
            - test2/
              - __init__.py
              - step1.py
              - step2.py
              - test2.cfg
            - __init__.py
          - __init__.py
        - __init__.py

A typical ``__init__.py`` defining a test case looks like this:

.. code-block:: python

    from compass.config import add_config
    from compass.testcase import get_testcase_default, run_steps
    from compass.examples.tests.example_expanded.res1km.test1 import step1, step2


    def collect():
        """
        Get a dictionary of testcase properties

        Returns
        -------
        testcase : dict
            A dict of properties of this test case, including its steps
        """
        # fill in a useful description of the test case
        description = 'Tempate 1km test1'
        module = __name__
        resolution = '1km'

        # the name of the testcase is the last part of the Python module (the
        # folder it's in, so "test1" or "test2" in the "example_expanded"
        # configuration
        name = module.split('.')[-1]
        # A subdirectory for the testcase after setup.  This can be anything that
        # will ensure that the testcase ends up in a unique directory
        subdir = '{}/{}'.format(resolution, name)
        # make a dictionary of steps for this testcase by calling each step's
        # "collect" function
        steps = dict()
        for step_module in [step1, step2]:
            step = step_module.collect()
            steps[step['name']] = step

        # get some default information for the testcase
        testcase = get_testcase_default(module, description, steps, subdir=subdir)
        # add any parameters or other information you would like to have when you
        # are setting up or running the testcase or its steps
        testcase['resolution'] = resolution

        return testcase


    # this function can be used to add the contents of a config file as in the
    # example below or to add or override specific config options, as also shown
    # here.  The function must take only the "testcase" and "config" arguments, so
    # any information you need should be added to "testcase" if it is not available
    # in one of the config files used to build "config"
    def configure(testcase, config):
        """
        Modify the configuration options for this test case.

        Parameters
        ----------
        testcase : dict
            A dictionary of properties of this testcase from the ``collect()``
            function

        config : configparser.ConfigParser
            Configuration options for this testcase, a combination of the defaults
            for the machine, core and configuration
        """
        # add (or override) some configuration options that will be used during any
        # or all of the steps in this testcase
        add_config(config, 'compass.examples.tests.example_expanded.res1km.test1',
                   'test1.cfg')

        # add a config option to the config file
        config.set('example_expanded', 'resolution', testcase['resolution'])


    # The function must take only the "testcase" and "config" arguments, so
    # any information you need in order to run the testcase should be added to
    # "testcase" if it is not available in "config"
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
        # typically, this involves running all the steps in the testcase in the
        # desired sequence.  However, it may involve only running a subset of steps
        # if some are optional and not performed by default.
        run_steps(testcase, test_suite, config, logger)

And a typical step looks like this:

.. code-block:: python

    import xarray
    import os

    from mpas_tools.io import write_netcdf

    from compass.testcase import get_step_default
    from compass.io import download, symlink


    def collect():
        """
        Get a dictionary of step properties

        Returns
        -------
        step : dict
            A dictionary of properties of this step
        """
        # get some default information for the step
        step = get_step_default(__name__)
        # add any parameters or other information you would like to have when you
        # are setting up or running the testcase or its steps
        step['resolution'] = '2km'

        return step


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
        step['parameter4'] = 2.0
        step['parameter5'] = 250

        step['filename'] = 'particle_regions.151113.nc'

        initial_condition_database = config.get('paths',
                                                'initial_condition_database')
        step_dir = step['work_dir']

        # one of the required parts of setup is to define any input files from
        # other steps or testcases that are required by this step, and any output
        # files that are produced by this step that might be used in other steps
        # or testcases.  This allows COMPASS to determine dependencies between
        # testcases and their steps
        inputs = []
        outputs = []

        # download an input file if it's not already in the initial condition
        # database
        filename = download(
            file_name=step['filename'],
            url='https://web.lcrc.anl.gov/public/e3sm/mpas_standalonedata/'
                'mpas-ocean/initial_condition_database',
            config=config, dest_path=initial_condition_database)

        inputs.append(filename)

        symlink(filename, os.path.join(step_dir, 'input_file.nc'))

        # list all the output files that will be produced in the step1 subdirectory
        for file in ['output_file.nc']:
            outputs.append(os.path.join(step_dir, file))

        step['inputs'] = inputs
        step['outputs'] = outputs


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
        test_config = config['example_expanded']
        parameter1 = test_config.getfloat('parameter1')
        parameter2 = test_config.getboolean('parameter2')
        testcase = step['testcase']

        ds = xarray.open_dataset('input_file.nc')
        write_netcdf(ds, 'output_file.nc')

In these examples, the ``collect()`` function takes no arguments and building
up all the test cases in the configuration just looks like this:

.. code-block:: python

    from compass.examples.tests.example_expanded.res1km import test1 as \
        res1km_test1
    from compass.examples.tests.example_expanded.res1km import test2 as \
        res1km_test2
    from compass.examples.tests.example_expanded.res2km import test1 as \
        res2km_test1
    from compass.examples.tests.example_expanded.res2km import test2 as \
        res2km_test2


    # "collect" information about each testcase in the "example_expanded"
    # configuration, including any parameters ("resolution" in this example) that
    # distinguish different test cases in this configuration
    def collect():
        """
        Get a list of testcases in this configuration

        Returns
        -------
        testcases : list
            A list of tests within this configuration
        """
        # Get a list of information about the testcases in this configuration.
        # In this example, each testcase (test1 and test2) has a version at each
        # of two resolutions (1km and 2km), so this configuration has 4 testcases
        # in total.
        testcases = list()
        for test in [res1km_test1, res1km_test2, res2km_test1, res2km_test2]:
            testcases.append(test.collect())

        return testcases


The same test case is implemented in a more compact form that results in
exactly the same test cases with the same steps.  The compact example looks
like:

.. code-block:: none

  - compass/
    - example/
      - examples.cfg
      - __init__.py
      - tests/
        - example_compact
          - test1
            - __init__.py
            - test1.cfg
          - test2
            - __init__.py
            - test2.cfg
          - __init__.py
          - example_compact.cfg
          - step1.py
          - step2.py
          - testcase.py

The implementation of the step only differs meaningfully from the "expanded"
example in that the resolution is a parameter to the ``collect()`` function:

.. code-block:: python

    from compass.examples.tests.example_compact.testcase import collect as \
        collect_testcase

    from compass.config import add_config
    from compass.testcase import run_steps


    # "resolution" is just an example argument.  The argument can be any parameter
    # that distinguishes different variants of a test
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
        description = 'Tempate {} test1'.format(resolution)
        # This example assumes that it is possible to call a "collect" function
        # that is generic to all testcases with a different parameter ("resolution"
        # in this case).
        testcase = collect_testcase(__name__, description, resolution)
        return testcase

    ...

The modules ``step1.py`` and ``step2.py`` are now at the configuration level
and are shared between the two test cases and resolutions:

.. code-block:: python

    import xarray
    import os

    from mpas_tools.io import write_netcdf

    from compass.testcase import get_step_default
    from compass.io import download, symlink


    # "resolution" is just an example argument.  The argument can be any parameter
    # that distinguishes different variants of a test
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
        # get some default information for the step
        step = get_step_default(__name__)
        # add any parameters or other information you would like to have when you
        # are setting up or running the testcase or its steps
        step['resolution'] = resolution

        return step


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

        initial_condition_database = config.get('paths',
                                                'initial_condition_database')
        step_dir = step['work_dir']

        # one of the required parts of setup is to define any input files from
        # other steps or testcases that are required by this step, and any output
        # files that are produced by this step that might be used in other steps
        # or testcases.  This allows COMPASS to determine dependencies between
        # testcases and their steps
        inputs = []
        outputs = []

        # download an input file if it's not already in the initial condition
        # database
        filename = download(
            file_name=step['filename'],
            url='https://web.lcrc.anl.gov/public/e3sm/mpas_standalonedata/'
                'mpas-ocean/initial_condition_database',
            config=config, dest_path=initial_condition_database)

        inputs.append(filename)

        symlink(filename, os.path.join(step_dir, 'input_file.nc'))

        # list all the output files that will be produced in the step1 subdirectory
        for file in ['output_file.nc']:
            outputs.append(os.path.join(step_dir, file))

        step['inputs'] = inputs
        step['outputs'] = outputs


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
        test_config = config['example_compact']
        parameter1 = test_config.getfloat('parameter1')
        parameter2 = test_config.getboolean('parameter2')
        testcase = step['testcase']

        ds = xarray.open_dataset('input_file.nc')
        write_netcdf(ds, 'output_file.nc')

Python dictionaries are used to define different sets of parameters for
different resolutions in this example.  Alternatively, parameters could be
analytic functions of the resolution, or they could also be passed in as
additional arguments to the ``collect()`` function along with the resolution.

A similar strategy to these examples was employed in the idealized ocean
configurations: ``baroclinic_channel``, ``ziso`` and ``ice_shelf_2d``.

global ocean configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``global_ocean`` configuration works with variable resolution meshes,
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
              - spinup
                - __init__.py
                - streams.template
              - __init__.py
              - ec30to60.cfg
              - namelist.split_explicit
            - qu240
              - spinup
                - __init__.py
                - streams.template
              - __init__.py
              - namelist.rk4
              - namelist.split_explicit
              - qu240.cfg
          ...

To implement a new global mesh, one would need to define the resolution
in the ``__init__.py`` file:

.. code-block:: python

    import numpy as np
    import mpas_tools.mesh.creation.mesh_definition_tools as mdt


    def build_cell_width_lat_lon():
        """
        Create cell width array for this mesh on a regular latitude-longitude grid

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
        dlat = 0.1
        nlon = int(360./dlon) + 1
        nlat = int(180./dlat) + 1
        lon = np.linspace(-180., 180., nlon)
        lat = np.linspace(-90., 90., nlat)

        cellWidthVsLat = mdt.EC_CellWidthVsLat(lat)
        cellWidth = np.outer(cellWidthVsLat, np.ones([1, lon.size]))

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

Finally, the developer would implement the ``spinup`` test case, using one of
the existing spin-up test cases as a kind of a template:

.. code-block:: python

    from compass.testcase import run_steps, get_testcase_default
    from compass.ocean.tests.global_ocean import forward
    from compass.ocean.tests.global_ocean.description import get_description
    from compass.ocean.tests.global_ocean.init import get_init_sudbdir
    from compass.ocean.tests import global_ocean
    from compass.validate import compare_variables


    def collect(mesh_name, with_ice_shelf_cavities, initial_condition, with_bgc,
                time_integrator):
        """
        Get a dictionary of testcase properties

        Parameters
        ----------
        mesh_name : str
            The name of the mesh

        with_ice_shelf_cavities : bool
            Whether the mesh should include ice-shelf cavities

        initial_condition : {'PHC', 'EN4_1900'}
            The initial condition to build

        with_bgc : bool
            Whether to include BGC variables in the initial condition

        time_integrator : {'split_explicit', 'RK4'}
            The time integrator to use for the run

        Returns
        -------
        testcase : dict
            A dict of properties of this test case, including its steps
        """
        if time_integrator != 'split_explicit':
            raise ValueError('{} spin-up not defined for {}'.format(
                mesh_name, time_integrator))

        description = get_description(
            mesh_name, initial_condition, with_bgc, time_integrator,
            description='spin-up')
        module = __name__

        init_subdir = get_init_sudbdir(mesh_name, initial_condition, with_bgc)

        name = module.split('.')[-1]
        subdir = '{}/{}/{}'.format(init_subdir, name, time_integrator)

        steps = dict()

        restart_times = ['0001-01-11_00:00:00', '0001-01-21_00:00:00']
        restart_filenames = [
            'restarts/rst.{}.nc'.format(restart_time.replace(':', '.'))
            for restart_time in restart_times]

        step_name = 'damped_spinup_1'
        inputs = None
        outputs = ['output.nc', '../{}'.format(restart_filenames[0])]
        namelist_replacements = {
            'config_run_duration': "'00-00-10_00:00:00'",
            'config_dt': "'00:20:00'",
            'config_Rayleigh_friction': '.true.',
            'config_Rayleigh_damping_coeff': '1.0e-4'}
        stream_replacements = {
            'output_interval': '00-00-10_00:00:00',
            'restart_interval': '00-00-10_00:00:00'}
        step = forward.collect(mesh_name, with_ice_shelf_cavities,
                               with_bgc,  time_integrator,
                               testcase_module=module,
                               streams_file='streams.template',
                               namelist_replacements=namelist_replacements,
                               stream_replacements=stream_replacements,
                               inputs=inputs, outputs=outputs)
        step['name'] = step_name
        step['subdir'] = step['name']
        steps[step['name']] = step

        step_name = 'simulation'
        inputs = ['../{}'.format(restart_filenames[0])]
        outputs = ['../{}'.format(restart_filenames[1])]
        namelist_replacements = {
            'config_run_duration': "'00-00-10_00:00:00'",
            'config_do_restart': '.true.',
            'config_start_time': "'{}'".format(restart_times[0])}
        stream_replacements = {
            'output_interval': '00-00-10_00:00:00',
            'restart_interval': '00-00-10_00:00:00'}
        step = forward.collect(mesh_name, with_ice_shelf_cavities,
                               with_bgc,  time_integrator,
                               testcase_module=module,
                               streams_file='streams.template',
                               namelist_replacements=namelist_replacements,
                               stream_replacements=stream_replacements,
                               inputs=inputs, outputs=outputs)
        step['name'] = step_name
        step['subdir'] = step['name']
        steps[step['name']] = step

        testcase = get_testcase_default(module, description, steps, subdir=subdir)
        testcase['mesh_name'] = mesh_name
        testcase['with_ice_shelf_cavities'] = with_ice_shelf_cavities
        testcase['initial_condition'] = initial_condition
        testcase['with_bgc'] = with_bgc
        testcase['restart_filenames'] = restart_filenames

        return testcase


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
        # get the these properties from the config options
        for step_name in testcase['steps_to_run']:
            step = testcase['steps'][step_name]
            for option in ['cores', 'min_cores', 'max_memory', 'max_disk',
                           'threads']:
                step[option] = config.getint('global_ocean',
                                             'forward_{}'.format(option))

        run_steps(testcase, test_suite, config, logger)

        variables = ['temperature', 'salinity', 'layerThickness', 'normalVelocity']

        compare_variables(variables, config, work_dir=testcase['work_dir'],
                          filename1='simulation/output.nc')

The global ocean configuration includes many other test cases and steps, and
is quite complex compared to idealized test cases, so may need the most
discussion.

Whew! That was a lot, thanks for bearing with me.

.. _imp_shared_code:

Implementation: Shared code
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/01/15

Contributors: Xylar Asay-Davis

The package includes myriad examples of code sharing so I will highlight a few.

compass framework
~~~~~~~~~~~~~~~~~

The ``compass`` framework (modules and packages not in the core-specific
packages) has a lot of code that is shared across existing test cases and could
be very useful for future ones.

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
cores and nodes available for running steps, and
``update_namelist_pio()``, which updates the number of PIO tasks and the
stride between tasks based on the number of cores that a step is actually
running with.

within a core
~~~~~~~~~~~~~

Legacy COMPASS shares functionality with a core by having scripts at the core
level that are linked within test cases and which take command-line arguments
that function roughly the same way as function arguments.  But these scripts
are not able to share any code between them unless it is from ``mpas_tools``
or another external package.

A core in ``compass`` could, theoretically, build out functionality as complex
as in MPAS-Model if desired.  Indeed, it is my ambition to gradually replace
"init mode" in MPAS-Ocean with equivalent python functionality, starting with
simpler test cases.  This has already been accomplished for the 3 idealized
test cases included in the proposed design.

The current shared functionality in the ``ocean`` core includes:

  * ``compass.ocean.namelists`` and ``compass.ocean.streams``: namelist
    replacements and streams that are similar to core-level templates in legacy
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


within a configuration
~~~~~~~~~~~~~~~~~~~~~~

So far, the most common type of shared code within configurations are modules
defining steps that are used in multiple test cases.  For example, the
``baroclinic_channel`` configuration uses shared modules to define the
``initial_state`` and ``forward`` steps of each test case.  Configurations
also often include namelist and streams files with replacements to use across
test cases.

In addition to shared steps, the ``global_ocean`` configuration includes
some additional shared modules:

  * ``compass.ocean.tests.global_ocean.description``: creates a description of
    a test case (basically the long name of the testcase) given its parameters

  * ``compass.ocean.tests.global_ocean.mesh``: defines properties of each
    global mesh (as well as a ``mesh`` test case)

  * ``compass.ocean.tests.global_ocean.metadata``: determines the values of a
    set of metadata related to the E3SM mesh name, initial condition, conda
    environment, etc. that are added to nearly all ``global_ocean`` NetCDF
    output

  * ``compass.ocean.tests.global_ocean.subdir``: helps with maintaining the
    slightly complex subdirectory structure within ``global_ocean`` test cases.

The shared code in ``global_ocean`` has made it easy to define 138 different
test cases using the QU240 (or QUwISC240) mesh.  This is possible because
the same conceptual test (e.g. restart) can be defined:

  * with or without ice-shelf cavities

  * with the PHC or EN4 1900 initial conditions

  * with or without BGC support

  * with the RK4 or split-explicit time integrators

In practice, this is likely overkill and many of these variants will never be
used.  So we will likely wish to reduce the number of tests back to closer to
what it was in legacy COMPASS in the near future.  But I wanted to demonstrate
the capability first.

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

Date last modified: 2021/01/16

Contributors: Xylar Asay-Davis

As discussed in :ref:`alg_shared_config`, the proposed design builds up the
config file for a given test from several sources.  Some of the config options
are related to setting up the test case (e.g. the locations of cached data
files) but the majority are related to running the steps of the test case.

During setup of a test case and its steps, The config file is assembled from
a number of sources.  Before the ``configure()`` function of the test case is
called, config options come from:

* the default config file, ``default.cfg``, which sets a few options related to
  downloading files during setup (whether to download and whether to check the
  size of files already downloaded)

* the machine config file (using ``machines/default.cfg`` if none was
  specified) with information on the parallel system and (typically) the paths
  to cached data files

* the core's config file.  For the ocean core, this sets default paths to
  the MPAS model build (including the namelist templates).  It uses "extended
  interpolation" in the config file to use config opitons within other config
  options, e.g. ``model = ${paths:mpas_model}/ocean_model``.

* the configuration's config file if one is found.  For idealized
  configurations, these include config options that were previously init-mode
  namelist options.  For ``global_ocean``, these include defaults for mesh
  metadata (again using "extended interpolation"); the default number of cores
  and other resource usage for mesh, init and forward steps; and options
  related to files created for E3SM initial conditions.

Then, the ``configure()`` function is called on the test case itself.  All of
the current ocean test cases first call a shared ``configure()`` function at
the configuration level, e.g.:

.. code-block:: python

    from compass.ocean.tests import global_ocean

    def configure(testcase, config):
        global_ocean.configure(testcase, config)

where ``configure()`` in ``global_ocean`` is:

.. code-block:: python

    from compass.config import add_config
    from compass.ocean.tests.global_ocean.mesh.mesh import get_mesh_package
    from compass.ocean.tests.global_ocean.init import add_descriptions_to_config

    def configure(testcase, config):
        mesh_name = testcase['mesh_name']
        package, prefix = get_mesh_package(mesh_name)
        add_config(config, package, '{}.cfg'.format(prefix), exception=True)
        if testcase['with_ice_shelf_cavities']:
            config.set('global_ocean', 'prefix', '{}wISC'.format(
                config.get('global_ocean', 'prefix')))

        name = testcase['name']
        add_config(config, 'compass.ocean.tests.global_ocean.{}'.format(name),
                   '{}.cfg'.format(name), exception=False)

        add_descriptions_to_config(testcase, config)

In this case, a config options related to the mesh are loaded, then those
related to ice-shelf cavities (if they are included in the mesh), then those
specific to the test case itself.

Although none of the existing ocean test cases do so, further changes could be
made to the config file beyond those at the configuration level.  Indeed,
there is no reason a test case cannot just set its config options or read them
from a file without calling a configuration-level function, this is just a
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

Each ``step`` dictionary is expected to include two keys, ``cores`` and
``min_cores`` by the time the test case call ``compass.testcase.run_steps()``.
``cores`` is the target number of cores for the step and ``min_cores`` is the
minimum number of cores, below which the test case would probably fail. Before
a step is run, ``compass`` finds out how many total cores are available to run
the test. If the number is below ``step['min_cores']``, an error is raised.
Otherwise, the test case will run with ``step['cores']`` or the number of
available cores, whichever is lower.

The idea is that the same test case could be run efficiently on one or more
nodes of an HPC machine but could also be run on a laptop or desktop if the
minimum number of required cores is reasonable.

There are a variety of ways that the ``cores`` and ``min_cores`` entries can be
added.  The most straightforward is to add them in the step's ``collect()``
function.  They could be arguments to the function:

.. code-block:: python

    from compass.testcase import get_step_default


    def collect(cores, min_cores=None):

        step = get_step_default(__name__)
        step['cores'] = cores
        if min_cores is None:
            min_cores = cores
        step['min_cores'] = min_cores
        return step

or just hard coded:

.. code-block:: python

    from compass.testcase import get_step_default


    def collect():
        step = get_step_default(__name__)
        step['cores'] = 1
        step['min_cores'] = 1
        return step

Or they could be defined later in the process, at setup or in the test
case's ``run()`` function.  (Defining them in the step's ``run()`` is too late,
since the number of cores to actually use is determined before this call is
made.)  In ``global_ocean``, the number of cores and minimum cores are set
using config options.  Since users could modify these before calling the
``run.py`` script, they are parsed in the test case's ``run()`` function
before ``run_steps()`` is called:

.. code-block:: python

    def run(testcase, test_suite, config, logger):
        work_dir = testcase['work_dir']
        steps = testcase['steps_to_run']
        if 'initial_state' in steps:
            step = testcase['steps']['initial_state']
            # get the these properties from the config options
            for option in ['cores', 'min_cores', 'max_memory', 'max_disk',
                           'threads']:
                step[option] = config.getint('global_ocean',
                                             'init_{}'.format(option))

        run_steps(testcase, test_suite, config, logger)


The ``steps_to_run`` entry in ``testcase`` is a list of the subset of the
steps that were actually requested to run from the test case.  For example,
if you run a step on its own, it still actually runs the test case but only
requesting that one step.  Some test cases include steps that are not run by
default, so ``step_to_run`` will be assigned to only those steps that should
run by default:

.. code-block:: python

    from compass.testcase import get_testcase_default


    def collect(with_ice_shelf_cavities):
        ...
        steps_to_run = ['initial_state']
        if with_ice_shelf_cavities:
            steps_to_run.append('ssh_adjustment')

        testcase = get_testcase_default(module, description, steps, subdir=subdir)
        testcase['steps_to_run'] = steps_to_run

        return testcase

The number of PIO tasks and the stride between tasks can then be modified
if needed so that there is one PIO task per node.


.. _imp_machine:

Implementation: Machine-specific data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/01/16

Contributors: Xylar Asay-Davis


Machine-specific configuration options are in a set of config files under
``compass/machines``.  As an example, the config file for Anvil looks like:

.. code-block:: cfg

    # The paths section describes paths that are used within the ocean core test
    # cases.
    [paths]

    # The mesh_database and the initial_condition_database are locations where
    # meshes / initial conditions might be found on a specific machine. They can be
    # the same directory, or different directory. Additionally, if they are empty
    # some test cases might download data into them, which will then be reused if
    # the test case is run again later.
    mesh_database = /lcrc/group/e3sm/public_html/mpas_standalonedata/mpas-ocean/mesh_database
    initial_condition_database = /lcrc/group/e3sm/public_html/mpas_standalonedata/mpas-ocean/initial_condition_database
    bathymetry_database = /lcrc/group/e3sm/public_html/mpas_standalonedata/mpas-ocean/bathymetry_database
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
       badger
       default
       cori-haswell
       chrysalis
       cori-knl
       compy
       grizzly

When setting up a test case or test suite, the ``--machine`` or ``-m`` flag
is used to specify the machine.


.. _imp_dir_struct:

Implementation: Looser, more flexible directory structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/01/16

Contributors: Xylar Asay-Davis

Test cases (and steps) in ``compass`` are uniquely defined by their relative
paths within the work directory.  The first two subdirectories in this path
must be the name of the core and of the configuration.  The names and
organization beyond that are quite flexible.  Steps are expected to be nested
somewhere within test cases but there is no restriction on the number of levels
of subdirectories or their meaning beyond that of the configuration.

The idealized ocean configurations that have been implemented so far and the
example configurations use the same organization as in legacy COMPASS:

.. code-block:: none

    core/configuration/resolution/testcase/step

For example:

.. code-block:: none

    ocean/baroclinic_channel/10km/default/initial_state

But the ``global_ocean`` configuration takes full advantage of the new
flexibility.  Here are the directories for test cases using the QU240 mesh:

.. code-block:: none

    ocean/global_ocean/QU240/mesh
    ocean/global_ocean/QU240/PHC/init
    ocean/global_ocean/QU240/PHC/performance_test/split_explicit
    ocean/global_ocean/QU240/PHC/performance_test/RK4
    ocean/global_ocean/QU240/PHC/restart_test/split_explicit
    ocean/global_ocean/QU240/PHC/restart_test/RK4
    ocean/global_ocean/QU240/PHC/decomp_test/split_explicit
    ocean/global_ocean/QU240/PHC/decomp_test/RK4
    ocean/global_ocean/QU240/PHC/threads_test/split_explicit
    ocean/global_ocean/QU240/PHC/threads_test/RK4
    ocean/global_ocean/QU240/PHC/analysis_test/split_explicit
    ocean/global_ocean/QU240/PHC/analysis_test/RK4
    ocean/global_ocean/QU240/PHC/daily_output_test/split_explicit
    ocean/global_ocean/QU240/PHC/daily_output_test/RK4
    ocean/global_ocean/QU240/PHC/spinup/split_explicit
    ocean/global_ocean/QU240/PHC/files_for_e3sm/split_explicit
    ocean/global_ocean/QU240/PHC/spinup/RK4
    ocean/global_ocean/QU240/PHC/files_for_e3sm/RK4
    ocean/global_ocean/QU240/PHC_BGC/init
    ocean/global_ocean/QU240/PHC_BGC/performance_test/split_explicit
    ocean/global_ocean/QU240/PHC_BGC/performance_test/RK4
    ocean/global_ocean/QU240/PHC_BGC/restart_test/split_explicit
    ocean/global_ocean/QU240/PHC_BGC/restart_test/RK4
    ocean/global_ocean/QU240/PHC_BGC/decomp_test/split_explicit
    ocean/global_ocean/QU240/PHC_BGC/decomp_test/RK4
    ocean/global_ocean/QU240/PHC_BGC/threads_test/split_explicit
    ocean/global_ocean/QU240/PHC_BGC/threads_test/RK4
    ocean/global_ocean/QU240/PHC_BGC/analysis_test/split_explicit
    ocean/global_ocean/QU240/PHC_BGC/analysis_test/RK4
    ocean/global_ocean/QU240/PHC_BGC/daily_output_test/split_explicit
    ocean/global_ocean/QU240/PHC_BGC/daily_output_test/RK4
    ocean/global_ocean/QU240/PHC_BGC/spinup/split_explicit
    ocean/global_ocean/QU240/PHC_BGC/files_for_e3sm/split_explicit
    ocean/global_ocean/QU240/PHC_BGC/spinup/RK4
    ocean/global_ocean/QU240/PHC_BGC/files_for_e3sm/RK4
    ocean/global_ocean/QU240/EN4_1900/init
    ocean/global_ocean/QU240/EN4_1900/performance_test/split_explicit
    ocean/global_ocean/QU240/EN4_1900/performance_test/RK4
    ocean/global_ocean/QU240/EN4_1900/restart_test/split_explicit
    ocean/global_ocean/QU240/EN4_1900/restart_test/RK4
    ocean/global_ocean/QU240/EN4_1900/decomp_test/split_explicit
    ocean/global_ocean/QU240/EN4_1900/decomp_test/RK4
    ocean/global_ocean/QU240/EN4_1900/threads_test/split_explicit
    ocean/global_ocean/QU240/EN4_1900/threads_test/RK4
    ocean/global_ocean/QU240/EN4_1900/analysis_test/split_explicit
    ocean/global_ocean/QU240/EN4_1900/analysis_test/RK4
    ocean/global_ocean/QU240/EN4_1900/daily_output_test/split_explicit
    ocean/global_ocean/QU240/EN4_1900/daily_output_test/RK4
    ocean/global_ocean/QU240/EN4_1900/spinup/split_explicit
    ocean/global_ocean/QU240/EN4_1900/files_for_e3sm/split_explicit
    ocean/global_ocean/QU240/EN4_1900/spinup/RK4
    ocean/global_ocean/QU240/EN4_1900/files_for_e3sm/RK4
    ocean/global_ocean/QU240/EN4_1900_BGC/init
    ocean/global_ocean/QU240/EN4_1900_BGC/performance_test/split_explicit
    ocean/global_ocean/QU240/EN4_1900_BGC/performance_test/RK4
    ocean/global_ocean/QU240/EN4_1900_BGC/restart_test/split_explicit
    ocean/global_ocean/QU240/EN4_1900_BGC/restart_test/RK4
    ocean/global_ocean/QU240/EN4_1900_BGC/decomp_test/split_explicit
    ocean/global_ocean/QU240/EN4_1900_BGC/decomp_test/RK4
    ocean/global_ocean/QU240/EN4_1900_BGC/threads_test/split_explicit
    ocean/global_ocean/QU240/EN4_1900_BGC/threads_test/RK4
    ocean/global_ocean/QU240/EN4_1900_BGC/analysis_test/split_explicit
    ocean/global_ocean/QU240/EN4_1900_BGC/analysis_test/RK4
    ocean/global_ocean/QU240/EN4_1900_BGC/daily_output_test/split_explicit
    ocean/global_ocean/QU240/EN4_1900_BGC/daily_output_test/RK4
    ocean/global_ocean/QU240/EN4_1900_BGC/spinup/split_explicit
    ocean/global_ocean/QU240/EN4_1900_BGC/files_for_e3sm/split_explicit
    ocean/global_ocean/QU240/EN4_1900_BGC/spinup/RK4
    ocean/global_ocean/QU240/EN4_1900_BGC/files_for_e3sm/RK4

As in legacy COMPASS, there is a subdirectory for the mesh.  In the proposed
design, there is a ``mesh`` test case with a single ``mesh`` step within that
subdirectory.  The mesh constructed and culled within that test case serves
as the starting point for all other test cases using the mesh.

Then, there are 4 different subdirectories for variants of the initial
condition: either PHC or EN4_1900, and either with or without BGC.  Each of
these subdirectories has an ``init`` test case that creates the initial
condition.  The results of this test case are then used in all other steps
within the subdirectory for that initial condition.

Each remaining test case includes one or more forward model runs, or uses the
results of such a run.  Since the forward model can be run with either the
split-explicit or the RK4 time integrator, variants of each test case are
supported with each time integrator.  It is important that these are
conceptually separate test cases because we use both the split-explicit and
the RK4 versions of many test cases in our test suites.  Each requires a set of
corresponding namelist options and modifications to streams, so it is also not
trivial for a user to switch between the two time integrators simply by
manually modifying the test case at runtime.

We will likely never use many the 69 test cases defined for the QU240 mesh
so it will likely be better to drop many of these unused configurations from
the list of available test cases.  For now, I have provided the full set as
an indication of how powerful the flexibility and code reuse in the new
``compass`` can be, and how critical the flexible directory structure is for
this capability.


.. _imp_docs:

Implementation: User- and developer-friendly documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/01/16

Contributors: Xylar Asay-Davis


The documentation is still very much a work in progress and may be added with
a separate pull request so that commits related to the infrastructure don't get
intermixed with those for documentation.

Documentation will continue to be generated automatically with Azure Pipelines
using sphinx, as is the case for this design doc.

The legacy COMPASS documentation will be renamed with "legacy" added to its
titles (e.g. "Legacy User's Guide") and will be included at the end of the
table of contents.

More soon...


.. _imp_parallel:

Implementation: Considerations related to running test cases in parallel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/01/16

Contributors: Xylar Asay-Davis


While an implementation of test-case parallelism will be left to a future
design document and implementation, several parts of the current ``compass``
design and implementation were made with this work in mind.

test_suite argument
~~~~~~~~~~~~~~~~~~~

The ``run()`` functions of test cases and steps includes a ``test_suite``
dictionary as an argument.  This dictionary currently has little information
of use to individual test cases (the work directory, the list of test cases
and the name of the suite) but could be used in the future to hold information
used to configure Parsl.

cores, max_memory and max_disk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each step keeps track of not only the number of cores (and threads) used but
also the maximum allowed memory and disk.  While the latter two are not
currently used for anything and the values in most test case are just
placeholders, they are expected to be useful for Parsl ``WorkerQueues``.

inputs and outputs
~~~~~~~~~~~~~~~~~~

An effort has been made to be thorough about providing an absolute path to the
inputs and outputs of each step.  These are not used for very anything at
present but they would be used by Parsl to determine dependencies between
steps to figure out which can run in parallel with one another.


.. _imp_res:

Implementation: Resolution can be a test case parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/01/16

Contributors: Xylar Asay-Davis


This was discussed in :ref:`imp_dir_struct`.  For all of the ocean
configurations, either the resolution or the name of the mesh (which implicitly
includes the resolution) is an argument to the ``collect()`` function of each
test case and step.  All test cases use that resolution or mesh name as a
subdirectory within the relative path of the test case.  So far, no convergence
tests have been added where resolution is a parameter that varies across steps
in a test case but the ``rpe_test`` test case of the ``baroclinic_channel``
includes viscosity as a parameter that varies across steps, and resolution is
expected to be easy to use in the same way for future test cases.


.. _imp_alter_code:

Implementation: Test case code is easy to alter and rerun
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2021/01/16

Contributors: Xylar Asay-Davis


Local symlinks to the ``compass`` directory seem to provide and easy method for
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

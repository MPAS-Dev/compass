
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
that it is still considered to be ``beta'' and is not available in the latest
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


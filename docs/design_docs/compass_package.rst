
compass python package
======================

Author: Xylar Asay-Davis

date: 2020/11/16


Summary
-------

While the existing COMPASS infrastructure has served us well in providing a
framework for setting up MPAS testcases and test suites, several shortcomings
have emerged over the years.  First, new users have not found the current
system of creating XML files that are parsed into python scripts, namelists and
streams files very intuitive or easy to modify.  Second, the current scripts and
XML files do not lend themselves to code reuse, leading to a cumbersome system
of symlinked scripts and XML config files.  Third, the only way that users
currently have of modifying testcases to edit namelists, streams files and run
scripts for each step individually after the testcase has been set up.  Fourth
and related, there is not a way for users to easily constrain or modify
how many cores a given testcase uses, making it hard to configure testcases
in a way that is appropriate for multiple machines.  Fifth and also related,
COMPASS does not currently have a way to provide machine-specific paths and
other information that could allow for better automation on supported machines.
Sixth, the directory structure imposed by COMPASS
(``core/configuration/resoltuion/testcase/step``) is too rigid for many
applications. Finally, COMPASS is not well documented and the documentation that
does exist is not very helpful either for new users or for new developers
interested in creating new testcases.

The proposed ``compass`` python package should address these challenges with
the hope of making the MPAS testcases significantly easier to develop and run.

Requirements
------------

Requirement: Make testcases easy to understand and modify
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2020/11/16

Contributors: Xylar Asay-Davis


Currently, test cases are written primarily in XML files that are then used to
generate a python script along with namelist and streams files for MPAS-Model.
We have found that this system is not very intuitive for new users or very easy
to get started with.  New users would likely have an easier time if test cases
were written in a more direct way, using a common language rather than custom
XML tags.


Requirement: Shared code
^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2020/11/16

Contributors: Xylar Asay-Davis

Currently, there are two approaches to sharing code between COMPASS testcases.
In some cases, shared code is part of an external package, often ``mpas_tools``,
which is appropriate for code that may be used outside of COMPASS.  However,
this approach is cumbersome for testing, so it is not the preferred approach for
COMPASS-specific code.  In other cases, scripts are symlinked in testcases and
run with testcase-specific flags.  This approach is also cumbersome and does
not lend itself to code reuse between scripts.  Finally, many test cases attempt
to share XML files using symlinks, a practice that has led to frequent
unintended consequences when a linked file is modified with changes appropriate
for only one of the test cases that uses it.  A more sophisticate method
for code reuse should be developed beyond symlinks to isolated scripts and
shared XML files.


Requirement: Shared configuration options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2020/11/16

Contributors: Xylar Asay-Davis

Currently, COMPASS reads a configuration file as part of setting up testcases,
but these configuration options are largely unavailable to testcases themselves.
Some steps of some testcases (e.g. ``files_for_e3sm`` in some
``ocean/global_ocean`` test cases) have their own dedicated config files, but
these are again separate from the config file used to set up testcases, are
awkward to modify (requiring editing after testcase generation).


Requirement: Ability specify/modify core counts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2020/11/16

Contributors: Xylar Asay-Davis

Some testcases involve multiple steps of running the MPAS model, each with a
hard-coded number of cores (and often with a corresponding hard-coded number of
PIO tasks), which makes it tedious to modify the number of cores or nodes that
a given testcase uses.  This problem is exacerbated in test suites, where it is
even more difficult and tedious to modify processor counts for individual test
cases.  A system is needed where the user can more easily override the default
number of cores used in one or more steps of a testcase.  The number of PIO
tasks and the stride between them should be updated automatically to accommodate
the new core count.


Requirement: Machine-specific data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2020/11/16

Contributors: Xylar Asay-Davis

Currently, many COMPASS testcases have hard-coded processor counts and related
information that are likely only appropriate for one machine.  Users must
specify the paths to shared datasets such as meshes and initial conditions.
Users must also know where to load the ``compass`` conda environment appropriate
for running testcases.  If information were available on the system being used,
such as the number of cores per node and the locations of shared paths,
testcases and the COMPASS infrastructure could take advantage of this to
automate many aspects of setting up and running testcases that are currently
unnecessarily redundant and tedious.

Requirement: Looser, more flexible directory structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2020/11/16

Contributors: Xylar Asay-Davis

The directory structure currently imposed by COMPASS
(``core/configuration/resoltuion/testcase/step``) is too rigid for many
applications.  Some testcases (e.g. convergence tests) require multiple
resolutions within the testcase.  Some configurations would prefer to sort
testcases based on another parameter or property besides resolution.  It would
be convenient if the directory structure could be more flexible, depending on
the needs of a given configuration and testcase.  Even so, it is important that
the subdirectory of each testcase and step is unique, they do not overwrite one
another.


Requirement: User- and developer-friendly documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2020/11/16

Contributors: Xylar Asay-Davis

We need a set of user-friendly documentation on how to setup and activate an
appropriate conda environment; build the appropriate MPAS core; list and setup
a testcase; and run the testcase in via a batch queuing system.

Similarly, we need a set of developer-friendly documentation to describe how to
create a new "configuration" with one or more "testcases", each made up of
one or more "steps".


Algorithmic Formulations
------------------------

Design solution: Make testcases easy to understand and modify
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2020/11/16

Contributors: Xylar Asay-Davis


The proposed solution would be to write testcases as Python packages made up
of modules and functions within a larger ``compass`` package.  A testcase will
have separate functions to collect information on the testcase (the equivalent
of parsing ``config_driver.xml`` in the current implementation), configure it
by adding testcase-specific config options, and run the default steps.  Each
step of a testcase (equivalent to the other ``config_*.xml`` files) will be
a module (possibly shared between testcases) that implements functions for
collecting information of the step (equivalent to parsing ``config_*.xml``),
setting up the step (downloading files, making symlinks, creating namelist and
streams files), and running the step.  A balance will have to be struck between
code reusability and readability within each configuration (a set of testcases).

Readability would be improved by using Jinja2 templates for code generation,
rather than via string manipulation within python scripts as is the case in the
current COMPASS implementation:

.. code-block:: python

    #!/usr/bin/env python
    import pickle
    import configparser

    from {{ step.module }} import {{ step.run }} as run


    def main():
        with open('{{ step.name }}.pickle', 'rb') as handle:
            test = pickle.load(handle)

        config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())
        config.read('{{ step.config }}')

        run(test, config)


    if __name__ == '__main__':
        main()


The only XML files that would be used would be
templates for streams files, written in the same syntax as the resulting
streams files.

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


Design solution: Shared code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2020/11/18

Contributors: Xylar Asay-Davis


By organizing both the testcases themselves and shared framework code into a
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
would also retain its current meaning -- a group of testcases such as
``global_ocean`` or ``MISMIP3D``.  For at least two reasons described above
in "Looser, more flexible directory structure", we do not include
``resolution`` as the next level of hierarchy.  Instead, a ``configuration``
contains ``testcases`` which can be given any convenient name to distinguish it
from other testcases within that ``configuration``.  Several variants of a
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
testcases, and others for tasks common to many testcases.  As an example of the
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


Design solution: Shared configuration options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2020/11/18

Contributors: Xylar Asay-Davis


In the work directory, each testcase will have a single config file that is
populated during the setup phase and which is symlinked within each step of the
testcase.  The idea of having a single config file per testcase, rather than
one for each step, is to make it easier for users to modify config options in
one place at runtime before running all the steps in a testcase.  This will
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
* any additions or modifications made within the testcase's ``configure()``
  function.

The ``configure()`` function allows each test case to load one or more config
files specific to the testcase (e.g. ``<testcase>.cfg`` within the testcase's
module) and would also allow calls to ``config.set()`` that define config
options directly.

The resulting config file would be written to ``<testcase>.cfg`` within the
testcase directory and symlinked to each step subdirectory as stated above.


Design solution: Ability specify/modify core counts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2020/11/18

Contributors: Xylar Asay-Davis


(The design solution here is still a work in progress.)

Within the ``run()`` function, a ``step`` will be able to call a function to
find out how many nodes and cores are available on the system (or in the batch
job) to run jobs.  Based on this information and a target number of cores, the
step can figure out how many cores to run with and can (if needed) update
namelist options related to PIO tasks to be compatible.


Design solution: Machine-specific data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2020/11/18

Contributors: Xylar Asay-Davis


The machine config file mentioned in "Shared configuration options" would have
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
    account = climateacme

    # the number of multiprocessing or dask threads to use
    threads = 18

The various ``paths`` would help with finding mesh or initial condition files.
These paths are currently assumed to be core independent, so would need to be
renamed or moved to core-specific sections if different cores wish to have their
own versions of these paths.

The ``parallel`` options are intended to contain all of the machine-specific
information needed to determine how many cores a given ``step`` would require
and to create a job script for each ``testcase`` and ``step``.  The use of
python thread parallelism is relatively new and experimental in COMPASS, so the
way that an appropriate value for ``threads`` is determined may need to evolve
as that capability gets more exploration.


Design solution: Looser, more flexible directory structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2020/11/18

Contributors: Xylar Asay-Davis


Each testcase and step will be defined by a unique subdirectory within the
work directory.  Within the base work directory, the first two levels of
subdirectories will be the same as in the current implementation:
``core/configuration``.  However, testcases will be free to determine the
(unique) subdirectory structure beyond this top-most level.  Many existing
testcases will likely stick with the ``resolution/testcase/step`` organization
structure imposed in the existing COMPASS framework, but others may choose a
different way of organizing (and, indeed, many test cases already have given the
``resolution`` subdirectory a name that is seemingly unrelated to the mesh
resolution).  A unique subdirectory for each testcase and step will be provided
as a value in ``testcase['subdir']`` or ``step['subdir']`` within the python
dictionary that describes each testcase or step.  The default ``subdir`` will
be the name of the testcase or step, but each testcase or step can modify this
as appropriate in the ``collect()`` function.

COMPASS will list testcases based on their full paths within the work directory,
since this is they way that they can be uniquely identified.


Design solution: User- and developer-friendly documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: 2020/11/16

Contributors: Xylar Asay-Davis


Documentation using ``sphinx`` and the ``ReadTheDocs`` template will be built
out in a manner similar to what has already been done for:

* `geometric_features <https://mpas-dev.github.io/geometric_features/stable/>`_

* `pyremap <https://mpas-dev.github.io/pyremap/stable/>`_

* `MPAS-Tools <https://mpas-dev.github.io/MPAS-Tools/stable/>`_

* `MPAS-Analysis <https://mpas-dev.github.io/MPAS-Analysis/latest/>`_

The documentation will include:

* A user's guide for listing, setting up, and cleaning up testcase

* A user's guide for regression suites

* More detailed tutorials:

  * Running a test case

  * Running the regression suite

* A section for each core

  * A subsection describing the configurations

    * A sub-subsection for each testcase and its steps

  * A subsection for the core's framework code

* A description of the ``compass`` framework code:

  * for use within testcases

  * for listing, setting up and cleaning up testcases

  * for managing regression test suites

* An automated documentation of the API pulled from docstrings

* A developer's guide for creating new testcases

  * core-specific details for developing new testcases


Design and Implementation
-------------------------

Implementation: short-description-of-implementation-here
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: YYYY/MM/DD

Contributors: (add your name to this list if it does not appear)

This section should detail the plan for implementing the design solution for
requirement XXX. In general, this section is software-centric with a focus on
software implementation. Pseudo code is appropriate in this section. Links to
actual source code are appropriate. Project management items, such as git
branches, timelines and staffing are also appropriate. Pseudo code can be
included via blocks like

.. code-block:: python

   def example_function(foo):
       return foo**2.0


Testing
-------

Testing and Validation: short-description-of-testing-here
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Date last modified: YYYY/MM/DD

Contributors: (add your name to this list if it does not appear)

How will XXX be tested, i.e., how will be we know when we have met requirement
XXX? What testing will be included for use with ``py.test`` for continuous
integration? Description of how testing that requires off-line or specialized
setup will be used.

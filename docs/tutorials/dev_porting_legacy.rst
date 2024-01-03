.. _dev_tutorial_porting_legacy:

Developer Tutorial: Porting a legacy COMPASS test group
=======================================================

This tutorial presents a step-by-step guide to porting a legacy COMPASS test
group to the ``compass`` python package (see the :ref:`glossary` for
definitions of these terms). The tutorial uses the legacy COMPASS test group
``ocean/gotm`` as an example.  We will build a new python package
``compass.ocean.tests.gotm`` for the test group in ``compass``.  The example
test group creates a tiny (4 x 4 cell) doubly periodic mesh and uses it to
test MPAS-Ocean calls to the `General Ocean Turbulence Model (GOTM) <https://gotm.net/>`_.

Getting started
---------------

To begin with, you will need to check out two different branches of compass.
First, you will need the ``legacy`` branch.  Since we will simply be looking
at the code and copying it as needed, you can simply browse the branch directly
on GitHub: https://github.com/MPAS-Dev/compass/tree/legacy/ocean/gotm/2.5km/default.
If you prefer, you can use either of the approaches described in
:ref:`dev_compass_repo` or :ref:`dev_compass_repo_advanced` to clone the repo
and check out the ``legacy`` branch.

Next, you will need to create a new branch from ``main`` for developing the
new test group.  For this purpose, we will stick with the simpler approach in
:ref:`dev_compass_repo` here, but feel free to use the ``worktree`` approach
instead if you are comfortable with it.

.. code-block:: bash

    git clone git@github.com:MPAS-Dev/compass.git add_gotm
    cd add_gotm

Now, you will need to create a conda environment for developing compass, as
described in :ref:`dev_conda_env`.  We will assume a simple situation where
you are working on a "supported" machine and using the default compilers and
MPI libraries, but consult the documentation to make an environment to suit
your needs.

.. code-block:: bash

  # this one will take a while the first time
  ./conda/configure_compass_env.py --conda $HOME/miniforge

If all goes well, you will have a file named ``load_dev_compass_1.0.0*.sh``, where
the details of the ``*`` depend on your specific machine and compilers.  For
example, on Chrysalis, you will have ``load_dev_compass_1.0.0_chrysalis_intel_impi.sh``,
which will be the example used here:

.. code-block:: bash

  source load_dev_compass_1.0.0_chrysalis_intel_impi.sh

Now, we're ready to get the MPAS-Ocean source code from the E3SM repository and
build the MPAS-Ocean executable:

.. code-block:: bash

  # Get the E3SM code -- this one takes a while every time
  git submodule update --init --recursive
  cd E3SM-Project/components/mpas-ocean/
  make intel-mpi
  cd ../../..

The make target will be different depending on the machine and compilers, see
:ref:`dev_supported_machines` or :ref:`dev_other_machines` for the right one
for your machine.

Now, we're ready to start developing!

The legacy COMPASS test group
-----------------------------

...But before we get started, a little background on legacy COMPASS for those
who haven't used it extensively.  In legacy COMPASS, the test group is just a
directory with multiple test cases and, optionally, templates and other files
in it.  Test cases are made up of XML config and template files, and sometimes
additional files like python scripts, python config files, namelist files, and
geojson files.

All test cases have a driver, ``config_driver.xml``, that lists the steps in
the test case.  The ``gotm`` test group that we are using as an example has a
single test case, ``ocean/gotm/2.5km/default``.  Its ``config_driver.xml``
looks like this:

.. code-block:: xml

    <driver_script name="run_test.py">
        <case name="init">
            <step executable="./run.py" quiet="true" pre_message=" * Running init" post_message="     Complete"/>
        </case>
        <case name="forward">
            <step executable="./run.py" quiet="true" pre_message=" * Running forward" post_message="     Complete"/>
        </case>
        <case name="analysis">
            <step executable="./run.py" quiet="true" pre_message=" * Running analysis" post_message="     Complete"/>
        </case>
    </driver_script>

The test case is made up of 3 steps, ``init``, ``forward`` and ``analysis``.
Each has its own XML file.  For example, ``config_init.xml`` looks like this:

.. code-block:: xml

    <?xml version="1.0"?>
    <config case="init">

        <add_executable source="model" dest="ocean_model"/>

        <namelist name="namelist.ocean" mode="init">
            <option name="config_init_configuration">'periodic_planar'</option>
            <option name="config_vert_levels">-1</option>
            <option name="config_periodic_planar_vert_levels">250</option>
            <option name="config_periodic_planar_bottom_depth">15.0</option>
            <option name="config_periodic_planar_velocity_strength">0.0</option>
            <option name="config_ocean_run_mode">'init'</option>
            <option name="config_write_cull_cell_mask">.false.</option>
            <option name="config_vertical_grid">'uniform'</option>
        </namelist>

        <streams name="streams.ocean" keep="immutable" mode="init">
            <stream name="input_init">
                <attribute name="filename_template">mesh.nc</attribute>
            </stream>
            <stream name="output_init">
                <attribute name="type">output</attribute>
                <attribute name="output_interval">0000_00:00:01</attribute>
                <attribute name="clobber_mode">truncate</attribute>
                <attribute name="filename_template">ocean.nc</attribute>
                <add_contents>
                    <member name="input_init" type="stream"/>
                    <member name="layerThickness" type="var"/>
                    <member name="restingThickness" type="var"/>
                    <member name="refBottomDepth" type="var"/>
                    <member name="bottomDepth" type="var"/>
                    <member name="maxLevelCell" type="var"/>
                    <member name="vertCoordMovementWeights" type="var"/>
                    <member name="edgeMask" type="var"/>
                </add_contents>
        </stream>
        </streams>

        <run_script name="run.py">
            <step executable="planar_hex">
                <argument flag="--nx">4</argument>
                <argument flag="--ny">4</argument>
                <argument flag="--dc">2500.0</argument>
                <argument flag="-o">grid.nc</argument>
            </step>
            <step executable="MpasCellCuller.x">
                <argument flag="">grid.nc</argument>
                <argument flag="">culled_mesh.nc</argument>
            </step>
            <step executable="MpasMeshConverter.x">
                <argument flag="">culled_mesh.nc</argument>
                <argument flag="">mesh.nc</argument>
            </step>
            <model_run procs="1" threads="1" namelist="namelist.ocean" streams="streams.ocean"/>
        </run_script>

    </config>

The XML files for the other steps look similar.  We will go through these files
in detail later in the tutorial.

The example test case also has a namelist file used by GOTM (``gotmturb.nml``)
and a python script for plotting the results compared to analytic solutions
(``plot_profile.py``).

Making a new test group
-----------------------

Okay, with those details as a reference point from legacy COMPASS, let's jump
into developing the new test group in ``compass``.  Use any method you like
for editing code.  If you haven't settled on a method and are working on your
own laptop or desktop, you may want to try an integrated development
environment (`PyCharm <https://www.jetbrains.com/pycharm/>`_ is a really nice
one).  They have features to make sure your code adheres to the style required
for compass (see :ref:`dev_style`).  ``vim`` or a similar tool will work fine
on supercomputers.

In ``compass``, the ``gotm`` test group will be a new python package.  We will
make a new ``gotm`` directory in ``compass/ocean/tests``.  In that directory,
we will make a new, initially empty file ``__init__.py``.  Now, ``gotm`` is a
new package in ``compass`` that could be imported as

.. code-block:: python

    from compass.ocean.tests import gotm

Next, let's make a new class for the ``gotm`` test group in ``__init__.py``:

.. code-block:: python

    from compass.testgroup import TestGroup


    class Gotm(TestGroup):
        """
        A test group for General Ocean Turbulence Model (GOTM) test cases
        """
        def __init__(self, mpas_core):
            """
            mpas_core : compass.MpasCore
                the MPAS core that this test group belongs to
            """
            super().__init__(mpas_core=mpas_core, name='gotm')


The method (a function for a class) called ``__init__()`` is the constructor
used to make an instance (an object) representing the test group.  It needs
to know what MPAS Core it belongs to so that is passed in as the ``mpas_core``
argument.  The only thing that happens so far is that the constructor for the
base class ``TestGroup`` gets called.  In the process, we give the test group
the name ``gotm``.  You can take a look at the base class ``TestGroup`` in
``compass/testgroup.py`` if you want.  That's not necessary for the tutorial,
but some new developers have found reading the base class code to be
highly instructive.

Naming conventions in python are that we use
`CamelCase <https://en.wikipedia.org/wiki/Camel_case>`_ for classes, which
always start with a capital letter, and all lowercase, possibly with
underscores, for variable, module, package and function names.  We avoid
all-caps like ``GOTM`` or ``MPAS``, even though these might seem preferable.
(We use ``E3SM`` in a few places because ``E3sm`` was simply too much for us to
bear.)

Our new ``Gotm`` class defines the test group, but so far it doesn't have any
test cases in it.  We'll come back and add them later in the tutorial.  Before
we add a test case, let's make ``compass`` aware that the test group exists.
To do that, we need to open ``compass/ocean/__init__.py``, add an import for
the new test group, and add an instance of the test group to the list of test
groups in the ocean core:

.. code-block:: python
    :emphasize-lines: 5, 24

    from compass.mpas_core import MpasCore
    from compass.ocean.tests.baroclinic_channel import BaroclinicChannel
    from compass.ocean.tests.global_convergence import GlobalConvergence
    from compass.ocean.tests.global_ocean import GlobalOcean
    from compass.ocean.tests.gotm import Gotm
    from compass.ocean.tests.ice_shelf_2d import IceShelf2d
    from compass.ocean.tests.ziso import Ziso


    class Ocean(MpasCore):
        """
        A test group for General Ocean Turbulence Model (GOTM) test cases
        """

        def __init__(self):
            """
            Construct the collection of MPAS-Ocean test cases
            """
            super().__init__(name='ocean')

            self.add_test_group(BaroclinicChannel(mpas_core=self))
            self.add_test_group(GlobalConvergence(mpas_core=self))
            self.add_test_group(GlobalOcean(mpas_core=self))
            self.add_test_group(Gotm(mpas_core=self))
            self.add_test_group(IceShelf2d(mpas_core=self))
            self.add_test_group(Ziso(mpas_core=self))

We make an instance of the ``Gotm`` class and we immediately add it to the
``Ocean`` core's list of test groups.  That's all we need to do.  Now
``compass`` knows about the test group.

Adding a test case
------------------

We'll add a test case called ``default`` to ``gotm``.  Unlike in legacy
COMPASS, we don't need to specify the resolution of the test case.  We want
to encourage as much :ref:`dev_code_sharing` as can reasonably be achieved,
and that typically means that the code for a single test case support multiple
resolutions.

We'll make a ``default`` package within ``compass/ocean/tests/gotm``, again
with an ``__init__.py`` file in it.  As we build out this file, it will play
the same role as ``config_driver.xml`` played in legacy COMPASS, adding the
steps in the test case and running them.

As a starting point, we'll create a new ``Default`` class in this file that
descends from the ``TestCase`` base class (take a look at
``compass/testcase.py`` if you want to see the contents of
:py:class:`compass.testcase.TestCase` if you're interested).

.. code-block:: python

    from compass.testcase import TestCase


    class Default(TestCase):
        """
        The default test case for the General Ocean Turbulence Model (GOTM) test
        group creates an initial condition on a 4 x 4 cell, doubly periodic grid,
        performs a short simulation, then vertical plots of the velocity and
        viscosity.
        """

        def __init__(self, test_group):
            """
            Create the test case

            Parameters
            ----------
            test_group : compass.ocean.tests.gotm.Gotm
                The test group that this test case belongs to
            """
            super().__init__(test_group=test_group, name='default')

As a starting point, we just pass along the test group (``Gotm``) this test
case belongs to on to the base class's constructor (``super().__init__()``)
and give the test case a name, ``default``.


Varying resolution (or other parameters)
----------------------------------------

Since the ``Gotm`` test group only has one test case at one resolution (and the
resolution isn't an important property of the setup----it's using multiple
horizontal grid cells but it's acting like a single column), we will just
hard-code the resolution into this particular test case.  Other test cases,
like those in the baroclinic channel test group, do support multiple
resolutions.  It is typically convenient to define multiple versions of the
test case by passing the resolutions as a parameter to the constructor.

This tutorial won't describe how to do a parameter study.  There will be a
separate tutorial for that purpose.  Instead, what is described here is how to
make different variants of a test case with a list of parameter values.  So
far, this is mostly used to create test cases at different resolutions in
``compass`` but the ``ocean/global_ocean`` test group includes a number of
test cases that vary base on:

* whether ice-shelf cavities are included in the ocean domain

* which initial condition is used

* whether biogeochemistry is included in the initial condition

* which time integrator (RK4 or split-explicit) to use

The details here are not important.  The point is that there is little
restriction on what types of parameters can be used to create variants of
test cases.

Here is an example of how resolution is used in the
``barotropic_channel/default`` test case.  This is just an excerpt:

.. code-block:: python

    from compass.testcase import TestCase


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

In this test case, we make a subdirectory that includes the resolution as well
as the name of the test case, and we store the ``resolution`` in the test case
object itself.  Later on, we can access it with ``self.resolution`` whenever
we need it.  For example, we can use it to determine other parameters of the
simulation.  In the following example, we use nested python dictionaries to
give different parameters for different resolution.  We use the resolution to
pick the right inner dictionary, and then set config options (see
:ref:`config_files`).  This example is a slight modification of
``baroclinic_channel/default``:

.. code-block:: python

    def configure(self):
        """
        Modify the configuration options for this test case.
        """
        resolution = self.resolution
        config = self.config

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

Adding the init step
--------------------

In legacy COMPASS, the other ``config_*.xml`` files besides ``config_driver.xml``
define the step in the test case.  In ``compass``, these are defined in classes
that descend from the ``Step`` base class in modules.  The modules can be
defined within the test case package (if they are unique to the test case)
or in the test group (if they are shared among several test cases).  In this
example, there is only one test case, so we will just put the steps in that
test case's package.  You can browse other ``ocean`` and ``landice`` test cases
to see examples of steps shared across test cases.  The ``baroclinic_channel``
test group is a good place to start.

The ``gotm/default`` test case has 3 steps: ``init``, ``forward`` and
``analysis``.  We'll start with ``init``, which creates the grid and calls
MPAS-Ocean in "init" mode to create the initial condition.  To start with,
we'll just create a new ``Init`` class that descends from ``Step``:

.. code-block:: python

    from compass.step import Step


    class Init(Step):
        """
        A step for creating a mesh and initial condition for General Ocean
        Turbulence Model (GOTM) test cases
        """
        def __init__(self, test_case):
            """
            Create the step

            Parameters
            ----------
            test_case : compass.ocean.tests.gotm.default.Default
                The test case this step belongs to
            """
            super().__init__(test_case=test_case, name='forward', ntasks=1,
                             min_tasks=1, openmp_threads=1)

This pattern is probably starting to look familiar.  The step takes the test
case it belongs to as an input to its constructor, and passes that along to
the base class' version of the constructor, along with the name of the step.
By default, the subdirectory for the step is the same as the step name, but
just like for a test case, you can give the step a more complicated
subdirectory name, possibly with multiple levels of directories.  See the
steps in the ``ocean/global_convergence/cosine_bell`` test case for examples
of this.  The ``init`` step runs on one core (so ``ntasks`` and ``min_tasks``
are both 1) and one thread.

The next step is to define the namelist, streams file, outputs from the step:

.. code-block:: python
    :emphasize-lines: 4, 5, 7, 8, 10, 12, 13

    super().__init__(test_case=test_case, name='forward', ntasks=1,
                     min_tasks=1, openmp_threads=1)

    self.add_namelist_file('compass.ocean.tests.gotm.default',
                           'namelist.init', mode='init')

    self.add_streams_file('compass.ocean.tests.gotm.default',
                          'streams.init', mode='init')

    self.add_model_as_input()

    for file in ['mesh.nc', 'graph.info', 'ocean.nc']:
        self.add_output_file(file)

We will discuss the contents of the namelist and streams files below. By
calling :py:meth:`compass.Step.add_model_as_input()`, we add the MPAS-Ocean
executable as an input to the step (meaning that a symlink to the executable
will be made in the step's work directory, and that the step will fail right
away if the model hasn't been built yet).

Finally, we add outputs from the step.  The outputs are any files produced by
this step that any other step should be allowed to use as inputs.  In this
case, the ``forward`` step needs all three of these files as inputs, which is
how we decided which of the outputs from the test case to include in this list.
``mesh.nc`` is the mesh, ``graph.info`` is the graph file used by
`Metis <http://glaros.dtc.umn.edu/gkhome/metis/metis/overview>`_ to partition
the mesh across processors, and ``ocean.nc`` is the initial condition.

Defining namelist options
~~~~~~~~~~~~~~~~~~~~~~~~~

In ``compass``, there are two main ways to set namelist options for MPAS model
runs and we will demonstrate both in this test case.  First, you can define a
namelist file with the desired values.  This is useful for namelist options that
are always the same for this test case and can't be changed based on config
options from the config file (see above).

The original ``config_init.xml`` contained:

.. code-block:: xml

    <namelist name="namelist.ocean" mode="init">
        <option name="config_init_configuration">'periodic_planar'</option>
        <option name="config_vert_levels">-1</option>
        <option name="config_periodic_planar_vert_levels">250</option>
        <option name="config_periodic_planar_bottom_depth">15.0</option>
        <option name="config_periodic_planar_velocity_strength">0.0</option>
        <option name="config_ocean_run_mode">'init'</option>
        <option name="config_write_cull_cell_mask">.false.</option>
        <option name="config_vertical_grid">'uniform'</option>
    </namelist>

In ``compass`` the formatting is much more similar to the resulting namelist
file.  Here is the ``namelist.init`` file from our example ``gotm/default``
test case:

.. code-block:: none

    config_init_configuration = 'periodic_planar'
    config_vert_levels = -1
    config_periodic_planar_velocity_strength = 0.0
    config_write_cull_cell_mask = .false.
    config_vertical_grid = 'uniform'

We do not need to specify ``config_ocean_run_mode = 'init'`` because this will
be taken care of because we specified ``mode='init'`` when we added the
namelist to the step above.

Though it would be possible, users are not intended to change these to
customize this step of the test case.

Another way to set namelist options is to use a python dictionary and to call
:py:meth:`compass.Step.add_namelist_options()`.  This is the way to handle
namelist options that depend on parameters (such as resolution) that are not
known in advance.

We will show later on that there is yet another way to handle namelist options
that can come from config options, using
:py:meth:`compass.Step.update_namelist_at_runtime()`.  This is why we haven't
yet included the ``config_periodic_planar_vert_levels`` and
``config_periodic_planar_bottom_depth`` options from the legacy test case.

Defining streams
~~~~~~~~~~~~~~~~

Similarly, it is convenient to define input and output streams for MPAS-Ocean
using a streams file, very similar to what you will see when the test case
is set up.  The syntax in ``compass`` for defining streams is a lot simpler
than in legacy COMPASS (where a different XML convention was used to define
streams than the XML of the streams files themselves).  In legacy COMPASS,
the streams for the ``init`` step are defined in ``config_init.xml`` as:

.. code-block:: xml

    <streams name="streams.ocean" keep="immutable" mode="init">
        <stream name="input_init">
            <attribute name="filename_template">mesh.nc</attribute>
        </stream>
        <stream name="output_init">
            <attribute name="type">output</attribute>
            <attribute name="output_interval">0000_00:00:01</attribute>
            <attribute name="clobber_mode">truncate</attribute>
            <attribute name="filename_template">ocean.nc</attribute>
            <add_contents>
                <member name="input_init" type="stream"/>
                <member name="layerThickness" type="var"/>
                <member name="restingThickness" type="var"/>
                <member name="refBottomDepth" type="var"/>
                <member name="bottomDepth" type="var"/>
                <member name="maxLevelCell" type="var"/>
                <member name="vertCoordMovementWeights" type="var"/>
                <member name="edgeMask" type="var"/>
            </add_contents>
        </stream>
    </streams>

In ``compass``, we add a ``streams.init`` file to the ``default`` test case:

.. code-block:: xml

    <streams>

    <immutable_stream name="input_init"
                      filename_template="mesh.nc"/>

    <stream name="output_init"
            type="output"
            output_interval="0000_00:00:01"
            clobber_mode="truncate"
            filename_template="ocean.nc">

        <stream name="input_init"/>
        <var name="layerThickness"/>
        <var name="restingThickness"/>
        <var name="refBottomDepth"/>
        <var name="bottomDepth"/>
        <var name="maxLevelCell"/>
        <var name="vertCoordMovementWeights"/>
        <var name="edgeMask"/>
    </stream>

    </streams>

As in legacy COMPASS, streams that are already defined like ``input_init``
will use the default attributes defined by the MPAS component unless they are
explicitly replaced in the streams file.  On setting up the test case, the
stream in the ``streams.ocean`` file becomes:

.. code-block:: xml

    <immutable_stream name="input_init"
                      type="input"
                      filename_template="mesh.nc"
                      input_interval="initial_only"/>

Defining config options
~~~~~~~~~~~~~~~~~~~~~~~

The remainder of the ``init`` step will consist of defining the ``run()``
method that does the real work of the step.  To make it easier for users to
modify the test case a little bit to suit their needs, we may want to include
parameters in the config file for the test case.  To do this, we can make a
config file with the test group's package, the test case's package, or both.
In our example, we will just add a config file ``defaults.cfg`` to the
``defaults`` test case:

.. code-block:: cfg

    # config options for General Ocean Turbulence Model (GOTM) test cases
    [gotm]

    # the number of grid cells in x and y
    nx = 4
    ny = 4

    # the size of grid cells (m)
    dc = 2500.0

    # the number of vertical levels
    vert_levels = 250

    # the depth of the sea floor (m)
    bottom_depth = 15.0

By default, the domain is 4 x 4 horizontal cells, each 2.5 km in size.  The
ocean is 15 m deep, divided over 250 uniformly spaced levels.  A user could
change any of these parameters before running the ``init`` step to modify
the initial condition (and therefore the rest of the test case).

Since the config file has the same name (``default``) as the test case, it will
be included automatically when the config file is produced when the test case gets
set up.

Defining the run method
~~~~~~~~~~~~~~~~~~~~~~~

With these config options, namelists and streams files defined, we will
implement the ``run()`` method of the ``init`` step to do the rest of the work
for this step.  In legacy COMPASS, the XML for defining the run scrip was:

.. code-block:: xml

    <run_script name="run.py">
        <step executable="planar_hex">
            <argument flag="--nx">4</argument>
            <argument flag="--ny">4</argument>
            <argument flag="--dc">2500.0</argument>
            <argument flag="-o">grid.nc</argument>
        </step>
        <step executable="MpasCellCuller.x">
            <argument flag="">grid.nc</argument>
            <argument flag="">culled_mesh.nc</argument>
        </step>
        <step executable="MpasMeshConverter.x">
            <argument flag="">culled_mesh.nc</argument>
            <argument flag="">mesh.nc</argument>
        </step>
        <model_run procs="1" threads="1" namelist="namelist.ocean" streams="streams.ocean"/>
    </run_script>

In ``compass``,  the equivalent ``run()`` method is:

.. code-block:: python

    from mpas_tools.planar_hex import make_planar_hex_mesh
    from mpas_tools.io import write_netcdf
    from mpas_tools.mesh.conversion import convert, cull

    from compass.model import run_model

    ...

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        logger = self.logger

        section = config['gotm']
        nx = section.getint('nx')
        ny = section.getint('ny')
        dc = section.getfloat('dc')

        dsMesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc, nonperiodic_x=False,
                                      nonperiodic_y=False)
        write_netcdf(dsMesh, 'grid.nc')

        dsMesh = cull(dsMesh, logger=logger)
        dsMesh = convert(dsMesh, graphInfoFileName='graph.info',
                         logger=logger)
        write_netcdf(dsMesh, 'mesh.nc')

        replacements = dict()
        replacements['config_periodic_planar_vert_levels'] = \
            config.get('gotm', 'vert_levels')
        replacements['config_periodic_planar_bottom_depth'] = \
            config.get('gotm', 'bottom_depth')
        self.update_namelist_at_runtime(options=replacements)

        run_model(self)

First, we make a doubly periodic mesh.  Rather than hard-coding the mesh size,
we get the relevant config options.  Legacy COMPASS used the command-line
tool ``planar_hex``, but ``compass`` will typically use the
:py:func:`mpas_tools.planar_hex.make_planar_hex_mesh()` function instead to
avoid the complexity of ``subprocess`` calls and unnecessary file I/O.  The
result is an :py:class:`xarray.Dataset` containing the mesh.

Second, we make sure any land cells are culled by calling the cell culler.
In legacy COMPASS, this is done with the command-line tool ``MpasCellCuller.x``
but in ``compass``, the same can be achieved with
:py:func:`mpas_tools.mesh.conversion.cull()` (which is a wrapper around a
``subprocess``` call to ``MpasCellCuller.x``).  ``cull()`` has
:py:class:`xarray.Dataset` objects as its input an return value, and also
takes a "logger" where it can write output (sometimes a log file and sometimes
directly to the terminal via stdout).  You should always pass ``self.logger``
so ``compass`` can figure out whether a file or stdout is the right place for
output to go.

.. note::

    In this particular test case, there are no land cells defined and
    the mesh is doubly periodic, so the call to the cell culler is probably
    not needed, but it has been retained because many test cases *will* need
    it.

Third, we call :py:func:`mpas_tools.mesh.conversion.convert()` to ensure that the
mesh conforms to the MPAS conventions.  In the legacy COMPASS version, the
equivalent is achieved with a call to ``MpasMeshConverter.x``.  Then, we write
out the mesh to a file ``mesh.nc``.

Fourth, we use :py:meth:`compass.Step.update_namelist_at_runtime()`to update
the ``config_periodic_planar_vert_levels`` and
``config_periodic_planar_bottom_depth`` namelist options based on the
``vert_levels`` and ``bottom_depth`` config options.  Since config options come
from a config file in the test case's work directory (symlinked into each
step's work directory), a user may have decided to change these config options
before running the test case so we update the namelist file right before
running the model.

Finally, we run MPAS-Ocean by calling :py:func:`compass.model.run_model()`.
We pass the step itself as an argument because this is how ``compass`` knows
how many cores and threads to run on, which namelist and streams files to use,
which MPAS core this test case belongs to, and so on.

Adding the forward step
-----------------------

The ``Forward`` step will be conceptually similar to the ``Init`` step.  Again,
we make a ``Forward`` class that descends from ``Step`` with a constructor that
calls the base constructor with the name of the step as well as the requested
number of cores, minimum number of cores, and number of threads:

.. code-block:: python

    from compass.step import Step


    class Forward(Step):
        """
        A step for performing forward MPAS-Ocean runs as part of General Ocean
        Turbulence Model (GOTM) test cases.
        """
        def __init__(self, test_case):
            """
            Create a new test case

            Parameters
            ----------
            test_case : compass.ocean.tests.gotm.default.Default
                The test case this step belongs to

            """
            super().__init__(test_case=test_case, name='forward', ntasks=1,
                             min_tasks=1, openmp_threads=1)

The following XML from legacy COMPASS:

.. code-block:: xml

    <add_link source="../init/ocean.nc" dest="init.nc"/>
    <add_link source="../init/mesh.nc" dest="mesh.nc"/>
    <add_link source="../init/graph.info" dest="graph.info"/>

is replaced by these method calls within the step's constructor in ``compass``:

.. code-block:: python

    self.add_input_file(filename='mesh.nc', target='../init/mesh.nc')
    self.add_input_file(filename='init.nc', target='../init/ocean.nc')
    self.add_input_file(filename='graph.info', target='../init/graph.info')

As in ``Init``, we want to make a link to the MPAS-Ocean executable.  The
legacy COMPASS version of this was:

.. code-block:: xml

    <add_executable source="model" dest="ocean_model"/>

and the ``compass`` version is:

.. code-block:: python

        self.add_model_as_input()

This step also needs to make a link to a namelist file that is specific to the
GOTM library called from within MPAS-Ocean (i.e. a different namelist than the
MPAS-Ocean ``namelist.ocean`` file).  In legacy COMPASS, a symlink from the
script test directory to the working directory was accomplished with:

.. code-block:: xml

    <copy_file source_path="script_test_dir" source="gotmturb.nml" dest="gotmturb.nml"/>

In ``compass``, this becomes:

.. code-block:: python

    self.add_input_file(filename='gotmturb.nml', target='gotmturb.nml',
                        package='compass.ocean.tests.gotm.default')

The target is the file ``gotmturb.nml`` that we will place in the ``default``
package that we're currently working on.

Finally, we'll add an output file, appropriately enough called ``output.nc``.

.. code-block:: python

    self.add_output_file(filename='output.nc')

The complete constructor looks like:

.. code-block:: python

    def __init__(self, test_case):
        """
        Create a new test case

        Parameters
        ----------
        test_case : compass.ocean.tests.gotm.default.Default
            The test case this step belongs to

        """
        super().__init__(test_case=test_case, name='forward', ntasks=1,
                         min_tasks=1, openmp_threads=1)
        self.add_namelist_file('compass.ocean.tests.gotm.default',
                               'namelist.forward')

        self.add_streams_file('compass.ocean.tests.gotm.default',
                              'streams.forward')

        self.add_input_file(filename='mesh.nc', target='../init/mesh.nc')
        self.add_input_file(filename='init.nc', target='../init/ocean.nc')
        self.add_input_file(filename='graph.info', target='../init/graph.info')

        self.add_input_file(filename='gotmturb.nml', target='gotmturb.nml',
                            package='compass.ocean.tests.gotm.default')

        self.add_model_as_input()

        self.add_output_file(filename='output.nc')

We will just copy the file ``gotmturb.nml`` from the legacy test case as it
is.

The MPAS-Ocean namelist file ``namelist.forward`` contains the same contents
as the legacy XML:

.. code-block:: xml

    <namelist name="namelist.ocean" mode="forward">
        <option name="config_ocean_run_mode">'forward'</option>
        <option name="config_dt">'000:00:25'</option>
        <option name="config_btr_dt">'000:00:25'</option>
        <option name="config_time_integrator">'split_explicit'</option>
        <option name="config_run_duration">'0000_12:00:00'</option>
        <option name="config_zonal_ssh_grad">-1.0e-5</option>
        <option name="config_pressure_gradient_type">'constant_forced'</option>
        <option name="config_use_cvmix">.false.</option>
        <option name="config_use_gotm">.true.</option>
        <option name="config_gotm_namelist_file">'gotmturb.nml'</option>
        <option name="config_gotm_constant_bottom_drag_coeff">1.73e-2</option>
        <option name="config_use_implicit_bottom_drag">.true.</option>
        <option name="config_implicit_bottom_drag_coeff">1.73e-2</option>
    </namelist>

but in a simpler, more readable form in ``namelist.forward`` in the
``gotm/default`` test case in ``compass``:

.. code-block:: none

    config_dt = '000:00:25'
    config_btr_dt = '000:00:25'
    config_time_integrator = 'split_explicit'
    config_run_duration = '0000_12:00:00'
    config_zonal_ssh_grad = -1.0e-5
    config_pressure_gradient_type = 'constant_forced'
    config_use_cvmix = .false.
    config_use_gotm = .true.
    config_gotm_namelist_file = 'gotmturb.nml'
    config_gotm_constant_bottom_drag_coeff = 1.73e-2
    config_use_implicit_bottom_drag = .true.
    config_implicit_bottom_drag_coeff = 1.73e-2

We omit ``config_ocean_run_mode = 'forward'`` because this is taken care of
by ``compass`` when we add a namelist with the keyword argument
``mode='forward'`` (which is the default mode).

Similarly, the legacy definition of the streams:

.. code-block:: xml

    <streams name="streams.ocean" keep="immutable" mode="forward">
        <stream name="mesh">
            <attribute name="filename_template">mesh.nc</attribute>
        </stream>
        <stream name="input">
            <attribute name="filename_template">init.nc</attribute>
        </stream>
        <stream name="output">
            <attribute name="type">output</attribute>
            <attribute name="filename_template">output.nc</attribute>
            <attribute name="output_interval">0000-00-00_00:10:00</attribute>
            <attribute name="clobber_mode">truncate</attribute>
            <add_contents>
                <member name="velocityZonal" type="var"/>
                <member name="velocityMeridional" type="var"/>
                <member name="vertViscTopOfCell" type="var"/>
                <member name="mesh" type="stream"/>
                <member name="xtime" type="var"/>
                <member name="normalVelocity" type="var"/>
                <member name="layerThickness" type="var"/>
            </add_contents>
        </stream>
    </streams>

takes this simpler form in ``streams.forward`` in ``compass`` that is nearly
identical to the full streams file in the work directory:

.. code-block:: xml

    <streams>

    <immutable_stream name="mesh"
                      filename_template="mesh.nc"/>

    <immutable_stream name="input"
                      filename_template="init.nc"/>

    <stream name="output"
            type="output"
            filename_template="output.nc"
            output_interval="0000-00-00_00:10:00"
            clobber_mode="truncate">

        <stream name="mesh"/>
        <var name="velocityZonal"/>
        <var name="velocityMeridional"/>
        <var name="vertViscTopOfCell"/>
        <var name="xtime"/>
        <var name="normalVelocity"/>
        <var name="layerThickness"/>
    </stream>

    </streams>

The run script from the legacy test case:

.. code-block:: xml

    <run_script name="run.py">
        <model_run procs="1" threads="1" namelist="namelist.ocean" streams="streams.ocean"/>
    </run_script>

becomes:

.. code-block:: python

    from compass.model import run_model

    ...

    def run(self):
        """
        Run this step of the test case
        """
        run_model(self)


Adding the analysis step
------------------------

The legacy ``analysis`` step is defined like this:

.. code-block:: xml

    <config case="analysis">
        <add_link source="../forward/output.nc" dest="output.nc"/>
        <add_link source_path="script_test_dir" source="plot_profile.py" dest="plot_profile.py"/>

        <run_script name="run.py">
            <step executable="./plot_profile.py">
            </step>
        </run_script>
    </config>

Symlinks are created to a plotting script and the output from the ``forward``
step, and then the plot script is run.

An identical approach could be used in ``compass`` but it is not the preferred
approach.  Instead, we prefer to use function calls in place of calling
python scripts via subprocesses whenever possible.  One major reason for this
is that having python scripts within a python package is confusing -- there is
not a clear way to know that they aren't python modules within the package,
but instead are meant to be symlinked and run elsewhere.  Another is that
these scripts typically don't reuse code very well, nor is it easy to use
config options from the test case within them. For all these reasons, we will
demonstrate how to convert the ``plot_profile.py`` script into a function
instead.

We start out with the same structure as in the other two steps:

.. code-block:: python

    from compass.step import Step


    class Analysis(Step):
        """
        A step for plotting the results of the default General Ocean Turbulence
        Model (GOTM) test case
        """
        def __init__(self, test_case):
            """
            Create a new test case

            Parameters
            ----------
            test_case : compass.ocean.tests.gotm.default.Default
                The test case this step belongs to

            """
            super().__init__(test_case=test_case, name='analysis', ntasks=1,
                             min_tasks=1, openmp_threads=1)

As before, we will define the inputs and outputs.  There will be no namelists
or streams files, nor an MPAS executable because we will not be calling
MPAS-Ocean in this step:

.. code-block:: python

    self.add_input_file(filename='output.nc', target='../forward/output.nc')

    self.add_output_file(filename='velocity_profile.png')
    self.add_output_file(filename='viscosity_profile.png')

Next, we will put the contents of the original ``plot_profile.py`` into the
``run()`` method:

.. code-block:: python

    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt

    ...

    def run(self):
        """
        Run this step of the test case
        """

        # render statically by default
        plt.switch_backend('agg')

        # constants
        kappa = 0.4
        z0b = 1.5e-3
        gssh = 1e-5
        g = 9.81
        h = 15
        # load output
        ds = xr.open_dataset('output.nc')
        # velocity
        u = ds.velocityZonal.isel(Time=-1, nCells=0).values
        # viscosity
        nu = ds.vertViscTopOfCell.isel(Time=-1, nCells=0).values
        # depth
        bottom_depth = ds.refBottomDepth.values
        z = np.zeros_like(bottom_depth)
        z[0] = -0.5*bottom_depth[0]
        z[1:] = -0.5*(bottom_depth[0:-1]+bottom_depth[1:])
        zi = np.zeros(bottom_depth.size+1)
        zi[0] = 0.0
        zi[1:] = -bottom_depth[0:]
        # analytical solution
        ustarb = np.sqrt(g*h*gssh)
        u_a = ustarb/kappa*np.log((z0b+z+h)/z0b)
        nu_a = -ustarb/h*kappa*(z0b+z+h)*z
        # infered drag coefficient
        cd = ustarb**2/u_a[-1]**2
        self.logger.info('C_d = {:6.4g}'.format(cd))
        # plot velocity
        plt.figure()
        plt.plot(u_a, z, 'k--', label='Analytical')
        plt.plot(u, z, 'k-', label='GOTM')
        plt.xlabel('Velocity (m/s)')
        plt.ylabel('Depth (m)')
        plt.legend()
        plt.savefig('velocity_profile.png')
        # plot viscosity
        plt.figure()
        plt.plot(nu_a, z, 'k--', label='Analytical')
        plt.plot(nu, zi, 'k-', label='GOTM')
        plt.xlabel('Viscosity (m$^2$/s)')
        plt.ylabel('Depth (m)')
        plt.legend()
        plt.savefig('viscosity_profile.png')

This particular function doesn't need it, but we could access config options
from ``self.config`` if they would be useful in the analysis.  ``print()``
statements should generally be replaced with ``self.logger.info()`` or other
logger calls (but ``print()`` statements are still okay, and will be captured
in log files rather than going to the terminal when multiple steps or test
cases are running).

Updating the test case and test group
-------------------------------------

The nearly final steps are to add the steps to the test case, and then the test
case to the test group:

.. code-block:: python
    :emphasize-lines: 2, 3, 4, 26, 27, 28

    from compass.testcase import TestCase
    from compass.ocean.tests.gotm.default.init import Init
    from compass.ocean.tests.gotm.default.forward import Forward
    from compass.ocean.tests.gotm.default.analysis import Analysis


    class Default(TestCase):
        """
        The default test case for the General Ocean Turbulence Model (GOTM) test
        group creates an initial condition on a 4 x 4 cell, doubly periodic grid,
        performs a short simulation, then vertical plots of the velocity and
        viscosity.
        """

        def __init__(self, test_group):
            """
            Create the test case

            Parameters
            ----------
            test_group : compass.ocean.tests.gotm.Gotm
                The test group that this test case belongs to
            """
            super().__init__(test_group=test_group, name='default')

            self.add_step(Init(test_case=self))
            self.add_step(Forward(test_case=self))
            self.add_step(Analysis(test_case=self))

and then

.. code-block:: python
    :emphasize-lines: 2, 17

    from compass.testgroup import TestGroup
    from compass.ocean.tests.gotm.default import Default


    class Gotm(TestGroup):
        """
        A test group for General Ocean Turbulence Model (GOTM) test cases
        """

        def __init__(self, mpas_core):
            """
            mpas_core : compass.MpasCore
                the MPAS core that this test group belongs to
            """
            super().__init__(mpas_core=mpas_core, name='gotm')

            self.add_test_case(Default(test_group=self))


Adding validation
-----------------

The legacy ``gotm/2.5km/default`` test case didn't include any
:ref:`dev_validation` but it is a very good idea to include some.  This way,
the test case can be used as part of a regression suite to determine if
unexpected changes have been introduced into the code it tests.  To perform
validation, we override the ``validate`` method from the base ``TestCase``
class in ``Default`` as follows:

.. code-block:: python

    from compass.validate import compare_variables

    ...

    def validate(self):
        """
        Validate variables against a baseline
        """
        compare_variables(test_case=self,
                          variables=['layerThickness', 'normalVelocity'],
                          filename1='forward/output.nc')

If the user ran the ``forward`` step as part of this test case (sometimes they
might run only some of the steps), the call to
:py:func:`from compass.validate.compare_variables()` will check whether
variables ``layerThickness`` and ``normalVelocity`` are exactly the same in
this run as they were in a previous run if a baseline run was provided when the
test case got set up (see :ref:`test_suites`).

Set up and run
--------------

You're all set!  You should be able to see your new test case when you run
``compass list``, set it up by running ``compass setup``, and run it by running
``compass run`` within the work directory.  See :ref:`dev_command_line` for
more on that process.

Documentation
-------------

Make sure to add some documentation of your new test group.  You need to add
all of the functions, classes and methods to the API documentation in
``docs/developers_guide/<core>/api.rst``, following the examples for other
test groups.  You also need to add a file to both the user's guide and the
developer's guide describing the test group and its test cases and steps.

For the user's guide, create a file
``docs/users_guide/<core>/test_groups/<test_group>.rst``.  In that file, you
should describe what the test group and what its test cases do in a way that would
be relevant for a user wanting to run the test case and look at the output.
This file should include a section giving the config options for the test case
and describing what they are used for, so that users know how to modify them
if they want to.  Add ``<test_group>`` in the appropriate place (in
alphabetical order) in the list of test groups in the file
``docs/users_guide/<core>/test_groups/index.rst``.

For the developer's guide, create a file
``docs/developers_guide/<core>/test_groups/<test_group>.rst``. In this file,
you will describe the test group, its test cases and steps in a way that is
relevant to developers who might want to modify the code or use it as an
example for developing their own test cases.  Currently, the descriptions are
brief in part because of the daunting task of documenting nearly 100 test cases
but should be fleshed out as time goes on.  It would help new developers if
newly added test cases are documented well. Add ``<test_group>`` in the
appropriate place (in alphabetical order) in the list of test groups in
``docs/developers_guide/<core>/test_groups/index.rst``.

At this point, you are ready to make a pull request with the ported test group!
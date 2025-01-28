.. _dev_tutorial_add_test_group:

Developer Tutorial: Adding a new test group
===========================================

This tutorial presents a step-by-step guide to adding a new test group to the
``compass`` python package (see the :ref:`glossary` for definitions of these
terms).  In this tutorial, I will use the :ref:`dev_ocean_baroclinic_channel`
as an example.  This test group was actually ported from :ref:`legacy_compass`,
roughly in the manner described in :ref:`dev_tutorial_porting_legacy`.  But we
will use it to describe the process for creating a test group from scratch.

.. _dev_tutorial_add_test_group_getting_started:

Getting started
---------------

To begin with, you will need to check out the compass repo and create  a new
branch from ``main`` for developing the new test group.  For this purpose, we
will stick with the simpler approach in :ref:`dev_compass_repo` here, but feel
free to use the ``worktree`` approach instead if you are comfortable with it.

.. code-block:: bash

    git clone git@github.com:MPAS-Dev/compass.git add_baroclinic_channel
    cd add_baroclinic_channel

Now, you will need to create a conda environment for developing compass, as
described in :ref:`dev_conda_env`.  We will assume a simple situation where
you are working on a "supported" machine and using the default compilers and
MPI libraries, but consult the documentation to make an environment to suit
your needs.

.. code-block:: bash

  # this one will take a while the first time
  ./conda/configure_compass_env.py --conda $HOME/miniforge

If you don't already have Miniforge3 installed in the directory pointed to by
``--conda``, it will be installed automatically for you.  If all goes well, you
will have a file named ``load_dev_compass_1.0.0*.sh``, where the details of the
``*`` depend on your specific machine and compilers.  For example, on
Chrysalis, you will have ``load_dev_compass_1.0.0_chrysalis_intel_impi.sh``,
which will be the example used here:

.. code-block:: bash

  source load_dev_compass_1.0.0_chrysalis_intel_impi.sh

Now, we're ready to get the MPAS-Ocean source code from the E3SM repository:

.. code-block:: bash

  # Get the E3SM code -- this one takes a while every time
  git submodule update --init --recursive

If your test group will require development in E3SM in addition to compass,
you will want to create a branch (possibly with ``git worktree``) for your
development there as well:

.. code-block:: bash

  cd E3SM-Project
  git fetch --all -p
  git branch xylar/mpas-ocean/add-baroclinic-channel origin/main
  git switch xylar/mpas-ocean/add-baroclinic-channel
  cd ../

.. note::

    E3SM has some pretty strict requirements on branch names.  If you are using
    your own fork of E3SM, you should start your branch name with the component
    you are developing (in this case ``mpas-ocean``).  If you wish to push your
    branch to the E3SM repo, you need to begin the branch name with your GitHub
    username (``xylar`` in this example), followed by the component name.  In
    either case, the branch name needs to be all lowercase, separated by
    hyphens, and to describe the work to be done.


Next, we're ready to build the MPAS-Ocean executable:

.. code-block:: bash

  cd E3SM-Project/components/mpas-ocean/
  make intel-mpi
  cd ../../..

The make target will be different depending on the machine and compilers, see
:ref:`dev_supported_machines` or :ref:`dev_other_machines` for the right one
for your machine.

Now, we're ready to start developing!

.. _dev_tutorial_add_test_group_make_test_group:

Making a new test group
-----------------------

Use any method you like for editing code.  If you haven't settled on a method
and are working on your own laptop or desktop, you may want to try an
integrated development environment (`PyCharm <https://www.jetbrains.com/pycharm/>`_
is a really nice one).  They have features to make sure your code adheres to
the style required for compass (see :ref:`dev_style`).  ``vim`` or a similar
tool will work fine on supercomputers.

Your new test group will be a new python package within the MPAS core
(``ocean`` here).  For this example, we create a new ``baroclinic_channel``
directory in ``compass/ocean/tests``.  In that directory, we will make a new
file called ``__init__.py`` that will initially be empty.  That's all it takes
to make ``baroclinic_channel`` a new package in ``compass``.  It can be
imported with:

.. code-block:: python

    from compass.ocean.tests import baroclinic_channel

Each test group in ``compass`` is a class that descends from the
:py:class:`compass.testgroup.TestGroup` class.  Let's make a new class for the
``baroclinic_channel`` test group in ``__init__.py``:

.. code-block:: python

    from compass.testgroup import TestGroup


    class BaroclinicChannel(TestGroup):
        """
        A test group for baroclinic channel test cases
        """
        def __init__(self, mpas_core):
            """
            mpas_core : compass.MpasCore
                the MPAS core that this test group belongs to
            """
            super().__init__(mpas_core=mpas_core, name='baroclinic_channel')


The method (a function for a class) called ``__init__()`` is the constructor
used to make an instance (an object) representing the test group.  It needs
to know what MPAS Core it belongs to so that is passed in as the ``mpas_core``
argument.  The only thing that happens so far is that the constructor for the
base class ``TestGroup`` gets called.  In the process, we give the test group
the name ``baroclinic_channel``.  You can take a look at the base class
``TestGroup`` in ``compass/testgroup.py`` if you want.  That's not necessary
for the tutorial, but some new developers have found reading the base class
code (particularly for ``TestCase`` and ``Step``) to be highly instructive.

Naming conventions in python are that we use
`CamelCase <https://en.wikipedia.org/wiki/Camel_case>`_ for classes, which
always start with a capital letter, and all lowercase, possibly with
underscores, for variable, module, package and function names.  We avoid
all-caps like ``MPAS``, even though this might seem preferable. (We use
``E3SM`` in a few places because ``E3sm`` looks really awkward.)

Our new ``BaroclinicChannel`` class defines the test group, but so far it
doesn't have any test cases in it.  We'll come back and add them later in the
tutorial.  Before we add a test case, let's make ``compass`` aware that the
test group exists. To do that, we need to open ``compass/ocean/__init__.py``,
add an import for the new test group, and add an instance of the test group to the list of test
groups in the ocean core:

.. code-block:: python
    :emphasize-lines: 2, 21

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

We make an instance of the ``BaroclinicChannel`` class and we immediately add
it to the ``Ocean`` core's list of test groups.  That's all we need to do.  Now
``compass`` knows about the test group.

.. _dev_tutorial_add_test_group_add_default:

Adding a "default" test case
----------------------------

We'll add a test case called ``default`` to ``baroclinic_channel`` by making a
``default`` package within ``compass/ocean/tests/baroclinic_channel``.  First,
we make the directory ``compass/ocean/tests/baroclinic_channel/default``, then
we add an empty ``__init__.py`` file into it. As a starting point, we'll create
a new ``Default`` class in this file that descends from the
:py:class:`compass.testcase.TestCase` base class (take a look at
``compass/testcase.py`` if you want to see the contents of ``TestCase`` if
you're interested).

.. code-block:: python

    from compass.testcase import TestCase


    class Default(TestCase):
        """
        The default test case for the baroclinic channel test group simply creates
        the mesh and initial condition, then performs a short forward run on 4
        cores.
        """

        def __init__(self, test_group):
            """
            Create the test case

            Parameters
            ----------
            test_group : compass.ocean.tests.baroclinic_channel.BaroclinicChannel
                The test group that this test case belongs to
            """
            name = 'default'
            super().__init__(test_group=test_group, name=name)

As a starting point, we just pass along the test group (``BaroclinicChannel``)
this test case belongs to on to the base class's constructor
(``super().__init__()``) and give the test case a name, ``default``.

Varying resolution (or other parameters)
----------------------------------------

The test cases in the baroclinic channel test group support multiple
resolutions.  In test groups like this one, it is typically convenient to
define multiple versions of the test case by passing the resolution as a
parameter to the constructor.

This tutorial won't describe how to do a parameter study.  There is a separate
tutorial for that purpose: :ref:`dev_tutorial_add_param_study`. Instead, what
is described here is how to make different variants of a test case with a list
of parameter values that a user cannot easily change.  So far, this is mostly
used to create test cases at different resolutions in ``compass`` but the
``compass/ocean/tests/global_ocean`` test group includes a number of test
cases that vary base on:

* whether ice-shelf cavities are included in the ocean domain

* which initial condition is used

* whether biogeochemistry is included in the initial condition

* which time integrator (RK4 or split-explicit) to use

The particular details of these parameters are not important.  The point is
that there is little restriction on what types of parameters can be used to
create variants of a test case.

Three resolutions supported in ``baroclinic_channel`` test group: ``'10km'``,
``'4km'`` and ``'1km'``.  We add resolution as a parameter to the ``default``
test case:

.. code-block:: python
    :emphasize-lines: 10-13, 16, 25-26, 29-32

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

We indicate that the work directory should include a subdirectory for
resolution as well as the name of the test case, and we store the ``resolution``
as an attribute of the test case object itself (``self.resolution``).  We add
resolution to the docstring for both the class (where we describe the
``resolution`` attribute) and the constructor (where we describe the
``resolution`` argument or parameter).  Later on in the test case in other
methods, we will access the resolution with ``self.resolution`` whenever we
need it.

The ``default`` test case doesn't do anything yet because we haven't added
any steps, but let's add it to the ``baroclinic_channel`` test group so we can
see how the resolution will be specified.  We add the following to the file
``__init__.py`` that defines the ``BaroclinicChannel`` test group:

.. code-block:: python
    :emphasize-lines: 2, 16-18

    from compass.testgroup import TestGroup
    from compass.ocean.tests.baroclinic_channel.default import Default


    class BaroclinicChannel(TestGroup):
        """
        A test group for baroclinic channel test cases
        """
        def __init__(self, mpas_core):
            """
            mpas_core : compass.MpasCore
                the MPAS core that this test group belongs to
            """
            super().__init__(mpas_core=mpas_core, name='baroclinic_channel')

            for resolution in ['10km']:
                self.add_test_case(
                    Default(test_group=self, resolution=resolution))

The ``default`` test case (and most other test cases in this test group) is
for regression testing and will only be run at the coarsest resolution, 10 km.

Adding the initial_state step
-----------------------------

In ``compass``, steps are defined in python modules in classes that descend
from the :py:class:`compass.step.Step` base class.  The modules can be defined
within the test case package (if they are unique to the test case) or in the
test group (if they are shared among several test cases).  In this example,
we have only added one test case (``default``) so far but we anticipate
adding more.  All test cases will require a similar ``initial_state`` step, so
it makes sense for the ``initial_state.py`` module to be located in the test
group's package to promote :ref:`dev_code_sharing`.

The ``initial_state`` step will create the MPAS mesh and initial condition for
the test case.  To start with, we'll just create a new ``InitialState`` class
that descends from ``Step``:

.. code-block:: python

    from compass.step import Step


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
            Create the step

            Parameters
            ----------
            test_case : compass.TestCase
                The test case this step belongs to

            resolution : str
                The resolution of the test case
            """
            super().__init__(test_case=test_case, name='initial_state')
            self.resolution = resolution


This pattern is probably starting to look familiar.  The step takes the test
case it belongs to as an input to its constructor, and passes that along to
the base class' version of the constructor, along with the name of the step.
By default, the subdirectory for the step is the same as the step name, but
just like for a test case, you can give the step a more complicated
subdirectory name, possibly with multiple levels of directories.  This is
particularly important for parameter studies, an example of which can be seen
in the ``compass/ocean/tests/global_convergence/cosine_bell`` test case.

Creating a horizontal mesh
~~~~~~~~~~~~~~~~~~~~~~~~~~

While :ref:`legacy_compass` typically used MPAS-Ocean itself to define initial
conditions for test cases (by running the model "init" mode), we have found
that it is usually much easier to set up a mesh and define an initial condition
in python.  The thinking behind "init" mode in MPAS-Ocean was that MPI
parallelism and MPAS computations like gradients or the equation of state might
be useful to have.  In practice, these features are seldom needed and are
outweighed by the fact that the MPAS framework is not well equipped to read in
NetCDF datasets on regular grids or interpolate them, and that the development
time needed to create an initial condition in MPAS-Ocean is typically
substantially longer than in python.

The ``run()`` method of the ``initial_state`` step does the actual work of
creating a mesh and initial condition. Below, We will present the method in 3
pieces.  Please browse the code yourself to see the complete method.

First, we create a regular, planar, hexagonal mesh that is periodic in the x
direction but not in y. The number of cells in mesh comes from config options
``nx`` and ``ny``, and the physical size of each cell from the config option
``dc``, as discussed below:

.. code-block:: python

    from mpas_tools.planar_hex import make_planar_hex_mesh
    from mpas_tools.io import write_netcdf
    from mpas_tools.mesh.conversion import convert, cull

    ...

        def run(self):
            """
            Run this step of the test case
            """
            config = self.config
            logger = self.logger

            section = config['baroclinic_channel']
            nx = section.getint('nx')
            ny = section.getint('ny')
            dc = section.getfloat('dc')

            dsMesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc, nonperiodic_x=False,
                                          nonperiodic_y=True)
            write_netcdf(dsMesh, 'base_mesh.nc')

            dsMesh = cull(dsMesh, logger=logger)
            dsMesh = convert(dsMesh, graphInfoFileName='culled_graph.info',
                             logger=logger)
            write_netcdf(dsMesh, 'culled_mesh.nc')

            ...

We will continue with the ``run()`` method below, but first it is worth
discussing how to test the config options used to generate the horizontal mesh.

Setting config options based on resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We need a way to get the number of mesh cells and the size of these cells for
a given resolution.  We could add these to the test case directly but it is
often a good idea to add them to a config file instead.  This way, a user
could alter these defaults with relative ease, allowing them to explore
variations on the test case.

To set config options (see :ref:`config_files`) for the test case, we define
a ``configure()`` method in the test case.  All the steps of a test case share
the same config file, so there isn't a ``configure()`` method for individual
steps.  The idea is that it isn't very convenient for a user to have to edit a
different config file for each step, so there should be one for the whole test
case.  (Even editing config files for individual test cases is kind of a pain,
so it can be more convenient to set config options in a "user"
:ref:`config_files` before setting up the test case.) Here, we use nested
python dictionaries to give different parameters for different resolution.  We
use the resolution to pick the right inner dictionary, and then set the config
options:

.. code-block:: python

    class Default(TestCase):

    ...

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

As noted above, we only support 3 resolutions (``'10km'``, ``'4km'`` and
``'1km'``), and each has an associated with mesh sizes (``nx`` and ``ny``)
and physical cell size (``dc``).  These are added to the ``baroclinic_channel``
section of the config file.  The ``configure()`` method will get called
automatically when the test case gets set up, so these config options will show
up in the config file that gets put in the test case's work directory and
symlinked into each steps work directory.

Creating a vertical coordinate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This step is specific to test groups in the ``ocean`` MPAS core.  Those in the
``landice`` core use a different approach to creating vertical coordinates.
Returning to the ``run()`` method in the ``initial_state`` step, the code
snippet below is an example of how to make use of the
:ref:`dev_ocean_framework` to create the vertical coordinate:

.. code-block:: python

    import xarray
    import numpy
    ...

    from compass.ocean.vertical import init_vertical_coord
        ...

        def run(self):
            ...

            ds = dsMesh.copy()
            xCell = ds.xCell

            bottom_depth = config.getfloat('vertical_grid', 'bottom_depth')

            ds['bottomDepth'] = bottom_depth * xarray.ones_like(xCell)
            ds['ssh'] = xarray.zeros_like(xCell)

            init_vertical_coord(config, ds)

This step, too, relies on config options, this time from the ``vertical_grid``
section (see :ref:`dev_ocean_framework_vertical` for more on this). The easiest
way to define these is to put a config file into the test group or test case's
python package.  In this case, we know that these config options are going to
be used across many test cases so it makes sense to put them directly in the
``baroclinic_channel`` test group.  If we put them in a file called
``baroclinic_channel.cfg``, they will automatically get read in and added to
the config file for each test case as part of setup:

.. code-block:: cfg

    # Options related to the vertical grid
    [vertical_grid]

    # the type of vertical grid
    grid_type = uniform

    # Number of vertical levels
    vert_levels = 20

    # Depth of the bottom of the ocean
    bottom_depth = 1000.0

    # The type of vertical coordinate (e.g. z-level, z-star)
    coord_type = z-star

    # Whether to use "partial" or "full", or "None" to not alter the topography
    partial_cell_type = None

    # The minimum fraction of a layer for partial cells
    min_pc_fraction = 0.1

    ...

Creating an initial condition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The final part of the ``run()`` method in the ``initial_state`` step is to
define the initial condition:

.. code-block:: python

    import xarray
    import numpy
    ...

    from compass.ocean.vertical import init_vertical_coord
        ...

        def run(self):
            ...

            section = config['baroclinic_channel']
            use_distances = section.getboolean('use_distances')
            gradient_width_dist = section.getfloat('gradient_width_dist')
            gradient_width_frac = section.getfloat('gradient_width_frac')
            bottom_temperature = section.getfloat('bottom_temperature')
            surface_temperature = section.getfloat('surface_temperature')
            temperature_difference = section.getfloat('temperature_difference')
            salinity = section.getfloat('salinity')
            coriolis_parameter = section.getfloat('coriolis_parameter')

            ...

            xMin = xCell.min().values
            xMax = xCell.max().values
            yMin = yCell.min().values
            yMax = yCell.max().values

            yMid = 0.5*(yMin + yMax)
            xPerturbMin = xMin + 4.0 * (xMax - xMin) / 6.0
            xPerturbMax = xMin + 5.0 * (xMax - xMin) / 6.0

            if use_distances:
                perturbationWidth = gradient_width_dist
            else:
                perturbationWidth = (yMax - yMin) * gradient_width_frac

            yOffset = perturbationWidth * numpy.sin(
                6.0 * numpy.pi * (xCell - xMin) / (xMax - xMin))

            temp_vert = (bottom_temperature +
                         (surface_temperature - bottom_temperature) *
                         ((ds.refZMid + bottom_depth) / bottom_depth))

            frac = xarray.where(yCell < yMid - yOffset, 1., 0.)

            mask = numpy.logical_and(yCell >= yMid - yOffset,
                                     yCell < yMid - yOffset + perturbationWidth)
            frac = xarray.where(mask,
                                1. - (yCell - (yMid - yOffset)) / perturbationWidth,
                                frac)

            temperature = temp_vert - temperature_difference * frac
            temperature = temperature.transpose('nCells', 'nVertLevels')

            # Determine yOffset for 3rd crest in sin wave
            yOffset = 0.5 * perturbationWidth * numpy.sin(
                numpy.pi * (xCell - xPerturbMin) / (xPerturbMax - xPerturbMin))

            mask = numpy.logical_and(
                numpy.logical_and(yCell >= yMid - yOffset - 0.5 * perturbationWidth,
                                  yCell <= yMid - yOffset + 0.5 * perturbationWidth),
                numpy.logical_and(xCell >= xPerturbMin,
                                  xCell <= xPerturbMax))

            temperature = (temperature +
                           mask * 0.3 * (1. - ((yCell - (yMid - yOffset)) /
                                               (0.5 * perturbationWidth))))

            temperature = temperature.expand_dims(dim='Time', axis=0)

            normalVelocity = xarray.zeros_like(ds.xEdge)
            normalVelocity, _ = xarray.broadcast(normalVelocity, ds.refBottomDepth)
            normalVelocity = normalVelocity.transpose('nEdges', 'nVertLevels')
            normalVelocity = normalVelocity.expand_dims(dim='Time', axis=0)

            ds['temperature'] = temperature
            ds['salinity'] = salinity * xarray.ones_like(temperature)
            ds['normalVelocity'] = normalVelocity
            ds['fCell'] = coriolis_parameter * xarray.ones_like(xCell)
            ds['fEdge'] = coriolis_parameter * xarray.ones_like(ds.xEdge)
            ds['fVertex'] = coriolis_parameter * xarray.ones_like(ds.xVertex)

            write_netcdf(ds, 'ocean.nc')

The details aren't critical for the purpose of this tutorial, though you may
find this example to be useful for developing other test cases, particularly
those for the ``ocean`` MPAS core.  The point is mostly to show how config
options are used to define the initial condition. Again, we use config options
from ``baroclinic_channel.cfg``, this time in a section specific to the test
group that we therefore call ``baroclinic_channel``:

.. code-block:: cfg

    ...
    # config options for baroclinic channel testcases
    [baroclinic_channel]

    # Logical flag that determines if locations of features are defined by distance
    # or fractions. False means fractions.
    use_distances = False

    # Temperature of the surface in the northern half of the domain.
    surface_temperature = 13.1

    # Temperature of the bottom in the northern half of the domain.
    bottom_temperature = 10.1

    # Difference in the temperature field between the northern and southern halves
    # of the domain.
    temperature_difference = 1.2

    # Fraction of domain in Y direction the temperature gradient should be linear
    # over.
    gradient_width_frac = 0.08

    # Width of the temperature gradient around the center sin wave. Default value
    # is relative to a 500km domain in Y.
    gradient_width_dist = 40e3

    # Salinity of the water in the entire domain.
    salinity = 35.0

    # Coriolis parameter for entire domain.
    coriolis_parameter = -1.2e-4

Again, the idea is that we make these config options rather than hard-coding
them in the test case so that users can more easily alter the test case and
also to provide a relatively obvious place to document these parameters.

Adding step outputs
~~~~~~~~~~~~~~~~~~~

Now that we've written the full ``run()`` method for the step, we know what
the output files will be.  It is a very good idea to define the outputs
explicitly.  For one, compass will check to make sure they are created as
expected and raise an error if not.  For another, we anticipate that defining
outputs will be a requirement for future work on task parallelism in which
the connection between test cases and steps will be determined based on their
inputs and outputs.  For this step, we add the following outputs in the
constructor:

.. code-block:: python

    class InitialState(Step):
        ...
        def __init__(self, test_case, resolution):
            ...
            for file in ['base_mesh.nc', 'culled_mesh.nc', 'culled_graph.info',
                         'ocean.nc']:
                self.add_output_file(file)

Only ``ocean.nc`` and ``culled_graph.info`` are strictly necessary, as these
are used as inputs to the ``forward`` step that we will define below, but
explicitly including other outputs is not a problem.

Adding the forward step
-----------------------

Now, we will add a ``forward`` step for running the MPAS-Ocean model forward
in time from the initial condition created in ``initial_state``.  ``forward``
is conceptually similar to ``initial_state`` in that we make a ``Forward``
class that descends from ``Step`` with a constructor and that calls the base
constructor with the name of the step.  This time, we also supply the target
number of cores, minimum number of cores, and number of threads (the
``initial_state`` always used the default of 1 core and 1 thread):

.. code-block:: python

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
                     ntasks=1, min_tasks=None, openmp_threads=1, nu=None):
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

            ntasks : int, optional
                the number of tasks the step would ideally use.  If fewer tasks
                are available on the system, the step will run on all available
                tasks as long as this is not below ``min_tasks``

            min_tasks : int, optional
                the number of tasks the step requires.  If the system has fewer
                than this number of tasks, the step will fail

            openmp_threads : int, optional
                the number of OpenMP threads the step will use

            nu : float, optional
                the viscosity (if different from the default for the test group)
            """
            self.resolution = resolution
            if min_tasks is None:
                min_tasks = ntasks
            super().__init__(test_case=test_case, name=name, subdir=subdir,
                             ntasks=ntasks, min_tasks=min_tasks,
                             openmp_threads=openmp_threads)


The default number of MPI tasks and threads is 1, and the default minimum
number of MPI tasks (``min_tasks``) is the same as the number of tasks (so
also 1 if ``ntasks`` isn't specified).  See :ref:`dev_steps` for more details.
There is also a parameter ``nu``, the viscosity, which will be set depending on
the test case.

Next, we add inputs that are outputs from the ``initial_state`` test case:
.. code-block:: python

            self.add_input_file(filename='init.nc',
                                target='../initial_state/ocean.nc')
            self.add_input_file(filename='graph.info',
                                target='../initial_state/culled_graph.info')

We also add a link to the MPAS-Ocean executable as an input:

.. code-block:: python

        self.add_model_as_input()

Defining namelist options
~~~~~~~~~~~~~~~~~~~~~~~~~

MPAS components require both namelist and streams files to work properly.  An
important part of compass' functionality is that it takes the default namelist
options from a given build of an MPAS component and modifies only those
options that are specific to the test case to produce the final namelist file
used to run the model.

In ``compass``, there are two main ways to set namelist options for MPAS model
runs and we will demonstrate both in this test case.  First, you can define a
namelist file with the desired values.  This is useful for namelist options
that are always the same for this test case and can't be changed based on
config options from the config file (see above).

In ``compass`` the formatting for a namelist file within a test group or test
case's python package similar to the resulting namelist file.  Here is the
``namelist.forward`` file from the ``baroclinic_channel`` test group:

.. code-block:: none

    config_write_output_on_startup = .false.
    config_run_duration = '0000_00:15:00'
    config_use_mom_del2 = .true.
    config_implicit_bottom_drag_coeff = 1.0e-2
    config_use_cvmix_background = .true.
    config_cvmix_background_diffusion = 0.0
    config_cvmix_background_viscosity = 1.0e-4

Some namelist options are specific to a given resolution, so it is handy to
define namelist files for each resolution.  As an example, here is
``namelist.10km.forward``:

.. code-block:: none

    config_dt = '00:05:00'
    config_btr_dt = '00:00:15'
    config_mom_del2 = 10.0

In the ``forward`` step, we add these namelists as follows:

.. code-block:: python

    ...
    class Forward(Step):
        ...
    def __init__(self, test_case, resolution, name='forward', subdir=None,
                 ntasks=1, min_tasks=None, openmp_threads=1, nu=None):
            ...

            self.add_namelist_file('compass.ocean.tests.baroclinic_channel',
                                   'namelist.forward')
            self.add_namelist_file('compass.ocean.tests.baroclinic_channel',
                                   'namelist.{}.forward'.format(resolution))

The first argument to :py:meth:`compass.Step.add_namelist_file()` is the
python package where the namelist file can be found, and the second is the
file name.  Files within the ``compass`` package can't be referenced directly
with a file path but rather with a package like in these examples.

Another way to set namelist options is to use a python dictionary and to call
:py:meth:`compass.Step.add_namelist_options()`.  This is the way to handle
namelist options that depend on parameters (such as resolution) that are not
known in advance.  In this case, we use this techinique to set the namelist
option for the viscosity ``config_mom_del2`` using the parameter ``nu`` passed
into the constructor (if it is not ``None``, indicating that it was not set).

.. code-block:: python

    ...
    class Forward(Step):
        ...
    def __init__(self, test_case, resolution, name='forward', subdir=None,
                 ntasks=1, min_tasks=None, openmp_threads=1, nu=None):
            ...

            if nu is not None:
                # update the viscosity to the requested value
                options = {'config_mom_del2': '{}'.format(nu)}
                self.add_namelist_options(options)

Defining streams
~~~~~~~~~~~~~~~~

Similarly, it is convenient to define input and output streams for MPAS-Ocean
using a streams file, very similar to what you will see when the test case
is set up. In the ``baroclinic_channel`` test group, we add a
``streams.forward`` file that looks like this:

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

Streams that are already defined like ``mesh``, ``input`` and ``restart``
will use the default attributes defined by the MPAS component unless they are
explicitly replaced in the streams file.  As an example, on setting up the
step, the stream ``mesh`` in the ``streams.ocean`` file becomes:

.. code-block:: xml

    <immutable_stream name="mesh"
                      type="input"
                      filename_template="init.nc"
                      input_interval="initial_only"/>

In the ``forward`` step, we add these streams file as follows:

.. code-block:: python

    ...
    class Forward(Step):
        ...
    def __init__(self, test_case, resolution, name='forward', subdir=None,
                 ntasks=1, min_tasks=None, openmp_threads=1, nu=None):
            ...

            self.add_streams_file('compass.ocean.tests.baroclinic_channel',
                                  'streams.forward')

Similarly to namelists, the first argument to
:py:meth:`compass.Step.add_streams_file()` is the python package where the
streams file can be found, and the second is the file name.

Defining the run method
~~~~~~~~~~~~~~~~~~~~~~~

With these inputs, outputs, namelists and streams files defined, we can
implement the ``run()`` method:

.. code-block:: python

    from compass.model import run_model
    from compass.step import Step


    class Forward(Step):
    ...

        def run(self):
            """
            Run this step of the test case
            """
            run_model(self)

We simply run MPAS-Ocean by calling :py:func:`compass.model.run_model()`.
We pass the step itself as an argument because this is how ``compass`` knows
how many cores and threads to run on, which namelist and streams files to use,
which MPAS core this test case belongs to, and so on.

Adding the steps to the test case
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Returning to the ``default`` test case, we are now ready to add
``initial_state`` and ``forward`` steps to the test case.  In
``compass/ocean/tests/baroclinic_channel/default/__init.py``, we add:

.. code-block:: python
    :emphasize-lines: 2-3, 37-40

    from compass.testcase import TestCase
    from compass.ocean.tests.baroclinic_channel.initial_state import InitialState
    from compass.ocean.tests.baroclinic_channel.forward import Forward
    from compass.ocean.tests import baroclinic_channel


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

We hard-code the ``forward`` test case to run on 4 cores and 1 thread, and do
not pass a viscosity (meaning it will use the default value from
``namelist.<resolution>.forward``).

Adding an "rpe_test" test case
------------------------------

The ``baroclinic_channel`` test group contains several test cases in addition
to ``default``.  The ``restart_test`` checks whether running the model for one
times step, writing out a restart file, loading the model state from the
restart file, and running for another time step produces the same results as
running for 2 time steps.  The ``decomp_test`` and ``threads_test`` check
whether the results are the same when the model runs on different numbers of
cores and threads, respectively.

The most interesting test case is the ``rpe_test``, which has been used to show
that MPAS-Ocean has lower spurious dissipation of reference potential energy
(RPE) than POP, MOM and MITgcm models
(`Petersen et al. 2015 <https://doi.org/10.1016/j.ocemod.2014.12.004>`_).

The ``rpe_test`` test case can be run at any of the supported resolutions: 1,
4 or 10 km.  It consists of an ``initial_state`` step exactly like the
``default`` test case, 5 variants of the ``forward`` step with different values
of the viscosity, and an ``analysis`` step that is unique to this test case
(and thus not part of the "framework" for the test group over all like the
``initial_state`` and ``forward`` steps).  Each ``forward`` step runs for
much longer than in the ``default`` test case (20 days, rather than 15
minutes).  This means that ``rpe_test`` isn't appropriate for regression
testing, since it is too time consuming to run.  Likewise, the higher
resolutions (1 and 4 km) are fairly resource heavy, and therefore not as well
suit to quick testing.  But this test case was the original purpose of the test
group as a whole, serving to validate the code in a specific context.

In analogy to the ``default`` test case, we will start by creating a directory
``rpe_test`` within the ``baroclinic_channel`` directory, adding a new file
``__init__.py``, and adding a class ``RpeTest`` that descends from the
``TestCase`` base class:

.. code-block:: python

    from compass.testcase import TestCase


    class RpeTest(TestCase):
        """
        The reference potential energy (RPE) test case for the baroclinic channel
        test group performs a 20-day integration of the model forward in time at
        5 different values of the viscosity at the given resolution.

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
            name = 'rpe_test'
            subdir = '{}/{}'.format(resolution, name)
            super().__init__(test_group=test_group, name=name,
                             subdir=subdir)
            self.resolution = resolution

So far, this is identical ot the ``default`` test case except for the name
changes.

Before we add steps, let's add the ``rpe_test`` test case to the
``baroclinic_channel`` test group so we can compare it with the ``default``
tet case. We add the following to the file ``__init__.py`` that defines the
``BaroclinicChannel`` test group:

.. code-block:: python
    :emphasize-lines: 3, 17-19

    from compass.testgroup import TestGroup
    from compass.ocean.tests.baroclinic_channel.default import Default
    from compass.ocean.tests.baroclinic_channel.rpe_test import RpeTest


    class BaroclinicChannel(TestGroup):
        """
        A test group for baroclinic channel test cases
        """
        def __init__(self, mpas_core):
            """
            mpas_core : compass.MpasCore
                the MPAS core that this test group belongs to
            """
            super().__init__(mpas_core=mpas_core, name='baroclinic_channel')

            for resolution in ['1km', '4km', '10km']:
                self.add_test_case(
                    RpeTest(test_group=self, resolution=resolution))
            for resolution in ['10km']:
                self.add_test_case(
                    Default(test_group=self, resolution=resolution))

The ``rpe_test`` test case, unlike all the other test cases in this group, can
be run at all three supported resolutions.

Adding the steps to the test case
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We are now ready to add the ``initial_state`` step and variants of the
``forward`` step to the test case.  In
``compass/ocean/tests/baroclinic_channel/rpe_test/__init.py``, we add:

.. code-block:: python
    :emphasize-lines: 2-3, 35-46, 50-66

    from compass.testcase import TestCase
    from compass.ocean.tests.baroclinic_channel.initial_state import InitialState
    from compass.ocean.tests.baroclinic_channel.forward import Forward


    class RpeTest(TestCase):
        """
        The reference potential energy (RPE) test case for the baroclinic channel
        test group performs a 20-day integration of the model forward in time at
        5 different values of the viscosity at the given resolution.

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
            name = 'rpe_test'
            subdir = f'{resolution}/{name}'
            super().__init__(test_group=test_group, name=name,
                             subdir=subdir)

            nus = [1, 5, 10, 20, 200]

            res_params = {'1km': {'ntasks': 144, 'min_tasks': 36},
                          '4km': {'ntasks': 36, 'min_tasks': 8},
                          '10km': {'ntasks': 8, 'min_tasks': 4}}

            if resolution not in res_params:
                raise ValueError(
                    f'Unsupported resolution {resolution}. Supported values are: '
                    f'{list(res_params)}')

            params = res_params[resolution]

            self.resolution = resolution

            self.add_step(
                InitialState(test_case=self, resolution=resolution))

            for index, nu in enumerate(nus):
                name = 'rpe_test_{}_nu_{}'.format(index + 1, nu)
                step = Forward(
                    test_case=self, name=name, subdir=name,
                    ntasks=params['ntasks'], min_tasks=params['min_tasks'],
                    resolution=resolution, nu=float(nu))

                step.add_namelist_file(
                    'compass.ocean.tests.baroclinic_channel.rpe_test',
                    'namelist.forward')
                step.add_streams_file(
                    'compass.ocean.tests.baroclinic_channel.rpe_test',
                    'streams.forward')
                self.add_step(step)

            self.add_step(
                Analysis(test_case=self, resolution=resolution, nus=nus))

Here, we use nested python dictionaries ``res_params`` to determine the target
number of cores and the minimum allowed cores for each resolution of the test
case.  (We also raise an error if an unexpected resolution is provided, just
in case.)

The list ``nus`` contains the viscosities for each forward step in the test
case.  We create a different forward run with a different name for each
viscosity, passing ``nu`` to the ``Forward`` step's constructor so it will
be used to set the appropriate config option.  Alternatively, given that this
test case is the only one to use the ``nu`` parameter, we could have left the
``nu`` parameter out of ``Forward`` and set it here instead, as follows:

.. code-block:: python

            ...

            for index, nu in enumerate(nus):
                name = 'rpe_test_{}_nu_{}'.format(index + 1, nu)
                step = Forward(
                    test_case=self, name=name, subdir=name,
                    ntasks=params['ntasks'], min_tasks=params['min_tasks'],
                    resolution=resolution)
                options = {'config_mom_del2': f'{nu}'}
                step.add_namelist_options(options)

                ...
                self.add_step(step)

Defining namelist options and streams files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``rpe_test`` requires a few specific namelist options and streams to
accommodate the longer run and to modify the variables that are written out.
We add these namelist options within ``namelist.forward`` in the test case's
directory:

.. code-block:: none

    config_run_duration = '20_00:00:00'

and the following stream in ``streams.forward``:

.. code-block:: xml

    <streams>

    <stream name="output"
            type="output"
            filename_template="output.nc"
            output_interval="0000-00-20_00:00:00"
            clobber_mode="truncate">

        <var_struct name="tracers"/>
        <var name="xtime"/>
        <var name="density"/>
        <var name="daysSinceStartOfSim"/>
        <var name="relativeVorticity"/>
    </stream>

    </streams>

This makes sure that each MPAS-Ocean simulation runs for 20 model days, writing
output only at the end of the simulation, and including the ``density`` and
``relativeVorticity`` fields, rather than ``normalVelocity`` and
``layerThickness``, as in the defaults.  These fields are needed in the
analysis step.

Adding the analysis step
------------------------

The ``rpe_test`` includes another step, ``analysis`` that plots results from
each simulation.  The full analysis step looks like this:

.. code-block:: python

    import numpy as np
    from netCDF4 import Dataset
    import matplotlib.pyplot as plt
    import cmocean

    from compass.step import Step


    class Analysis(Step):
        """
        A step for plotting the results of a series of RPE runs in the baroclinic
        channel test group

        Attributes
        ----------
        resolution : str
            The resolution of the test case

        nus : list of float
            A list of viscosities
        """
        def __init__(self, test_case, resolution, nus):
            """
            Create the step

            Parameters
            ----------
            test_case : compass.TestCase
                The test case this step belongs to

            resolution : str
                The resolution of the test case

            nus : list of float
                A list of viscosities
            """
            super().__init__(test_case=test_case, name='analysis')
            self.resolution = resolution
            self.nus = nus

            for index, nu in enumerate(nus):
                self.add_input_file(
                    filename='output_{}.nc'.format(index+1),
                    target='../rpe_test_{}_nu_{}/output.nc'.format(index+1, nu))

            self.add_output_file(
                filename='sections_baroclinic_channel_{}.png'.format(resolution))

        def run(self):
            """
            Run this step of the test case
            """
            section = self.config['baroclinic_channel']
            nx = section.getint('nx')
            ny = section.getint('ny')
            _plot(nx, ny, self.outputs[0], self.nus)


    def _plot(nx, ny, filename, nus):
        """
        Plot section of the baroclinic channel at different viscosities

        Parameters
        ----------
        nx : int
            The number of cells in the x direction

        ny : int
            The number of cells in the y direction (before culling)

        filename : str
            The output file name

        nus : list of float
            The viscosity values
        """

        ...

where the details of the ``_plot()`` function have been left out for
compactness.  ``analysis`` needs the results from each forward step's
``output.nc`` file as inputs, and plots the results together in a single image
that it writes out.

We add the ``analysis`` step to the test case as follows:

.. code-block:: python
    :emphasize-lines: 4, 41-42

    from compass.testcase import TestCase
    from compass.ocean.tests.baroclinic_channel.initial_state import InitialState
    from compass.ocean.tests.baroclinic_channel.forward import Forward
    from compass.ocean.tests.baroclinic_channel.rpe_test.analysis import Analysis
    from compass.ocean.tests import baroclinic_channel


    class RpeTest(TestCase):
        """
        The reference potential energy (RPE) test case for the baroclinic channel
        test group performs a 20-day integration of the model forward in time at
        5 different values of the viscosity at the given resolution.

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
            name = 'rpe_test'
            subdir = '{}/{}'.format(resolution, name)
            super().__init__(test_group=test_group, name=name,
                             subdir=subdir)

            nus = [1, 5, 10, 20, 200]

            ...

            self.add_step(
                Analysis(test_case=self, resolution=resolution, nus=nus))

Setting config options based on resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It turns out that we need a ``configure()`` method that is identical to that in
the ``Default`` test case.  We could copy the code but we have a strong
preference for code reuse when possible in ``compass``.  For this reason, it
makes sense to make a function in the ``baroclinic_channel`` framework that
each test case can use to do the same configuration.  In this example, we move
the ``configure`` method from ``Default`` into
``baroclinic_channel/__init__.py``, but you could choose to put it in a new
module called ``configure.py`` if you prefer.

.. code-block:: python

    ...

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

Since ``configure()`` is no longer a method of a class descending from
``TestCase``, it cannot have an argument ``self`` anymore.  Instead, the new
function must take the attributes from the test case that it needs:
``resolution`` and ``config``.  From there, the behavior is the same as before.

Now, each test case will just call this ``configure()`` function inside its own
``configure()`` method.  The following code applies to both the ``Default`` and
``RpeTest`` test cases:

.. code-block:: python

    ...
    from compass.ocean.tests import baroclinic_channel

    ...

        def configure(self):
            """
            Modify the configuration options for this test case.
            """
            baroclinic_channel.configure(self.resolution, self.config)

We import the ``baroclinic_channel`` module instead of the ``configure()``
function because otherwise there would be confusion between the ``configure()``
function and the ``configure()`` method.  An alternative would be to import
the function but give it a new name:

.. code-block:: python

    ...
    from compass.ocean.tests.baroclinic_channel import configure as bc_configure

    ...

        def configure(self):
            """
            Modify the configuration options for this test case.
            """
            bc_configure(self.resolution, self.config)

Set up and run
--------------

You're all set!  You should be able to see your new test cases when you run
``compass list``, set them up by running ``compass setup``, and run them by
calling ``compass run`` within the work directory.  See :ref:`dev_command_line`
for details.

.. _dev_tutorial_add_test_group_docs:

Documentation
-------------

Make sure to add some documentation of your new test group.  You need to add
all of the functions, classes and methods to the API documentation in
``docs/developers_guide/<core>/api.rst``, following the examples for other
test groups.  You also need to add a file to both the user's guide and the
developer's guide describing the test group and its test cases and steps.

For the user's guide, create a file
``docs/users_guide/<core>/test_groups/<test_group>.rst``.  In that file, you
should describe the test group and its test cases in a way that would be
relevant for a user wanting to run the test case and look at the output.
This file should include a section giving the config options for the test
group and each test case (if it has its own config options), describing what
they are used for so that users know how to modify them if they want to.  Add
``<test_group>`` in the appropriate place (in alphabetical order) to the list
of test groups in the file ``docs/users_guide/<core>/test_groups/index.rst``.

For the developer's guide, create a file
``docs/developers_guide/<core>/test_groups/<test_group>.rst``. In this file,
you will describe the test group, its test cases and steps in a way that is
relevant to developers who might want to modify the code or use it as an
example for developing their own test cases.  Currently, the descriptions are
brief in part because of the daunting task of documenting nearly 100 test cases
but should be fleshed out over time.  It would help new developers if new test
groups and test cases were documented well. Add ``<test_group>`` in the
appropriate place (in alphabetical order) to the list of test groups in
``docs/developers_guide/<core>/test_groups/index.rst``.

At this point, you are ready to make a pull request with the new test group!
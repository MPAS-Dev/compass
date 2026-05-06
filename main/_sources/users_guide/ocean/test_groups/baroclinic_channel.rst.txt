.. _ocean_baroclinic_channel:

baroclinic_channel
==================

The ``ocean/baroclinic_channel`` test group implements variants of the
Baroclinic Eddies test case from
`Ilicak et al. (2012) <https://doi.org/10.1016/j.ocemod.2011.10.003>`_.

The domain is zonally periodic with solid northern and southern boundaries.
Salinity is constant throughout the domain (at 35 PSU).  The initial
temperature is cooler in the southern half of the domain than in the north,
with a gradient between the two halves that is sinusoidally perturbed in the
meridional direction.  The surface temperature is also warmer than at depth.

.. image:: images/baroclinic_channel.png
   :width: 500 px
   :align: center

Variants of the test case are available at 1-km, 4-km and 10-km horizontal
resolution.  By default, the 20 vertical layers each have 50-m uniform
thickness.

The test group includes 5 test cases.  All test cases have 2 steps,
``initial_state``, which defines the mesh and initial conditions for the model,
and ``forward`` (given another name in many test cases to distinguish multiple
forward runs), which performs time integration of the model.

config options
--------------

All 5 test cases share the same set of config options:

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


    # namelist options for baroclinic channel testcases
    [baroclinic_channel]

    # Logical flag that determines if locations of features are defined by distance
    # or fractions. False means fractions.
    use_distances = False

    # Viscosity values to test for rpe test case
    viscosities = 1, 5, 10, 20, 200

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

All units are mks, with temperature in degrees Celsius and salinity in PSU.

default
-------

``ocean/baroclinic_channel/10km/default`` is the default version of the
baroclinic eddies test case for a short (15 min) test run and validation of
prognostic variables for regression testing.  Currently, only 10-km horizontal
resolution is supported.

decomp_test
-----------

``ocean/baroclinic_channel/10km/decomp_test`` runs a short (15 min) integration
of the model forward in time on 4 (``4proc`` step) and then on 8 processors
(``8proc`` step) to make sure the resulting prognostic variables are
bit-for-bit identical between the two runs. Currently, only 10-km horizontal
resolution is supported.

thread_test
-----------

``ocean/baroclinic_channel/10km/thread_test`` runs a short (15 min) integration
of the model forward in time on 1 threads per processor (``1thread`` step) and
then on 2 threads (``2thread`` step) to make sure the resulting prognostic
variables are bit-for-bit identical between the two runs. Currently, only 10-km
horizontal resolution is supported.

restart_test
------------

``ocean/baroclinic_channel/10km/restart_test`` runs a short (10 min)
integration of the model forward in time (``full_run`` step), saving a restart
file every 5 minutes.  Then, a second run (``restart_run`` step) is performed
from the restart file 5 minutes into the simulation and prognostic variables
are compared between the "full" and "restart" runs at minute 10 to make sure
they are bit-for-bit identical. Currently, only 10-km horizontal resolution is
supported.

rpe_test
--------

``ocean/baroclinic_channel/1km/rpe_test``,
``ocean/baroclinic_channel/4km/rpe_test``, and
``ocean/baroclinic_channel/10km/rpe_test`` perform longer (20 day) integration
of the model forward in time at 5 different values of the viscosity (with steps
named ``rpe_test_1_nu_1``, ``rpe_test_2_nu_5``, etc.) at any of the 3 supported
horizontal resolutions (1, 4 and 10 km).  Results of these tests have been used
to show that MPAS-Ocean has lower spurious dissipation of reference potential
energy (RPE) than POP, MOM and MITgcm models
(`Petersen et al. 2015 <https://doi.org/10.1016/j.ocemod.2014.12.004>`_).

.. _ocean_turbulence_closure:

turbulence_closure
==================

The ``turbulence_closure`` test group implements studies of the surface layer
evolution under different kinds of surface forcing for different turbulence
closures.

The domain is periodic in the horizontal dimensions and no-slip at both
vertical boundaries.

Initial temperature is 20 degC and salinity is 35 PSU. A linear gradient can be
given over the mixed layer or the interior. Furthermore, a polynomial function
can be used to describe the interior temperature and salinity. The default
parameters are based on a fit to a time average over the whole 1 year
deployment of ITP #118, located in the Beaufort Gyre (western Arctic).

Variants of the test case are available at 1-m, 2-m and 10-km horizontal
resolution. The O(1)m resolution test cases are designed to be run in LES mode
whereas the O(1)km test cases are designed to be used to test turbulence
closures in "standard" MPAS-Ocean.

The vertical resolution for the LES test cases is the same as the horizontal
resolution (1m or 2m) whereas the standard resolution test cases have a
vertical resolution of 50m.

There are currently two surface forcing variants available, cooling and
evaporation.

All test cases have 2 steps,
``initial_state``, which defines the mesh and initial conditions for the model,
and ``forward``, which performs time integration of the model. The ``default``
test also has a viz step which performs visualization.

 
config options
--------------

namelist options
----------------
When the horizontal resolution is less than 500m, LES mode is active with the
following config options:

.. code-block:: cfg

    config_LES_mode = .true.
    config_use_two_equation_turbulence_model = .true.

    # TODO Explain what 'omega' means
    config_two_equation_model_choice = 'omega'
    config_use_cvmix = .false.

    config_two_equation_length_min = 0.1
    config_two_equation_tke_min = 1e-10
    config_two_equation_psi_min = 1e-10
    config_tke_ini = 1.0e-30

    config_LES_noise_enable = .true.
    config_LES_noise_max_time = 150
    config_LES_noise_min_level = 5
    config_LES_noise_max_level = 45

    # TODO Can you provide any guidance here about what appropriate choices
    # might be?
    config_LES_noise_perturbation_magnitude = 1e-4

When the horizontal resolution is less than 100m, nonhydrostatic mode is
active with the following config options:

.. code-block:: cfg

    # turns the nonhydro model on or off
    config_enable_nonhydrostatic_mode = .true.  

    # preconditioner for the linear solver. Other options can be used, like
    # jacobi, sor and asm, but they were found to be slower than block jacobi.
    config_nonhydrostatic_preconditioner = 'bjacobi'

    # linear solver. Other options can be used, like gmres
    config_nonhydrostatic_solver_type = 'cg'

    # do not change!
    config_nonhydrostatic_solve_surface_boundary_condition = 'pressureTopGradientBottom'

    # Nonhydro will work with either 'centered' or 'upwind'.
    config_thickness_flux_type = 'upwind'

default
-------

The default version of the turbulence closure test case is available for
multiple resolutions: ``ocean/baroclinic_channel/1m/default``,
``ocean/baroclinic_channel/2m/default``, and
``ocean/baroclinic_channel/10km/default``. The default simulation time is 4 days.

decomp_test
-----------

``ocean/turbulence_closure/10km/decomp_test`` runs a short (15 min) integration
of the model forward in time on 4 (``4proc`` step) and then on 8 processors
(``8proc`` step) to make sure the resulting prognostic variables are
bit-for-bit identical between the two runs. Currently, only 10-km horizontal
resolution is supported.

restart_test
------------

``ocean/turbulence_closure/10km/restart_test`` runs a short (10 min)
integration of the model forward in time (``full_run`` step), saving a restart
file every 5 minutes.  Then, a second run (``restart_run`` step) is performed
from the restart file 5 minutes into the simulation and prognostic variables
are compared between the "full" and "restart" runs at minute 10 to make sure
they are bit-for-bit identical. Currently, only 10-km horizontal resolution is
supported.

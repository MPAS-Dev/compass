.. _dev_ocean_sphere_transport:

sphere_transport
================

The ``sphere_transport`` test group implements 4 test cases for standard
transport schemes.   See :ref:`ocean_sphere_transport` in the User's Guide for
more details.

.. _dev_ocean_sphere_transport_rotation_2d:

rotation_2d
-----------

The :py:class:`compass.ocean.tests.sphere_transport.rotation_2d.Rotation2D`
test performs a series of 12-day runs that advect 3 tracers around the sphere
with solid-body rotation. The resolution of the sphere varies (by default,
between 60 and 240 km).  Advected results are compared with a known exact
solution to determine the rate of convergence.  See
:ref:`ocean_sphere_transport_rotation_2d` for config options and
more details on the test case.

mesh
~~~~

The class :py:class:`compass.ocean.tests.sphere_transport.rotation_2d.mesh.Mesh`
defines a step for building a global mesh with uniform resolution using
:py:func:`mpas_tools.ocean.build_mesh.build_spherical_mesh()`.

init
~~~~

The class :py:class:`compass.ocean.tests.sphere_transport.rotation_2d.init.Init`
defines a step for setting up the initial state for each test case with a
tracer distributed in a cosine-bell shape.

forward
~~~~~~~

The class :py:class:`compass.ocean.tests.sphere_transport.rotation_2d.forward.Forward`
defines a step for running MPAS-Ocean from the initial condition produced in
the ``initial_state`` step.  The time step for a given resolution is given by
the ``timestep_munutes`` config option.  Other namelist options are taken from
the test case's ``namelist.forward``.

analysis
~~~~~~~~

The class :py:class:`compass.ocean.tests.sphere_transport.rotation_2d.analysis.Analysis`
defines a step for plotting various results of the simulations in
``rotation_2d_convergence.pdf`` and at each resolution in ``*_sol.pdf``.

.. _dev_ocean_sphere_transport_nondivergent_2d:

nondivergent_2d
---------------

The :py:class:`compass.ocean.tests.sphere_transport.nondivergent_2d.Nondivergent2D`
test performs a series of 12-day runs that advect 3 tracers around the sphere
within a divergence-free flow. The resolution of the sphere varies (by default,
between 60 and 240 km).  Advected results are compared with a known exact
solution to determine the rate of convergence.  See
:ref:`ocean_sphere_transport_nondivergent_2d` for config options and
more details on the test case.

mesh
~~~~

The class :py:class:`compass.ocean.tests.sphere_transport.nondivergent_2d.mesh.Mesh`
defines a step for building a global mesh with uniform resolution using
:py:func:`mpas_tools.ocean.build_mesh.build_spherical_mesh()`.

init
~~~~

The class :py:class:`compass.ocean.tests.sphere_transport.nondivergent_2d.init.Init`
defines a step for setting up the initial state for each test case with a
tracer distributed in a cosine-bell shape.

forward
~~~~~~~

The class :py:class:`compass.ocean.tests.sphere_transport.nondivergent_2d.forward.Forward`
defines a step for running MPAS-Ocean from the initial condition produced in
the ``initial_state`` step.  The time step for a given resolution is given by
the ``timestep_munutes`` config option.  Other namelist options are taken from
the test case's ``namelist.forward``.

analysis
~~~~~~~~

The class :py:class:`compass.ocean.tests.sphere_transport.nondivergent_2d.analysis.Analysis`
defines a step for plotting various results of the simulations in
``nondivergent2D_convergence.pdf`` and at each resolution in ``*_sol.pdf``.

.. _dev_ocean_sphere_transport_divergent_2d:

divergent_2d
------------

The :py:class:`compass.ocean.tests.sphere_transport.divergent_2d.Divergent2D`
test performs a series of 12-day runs that advect 3 tracers around the sphere
within a flow with nonzero divergence. The resolution of the sphere varies (by
default, between 60 and 240 km).  Advected results are compared with a known
exact solution to determine the rate of convergence.  See
:ref:`ocean_sphere_transport_divergent_2d` for config options and
more details on the test case.

mesh
~~~~

The class :py:class:`compass.ocean.tests.sphere_transport.divergent_2d.mesh.Mesh`
defines a step for building a global mesh with uniform resolution using
:py:func:`mpas_tools.ocean.build_mesh.build_spherical_mesh()`.

init
~~~~

The class :py:class:`compass.ocean.tests.sphere_transport.divergent_2d.init.Init`
defines a step for setting up the initial state for each test case with a
tracer distributed in a cosine-bell shape.

forward
~~~~~~~

The class :py:class:`compass.ocean.tests.sphere_transport.divergent_2d.forward.Forward`
defines a step for running MPAS-Ocean from the initial condition produced in
the ``initial_state`` step.  The time step for a given resolution is given by
the ``timestep_munutes`` config option.  Other namelist options are taken from
the test case's ``namelist.forward``.

analysis
~~~~~~~~

The class :py:class:`compass.ocean.tests.sphere_transport.divergent_2d.analysis.Analysis`
defines a step for plotting various results of the simulations in
``divergent2D_convergence.pdf`` and at each resolution in ``*_sol.pdf``.

.. _dev_ocean_sphere_transport_correlated_tracers_2d:

correlated_tracers_2d
---------------------

The :py:class:`compass.ocean.tests.sphere_transport.correlated_tracers_2d.CorrelatedTracers2D`
test performs a series of 12-day runs that advect 3 tracers around the sphere
with the same flow field as :ref:`dev_ocean_sphere_transport_nondivergent_2d`.
The resolution of the sphere varies (by default, between 60 and 240 km).
Advected results are compared with a known exact solution to determine the rate
of convergence.  See :ref:`ocean_sphere_transport_correlated_tracers_2d` for
config options and more details on the test case.

mesh
~~~~

The class :py:class:`compass.ocean.tests.sphere_transport.correlated_tracers_2d.mesh.Mesh`
defines a step for building a global mesh with uniform resolution using
:py:func:`mpas_tools.ocean.build_mesh.build_spherical_mesh()`.

init
~~~~

The class :py:class:`compass.ocean.tests.sphere_transport.correlated_tracers_2d.init.Init`
defines a step for setting up the initial state for each test case with a
tracer distributed in a cosine-bell shape.

forward
~~~~~~~

The class :py:class:`compass.ocean.tests.sphere_transport.correlated_tracers_2d.forward.Forward`
defines a step for running MPAS-Ocean from the initial condition produced in
the ``initial_state`` step.  The time step for a given resolution is given by
the ``timestep_munutes`` config option.  Other namelist options are taken from
the test case's ``namelist.forward``.

analysis
~~~~~~~~

The class :py:class:`compass.ocean.tests.sphere_transport.correlated_tracers_2d.analysis.Analysis`
defines a step for plotting various results of the simulations in
``correlatedTracers2D_triplots.pdf`` and at each resolution in ``*_sol.pdf``.
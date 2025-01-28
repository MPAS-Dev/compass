.. _dev_ocean_spherical_harmonic_transform:

spherical_harmonic_transform
============================

The ``spherical_harmonic_transform`` test group implements 
See :ref:`ocean_spherical_harmonic_transform` in the User's Guide for
more details.

.. _dev_ocean_spherical_harmonic_transform_qu_convergence:

qu_convergence
--------------

The :py:class:`compass.ocean.tests.spherical_harmonic_transform.qu_convergence.QuConvergence`
Performs a series of spherical harmonic transformations on an 
analytical function for various orders. The truncation error 
of each approximation order is compared with the original function.
The resolution of the sphere also varies (by default, 60 and 30 km).
Both the "serial" and "parallel" implemntations are used.
See :ref:`ocean_spherical_harmonic_transform` for config options and
more details on the test case.

mesh
~~~~

The class :py:class:`compass.ocean.tests.spherical_harmonic_transform.qu_convergence.mesh`
defines a step for building a global mesh with uniform resolution using
:py:func:`mpas_tools.ocean.build_mesh.build_spherical_mesh()`.

init
~~~~

The class :py:class:`compass.ocean.tests.spherical_harmonic_transform.qu_convergence.init`
defines a step for runnining MPAS-Ocean for the ``test_sht``
configuration to perform the spherical harmonic transform.

analysis
~~~~~~~~

The class :py:class:`compass.ocean.tests.spherical_harmonic_transform.qu_convergence.analysis`
defines a step for plotting the truncation error convergence for the
orders and resolutions run. The file produced is ``convergence.png``.


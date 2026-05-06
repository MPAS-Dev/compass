.. _ocean_gotm:

gotm
====

The ``gotm`` test group (:py:class:`compass.ocean.tests.gotm.Gotm`)
implements a test case for validating the General Ocean Turbulence Model
(`GOTM <https://gotm.net/portfolio/>`_) within MPAS-Ocean.
GOTM is a turbulence closure library that includes a set of two equation models
such as k-epsilon and the generic length scale model (GLS;
`Umlauf and Burchard, 2003 <https://doi.org/10.1357/002224003322005087>`_).

.. _ocean_gotm_default:

default
-------

The ``default`` test case implements implements a
single column test case following Section 5.1 of
`Kärnä, 2020 <https://doi.org/10.1016/j.ocemod.2020.101619>`_:

.. image:: images/gotm_eqns.png
   :width: 691 px
   :align: center

The simulated velocity and viscosity profiles are compared with the analytical
solution of Equations (67) and (69).

The results depends on the bottom drag coefficient, here set to 0.0173,
corresponding to the value inferred from the analytical solution with a layer
thickness of 0.06 m.

.. image:: images/gotm_vel.png
   :width: 500 px
   :align: center

.. image:: images/gotm_visc.png
   :width: 500 px
   :align: center

config options
~~~~~~~~~~~~~~

The ``default`` config options include:

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


These options control the size and resolution of the horizontal mesh and
vertical grid.


.. _ocean_vertical:

Vertical coordinate
===================

Most compass configurations and test cases support a z* vertical coordinate
(`Adcroft and Campin, 2004 <https://doi.org/10.1016/j.ocemod.2003.09.003>`_).
The number and interface locations of the initial ("resting") z* layers are
determined by config options in the ``vertical_grid`` section of the config
file:

.. code-block:: cfg

    # Options related to the vertical grid
    [vertical_grid]

    # the type of vertical grid
    grid_type = 100layerE3SMv1

    # Number of vertical levels
    vert_levels = 100

    # Depth of the bottom of the ocean
    bottom_depth = 2500.0

    # The minimum layer thickness
    min_layer_thickness = 3.0

    # The maximum layer thickness
    max_layer_thickness = 500.0

Possible grid types are: ``uniform``, ``tanh_dz``, ``60layerPHC``,
``42layerWOCE``, and ``100layerE3SMv1``.

The meaning of the other config options depends grid type.

uniform
-------

Uniform vertical grids have vertical layers all of the same thickness. The
layer thickness is simply ``bottom_depth`` (the positive depth of the bottom
interface of the deepest layer) divided by the number of layers
(``vert_levels``).  In this example, the vertical grid would have 10 vertical
layers, each 100 m thick.

.. code-block:: cfg

    # Options related to the vertical grid
    [vertical_grid]

    # the type of vertical grid
    grid_type = uniform

    # Number of vertical levels
    vert_levels = 10

    # Depth of the bottom of the ocean
    bottom_depth = 1000.0

tanh_dz
-------

This vertical coordinate is a variant on
`Stewart et al. (2017) <https://doi.org/10.1016/j.ocemod.2017.03.012>`_.  Layer
thickness is defined by:

.. math::

    \Delta z\left(z\right) = (\Delta z_2 - \Delta z_1)
               \mathrm{tanh}\left(\frac{-\pi z}{\Delta}\right) + \Delta z_1,

where :math:`\Delta z_1` is the value of the layer thickness
:math:`\Delta z` at :math:`z = 0` and :math:`\Delta z_2` is the same as
:math:`z \rightarrow \infty`.  Interface locations `z_k` are defined by:

.. math::

    z_0 = 0, \\
    z_{k+1} = z_k - \Delta z\left(z_k\right).

We use a root finder to solve for :math:`\Delta`, such that
:math:`z_{n_z+1} = -H_\mathrm{bot}`, where :math:`n_z:` is ``nVertLevels``, the
number of vertical levels (one less than the number of layer interfaces) and
:math:`H_\mathrm{bot}` is ``bottom_depth``, the depth of the seafloor.

The following config options are all required.  This is an example of a
64-layer vertical grid that has been explored in E3SM v2:

.. code-block:: cfg

    # Options related to the vertical grid
    [vertical_grid]

    # the type of vertical grid
    grid_type = tanh_dz

    # Number of vertical levels
    vert_levels = 64

    # Depth of the bottom of the ocean
    bottom_depth = 6000.0

    # The minimum layer thickness
    min_layer_thickness = 2.0

    # The maximum layer thickness
    max_layer_thickness = 210.0

60layerPHC
----------

This is the vertical grid used by the Polar science center Hydrographic Climatology
(`PHC <http://psc.apl.washington.edu/nonwp_projects/PHC/Climatology.html>`_).
Layer thicknesses vary over 60 layers from 5 m at the surface to 250 m at the
seafloor, which is at 5375 m depth.  To get the default grid, use:

.. code-block:: cfg

    # Options related to the vertical grid
    [vertical_grid]

    # the type of vertical grid
    grid_type = 60layerPHC

If the ``bottom_depth`` option is also defined, the depths will be renormalized
so that bottom of the deepest layer is at ``z = -bottom_depth``

42layerWOCE
-----------

This is the vertical grid used by the World Ocean Circulation Experiment
(`WOCE <https://icdc.cen.uni-hamburg.de/en/woce-climatology.html>`_) Global
Hydrographic Climatology. Layer thicknesses vary over 42 layers from 5 m at the
surface to 250 m at the seafloor, which is at 5875 m depth.  To get the default
grid, use:

.. code-block:: cfg

    # Options related to the vertical grid
    [vertical_grid]

    # the type of vertical grid
    grid_type = 42layerWOCE

If the ``bottom_depth`` option is also defined, the depths will be renormalized
so that bottom of the deepest layer is at ``z = -bottom_depth``

100layerE3SMv1
--------------

This is the vertical grid was used in some E3SM v1 experiments. Layer
thicknesses vary over 100 layers from 1.51 m at the surface to 221 m at the
seafloor, which is at 6000 m depth.  To get the default grid, use:

.. code-block:: cfg

    # Options related to the vertical grid
    [vertical_grid]

    # the type of vertical grid
    grid_type = 100layerE3SMv1

If the ``bottom_depth`` option is also defined, the depths will be renormalized
so that bottom of the deepest layer is at ``z = -bottom_depth``.  This is
the default approach in the :ref:`ocean_ziso` configuration:

.. code-block:: cfg

    # Options related to the vertical grid
    [vertical_grid]

    # the type of vertical grid
    grid_type = 100layerE3SMv1

    # Depth of the bottom of the ocean
    bottom_depth = 2500.0

In this case, the thickness of the 100 layers vary between ~0.63 m and 92.1 m,
with the sea floor at 2500 m.
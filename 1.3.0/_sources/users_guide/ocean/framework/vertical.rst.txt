.. _ocean_vertical:

Vertical coordinate
===================

The vertical coordinate used in most MPAS-Ocean test cases is determined by
config options in the ``vertical_grid`` section of the config file:

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

    # The characteristic number of levels over which the index_tanh_dz
    # transition between the min and max occurs
    transition_levels = 28

    # The type of vertical coordinate (e.g. z-level, z-star)
    coord_type = z-star

    # Whether to use "partial" or "full", or "None" to not alter the topography
    partial_cell_type = full

    # The minimum fraction of a layer for partial cells
    min_pc_fraction = 0.1

The vertical coordinate is typically defined based on a 1D reference grid.
Possible 1D grid types are: ``uniform``, ``tanh_dz``, ``index_tanh_dz``,
``60layerPHC``, ``80layerE3SMv1``, and ``100layerE3SMv1``.

The meaning of the config options ``vert_levels``, ``bottom_depth``,
``min_layer_thickness``, ``max_layer_thickness``, and ``transition_levels``
depends grid type and is described below.

The options ``coord_type``, ``partial_cell_type`` and ``min_pc_fraction``
relate to :ref:`ocean_vert_3d`, described below.

1D Grid type
------------

uniform
~~~~~~~

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
~~~~~~~

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
:math:`z_{n_z+1} = -H_\mathrm{bot}`, where :math:`n_z` is ``nVertLevels``, the
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

index_tanh_dz
~~~~~~~~~~~~~

This is similar to ``tanh_dz`` but the hyperbolic tangent function is defined
in layer index space rather than physical depth. Layer thickness is defined by:

.. math::

    \Delta z\left(k\right) = (\Delta z_2 - \Delta z_1)
               \mathrm{tanh}\left[\frac{\pi \left(k - k_0\right)}{\Delta}\right]+ \Delta z_1,

where :math:`\Delta z_1` (``min_layer_thickness``) is the value of the layer
thickness :math:`\Delta z` at :math:`z = 0` and :math:`\Delta z_2`
(``max_layer_thickness``) is the same as :math:`z \rightarrow \infty`.  The
vertical layer index is `k`, `\Delta` (``transition_levels``) is the number of
vertical levels over which the ``tanh`` transitions from the finer to the
coarser resolution, and :math:`k_0` is the origin in vertical index space of the
transition. Interface locations `z_k` are defined by:

.. math::

    z_0 = 0, \\
    z_{k+1} = z_k - \Delta z\left(k\right).

We use a root finder to solve for :math:`k_0`, such that
:math:`z_{n_z+1} = -H_\mathrm{bot}`, where :math:`n_z` is ``vert_levels``, the
number of vertical levels (one less than the number of layer interfaces) and
:math:`H_\mathrm{bot}` is ``bottom_depth``, the depth of the seafloor.

The following config options are all required.  This is an example of a
64-layer vertical grid that has been explored in E3SM v2:

.. code-block:: cfg

    # Options related to the vertical grid
    [vertical_grid]

    # the type of vertical grid
    grid_type = index_tanh_dz

    # Number of vertical levels
    vert_levels = 64

    # Depth of the bottom of the ocean
    bottom_depth = 5500.0

    # The minimum layer thickness
    min_layer_thickness = 10.0

    # The maximum layer thickness
    max_layer_thickness = 250.0

    # The characteristic number of levels over which the index_tanh_dz
    # transition between the min and max occurs
    transition_levels = 28

60layerPHC
~~~~~~~~~~

This is the vertical grid used by the Polar science center Hydrographic Climatology
(`PHC <http://psc.apl.washington.edu/nonwp_projects/PHC/Climatology.html>`_).
Layer thicknesses vary over 60 layers from 10 m at the surface to 250 m at the
seafloor, which is at 5500 m depth.  To get the default grid, use:

.. code-block:: cfg

    # Options related to the vertical grid
    [vertical_grid]

    # the type of vertical grid
    grid_type = 60layerPHC

If the ``bottom_depth`` option is also defined, the depths will be renormalized
so that bottom of the deepest layer is at ``z = -bottom_depth``

80layerE3SMv1
~~~~~~~~~~~~~~

This is the vertical grid was used in some E3SM v1 and v2 meshes, such as the
ARRM10to60 mesh. Layer thicknesses vary over 80 layers from 2 m at the surface
to 146 m at the seafloor, which is at 5550 m depth.  To get the default grid,
use:

.. code-block:: cfg

    # Options related to the vertical grid
    [vertical_grid]

    # the type of vertical grid
    grid_type = 80layerE3SMv1

If the ``bottom_depth`` option is also defined, the depths will be renormalized
so that bottom of the deepest layer is at ``z = -bottom_depth``.

100layerE3SMv1
~~~~~~~~~~~~~~

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
the default approach in the :ref:`ocean_ziso` test group:

.. code-block:: cfg

    # Options related to the vertical grid
    [vertical_grid]

    # the type of vertical grid
    grid_type = 100layerE3SMv1

    # Depth of the bottom of the ocean
    bottom_depth = 2500.0

In this case, the thickness of the 100 layers vary between ~0.63 m and 92.1 m,
with the sea floor at 2500 m.


.. _ocean_vert_3d:

3D vertical coordinates
-----------------------

Currently, ``z-star`` and ``z-level`` vertical coordinates are supported
(``coord_type``).  Each supports 3 options for ``partial_cell_type``: ``full``
meaning the topography (bottom depth and sea-surface height) are expanded so
that all layers have their full thickness; ``partial`` meaning that cells
adjacent to the topography are allowed to be a small fraction of a full layer
thickness; or ``None`` to indicate that no alteration is needed for full or
partial cells (typically only in cases where the topography is already flat).

If ``partial_cell_type = partial``, the option ``min_pc_fraction`` indicates
the smallest fraction of a layer that a partial cell is allowed to have before
it must either be expanded to the minimum or collapsed to the next adjacent
valid layer (whichever would cause the smallest change).

.. _ocean_z_star:

z-star
~~~~~~

Most MPAS-Ocean test cases currently use the z* vertical coordinate
(`Adcroft and Campin, 2004 <https://doi.org/10.1016/j.ocemod.2003.09.003>`_)
by default.  Typically (in the absence of ice-shelf cavities), the initial
"resting" grid uses a :ref:`ocean_z_level` coordinate.  As the sea-surface
height evolves, the coordinate stretches and squashes in proportion to changes
in the local watercolumn thickness.

In configurations with :ref:`ocean_ice_shelf_cavities`, the ice draft
(elevation of the ice shelf-ocean interface) also acts the sea-surface height
for a z-star coordinate.  This means that the initial layers are squashed
significantly from their "resting" thickness under ice shelves as if they were
being pressed down by the weight of the ice.

.. _ocean_z_level:

z-level
~~~~~~~

In the absence of :ref:`ocean_ice_shelf_cavities`, the z-level coordinate in
MPAS-Ocean is the same as the :ref:`ocean_z_star` coordinate.

When ice-shelf cavities are included, rather than depressing the vertical grid
under the weight of the ice, the z-level coordinate used top cells to mask out
parts of the mesh as "land" in the same way that cells below the batymetry are
masked as land.  The topography at the top of the ocean is represented by
"stair steps", using either "full" or "partial" cells to represent these steps
in exactly the same way as at the seafloor.

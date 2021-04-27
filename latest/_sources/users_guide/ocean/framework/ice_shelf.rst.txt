.. _ocean_ice_shelf_cavities:

Ice shelf-cavities
==================

The inclusion of ice-shelf cavities and melt rates below ice shelves around
Antarctica is a major objective of the
`E3SM Cryosphere Campaign <https://e3sm.org/research/cryosphere-ocean/v1-cryosphere-ocean/>`_.
Sub-ice-shelf melt rates are needed in order to estimate future mass loss
from the Antarctic Ice Sheet.  Along with dynamic ocean boundaries, they are an
important component in future coupling between MPAS-Ocean and MALI.

Currently, both the :ref:`ocean_global_ocean` and :ref:`ocean_ice_shelf_2d`
configurations include test cases where the ocean domain includes ice-shelf
cavities.

MPAS-Ocean implements the topography of ice-shelf cavities by allowing the
sea-surface height (SSH) to follow the ice shelf-ocean interface (the ice
draft).  The sea surface is depressed by applying the pressure of the overlying
ice shelf as a top boundary condition.  The :ref:`ocean_vertical` in ice-shelf
cavities is a bit more complex than the simpler z* coordinate used elsewhere
in the ocean domain because ocean layers have to be made thicker or their
slope has to be reduced via smoothing to prevent the Haney number
(`Haney 1991 <https://doi.org/10.1175/1520-0485(1991)021%3C0610:OTPGFO%3E2.0.CO;2>`_)
from becoming too large.

.. _ocean_ssh_adjustment:

Sea surface height adjustment
-----------------------------

Standalone-ocean test cases typically provide the ice draft (which is then used
as the ``ssh``)  rather than the pressure from the weight of the ice shelf
(the ``landIcePressure`` variable).  This is in contrast to coupled
ice sheet-ocean configurations that we expect to support in the future, in
which the weight of the ice is know, rather than the ice draft.  Ideally, the
initial ice draft and the ice-shelf pressure would be consistent with one
another, so that the SSH would remain nearly stationary in time once the
MPAS-Ocean simulation starts.  In practice, this is difficult to achieve.

Typically, ice shelves are assumed to be freely floating on the ocean with
negligible stresses, meaning that the ice draft and the weight
of the ice are related through the average density of the ice and of the
displaced ocean water in a given column.  However, in most circumstances it is
hard to accurately determine the average density of displaced ocean water, and
details of the numerical algorithm for computing the horizontal pressure
gradient can also affect how consistent the ice draft and pressure fields are.

MPAS-Ocean achieves a consistent ice draft and ice-shelf pressure by:

1. making an initial guess that the displaced ocean density is the same as the
   density in the top ocean layer and

2. iteratively performing short (typically 1-hour) forward simulations in which
   the SSH is free to evolve, then modifying the ice-shelf pressure to attempt
   to compensate for changes in the SSH during the forward run.

We have found this approach to be robust over a range of resolutions from <1 km
to 240 km and in both idealized and realistic model configurations.

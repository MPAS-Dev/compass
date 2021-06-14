.. _landice_greenland:

greenland
=========

The ``landice/greenland`` test group runs tests with a coarse (20-km)
`Greenland mesh <https://web.lcrc.anl.gov/public/e3sm/mpas_standalonedata/mpas-albany-landice/gis20km.150922.nc>`_.

.. figure:: images/gis_speed.png
   :width: 742 px
   :align: center

   FO velocity solution visualized in Paraview.

The test group includes 3 test cases, each of which has one or more steps
that are variants on ``run_model`` (given other names in the decomposition and
restart test cases to distinguish multiple model runs), which performs time
integration of the model.

The test cases in this test group can run with either the SIA or FO velocity
solvers. Running with the FO solver requires a build of MALI that includes
Albany, but the SIA variant of the test can be run without Albany.  The FO
version uses no-slip basal boundary condition.

config options
--------------

There are no config options specific to this test group.

smoke_test
----------

``landice/greenland/smoke_test`` is the default version of the greenland test
case for a short (5-day) test run.

decomposition_test
------------------

``landice/greenland/decomposition_test`` runs short (5-day) integrations of the
model forward in time on 1 (``1proc_run`` step) and then on 4 cores
(``4proc_run`` step) to make sure the resulting prognostic variables are
bit-for-bit identical between the two runs.

restart_test
------------

``landice/greenland/2000m/restart_test`` first run a short (5-day) integration
of the model forward in time (``full_run`` step).  Then, a second step
(``restart_run``) performs a 3-day, then a 2-day run, where the second begins
from a restart file saved by the first. Prognostic variables are compared
between the "full" and "restart" runs at year 2 to make sure they are
bit-for-bit identical.

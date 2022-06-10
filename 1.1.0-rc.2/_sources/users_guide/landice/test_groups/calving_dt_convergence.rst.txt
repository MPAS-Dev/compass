.. _landice_calving_dt_convergence:

calving_dt_convergence
======================

The ``landice/calving_dt_convergence`` test group supports tests for
assessing the timestep convergence of calving physics in MALI.  The tests all
use pre-generated meshes.

The test group includes a single test case with many variants for using
different meshes, calving laws, and velocity solver settings.

config options
--------------

There currently are no config options.

dt_convergence_test
-------------------

``landice/calving_dt_convergence/dt_convergence_test`` runs short
integrations repeatedly with different values for the fraction of the
calving CFL limit applied in the adpative timestepper.  Time series of the
total calving flux and the calving CFL to actual timestep ratio are then
plotted, as well as summary of the number of calving warnings for each choice
of calving CFL fraction (see below).
The individual tests are named for the mesh, the calving law, and the
velocity solver setting, separated by periods.

.. figure:: images/calving_dt_comparison.png
   :width: 777 px
   :align: center

   Example results of calving dt test for the
   ``humboldt.specified_calving_velocity.none`` test.  The top plot
   shows the total calving flux over time for different choices of
   the calving CFL fraction.  Results should be similar for small
   fraction values.  As the fraction is increased,
   significant differences can be seen, indicating values greater than
   about 1.0 are less accurate.  The second panel shows the cumulative
   total calving flux.  The third panel shows the actual dt used by the
   adaptive timestepper divided by the timestep limit due to the
   calving CFL.  This allows assessment of the current implementation
   limitation that the calving CFL limit is lagged by one time step;
   if this value is significantly less than the fraction specified,
   the method of lagging the calving CFL is not accurate.  The final
   panel shows the number of calving warnings (left axis) and the
   fraction of total timesteps with warnings (right axis) for each
   value of the calving CFL fraction that was run.  Warnings are
   triggered when the calving routine is unable to calve more than 1%
   of the calving volume it was asked to calve.

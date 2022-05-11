.. _landice_mismipplus:

mismipplus
==========

The ``landice/mismipplus`` test group runs tests with a 2km
`MISMIP+ mesh <https://web.lcrc.anl.gov/public/e3sm/mpas_standalonedata/mpas-albany-landice/MISMIP_2km_20220502.nc>`_.
The purpose of this test group is to provide a commonly used idealized
glacier configuration that includes an ice shelf.
The mesh and initial condition are already generated.  In the future,
additional test cases may be added for generating a new version of the
mesh at different resolutions and using different data sources.
See the
`MISMIP+ description paper <https://tc.copernicus.org/articles/14/2283/2020/>`_
for more details.  Note that the 2km mesh is not the most accurate resolution
used by MALI, but provides a convenient trade off between computational cost
and accuracy for running tests.

The test group only includes a single test - a smoke test.
More tests may be added later.
There are no config options.

smoke_test
----------

``landice/mismipplus/smoke_test`` runs the MISMIP+ test mesh for 5 years
with the adaptive timestepper on.  It is meant as a baseline configuration
that could be adjusted to specific needs.

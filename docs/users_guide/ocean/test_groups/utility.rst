.. _ocean_utility:

utility
=======

The ``utility`` test group is a general test group for ocean utilities that
may create datasets for other test groups to use.  It is partly designed to
provide provenance for data processing that may be more complex than a single,
short script.

combine_topo/lat_lon
--------------------
The ``ocean/utility/combine_topo/lat_lon`` test case is used to combine the
`BedMachine Antarctica v3 <https://nsidc.org/data/nsidc-0756/versions/3>`_
dataset with the `GEBCO 2023 <https://www.gebco.net/data_and_products/gridded_bathymetry_data/>`_
dataset.  The result is on a 1/80 degree latitude-longitude grid.  This utility
is intended for provenance.  It is intended to document the process for
producing the topography dataset used for E3SM ocean meshes.

combine_topo/cubed_sphere
-------------------------
The ``ocean/utility/combine_topo/cubed_sphere`` test case is used to combine the
`BedMachine Antarctica v3 <https://nsidc.org/data/nsidc-0756/versions/3>`_
dataset with the `GEBCO 2023 <https://www.gebco.net/data_and_products/gridded_bathymetry_data/>`_
dataset.  The result is on an ne3000 cubed sphere grid.  This utility
is intended for provenance.  It is intended to document the process for
producing the topography dataset used for E3SM ocean meshes.

cull_restarts
-------------
The ``ocean/utility/cull_restarts`` test case is used to cull ice-shelf
cavities from MPAS-Ocean and -Seaice restart files.  It is intended for
expert users wanting to create an E3SM branch run without ice-shelf cavities
from a previous run that included cavities.

extrap_woa
----------
The ``ocean/utility/extrap_woa`` test case is used to extrapolate
`WOA 2023 <https://www.ncei.noaa.gov/products/world-ocean-atlas>`_ data into
ice-shelf cavities and coastal regions, then into land, grounded ice and below
bathymetry.  It is provided mainly for developers to update for use on
future datasets and is not intended for users, so we do not provide full
documentation here.

create_salin_restoring
----------------------
The ``ocean/utility/create_salin_restoring`` test case is used to download
monthly average `WOA 2023 <https://www.ncei.noaa.gov/products/world-ocean-atlas>`_
data, extracts the surface layer and extrapolates into ice-shelf cavities and
across continents.  The fields are then remapped to a cubed-sphere grid at
~10 km (ne300) resolution for easier remapping to MPAS meshes.  This testcase
is provided mainly for developers to update the salinity restoring data and is
not intended for users, so we do not provide full documentation here.

.. _ocean_utility:

utility
=======

The ``utility`` test group is a general test group for ocean utilities that
may create datasets for other test groups to use.  It is partly designed to
provide provenance for data processing that may be more complex than a single,
short script.

extrap_woa
----------
The ``ocean/utility/extrap_woa`` test case is used to extrapolate
`WOA 2023 <https://www.ncei.noaa.gov/products/world-ocean-atlas>`_ data into
ice-shelf cavities and coastal regions, then into land, grounded ice and below
bathymetry.  It is provided mainly for developers to update for use on
future datasets and is not intended for users, so we do not provide full
documentation here.
.. _dev_ocean_utility:

utility
=======

The ``utility`` test group is a general test group for ocean utilities that
may create datasets for other test groups to use.  It is partly designed to
provide provenance for data processing that may be more complex than a single,
short script.

combine_topo
------------
The class :py:class:`compass.ocean.tests.utility.combine_topo.CombineTopo`
defines a test case for combining the
`BedMachine Antarctica v3 <https://nsidc.org/data/nsidc-0756/versions/3>`_
dataset with the `GEBCO 2023 <https://www.gebco.net/data_and_products/gridded_bathymetry_data/>`_
dataset.

combine
~~~~~~~
The class :py:class:`compass.ocean.tests.utility.combine_topo.Combine`
defines a step for combining the datasets above. The GEBCO and BedMachine data
are remapped to a common global grid to make later remapping to MPAS meshes
more manageable, and the two datasets are blended between 60 and 62 degrees
south latitude. The GEBCO global dataset is divided into regional tiles prior
to remapping to improve performance. Two common global target grid options are
provided: a 1/80 degree latitude-longitude grid and an ne3000 cubed sphere
grid. These target grid options are selectable via the ``target_grid`` argument
in the :py:class:`compass.ocean.tests.utility.combine_topo.CombineTopo` test
case class.

cull_restarts
-------------
The class :py:class:`compass.ocean.tests.utility.cull_restarts.CullRestarts`
defines a test case for culling ice-shelf cavities from MPAS-Ocean and -Seaice
restart files.

cull
~~~~

The class :py:class:`compass.ocean.tests.utility.cull_restarts.Cull` defines
a step for culling ice-shelf cavities from MPAS-Ocean and -Seaice
restart files.  The step produces a culled mesh file, graph file, maps between
the unculled and culled meshes in addition to culled versions of the restart
files.

extrap_woa
----------
The class :py:class:`compass.ocean.tests.utility.extrap_woa.ExtrapWoa`
defines a test case for extrapolating
`WOA 2023 <https://www.ncei.noaa.gov/products/world-ocean-atlas>`_ data into
ice-shelf cavities and coastal regions, then into land, grounded ice and below
bathymetry.

combine
~~~~~~~

The class :py:class:`compass.ocean.tests.utility.extrap_woa.Combine` defines
a step for combining the Annual and January temperature and salinity
climatology data into a single file.  We use the 30-year objectively analyzed
climatology (1991-2020) data sets on a 0.25 degree lon-lat grid.  The WOA data
is provided only in the top 1500 m of the ocean in the monthly files, whereas
it extends to 5500 m depth in the annual dataset.  The test case uses the
January data where it is available and the annual data when monthly data is
not provided. Finally, the step converts in situ temperature to potential
temperature.

remap_topography
~~~~~~~~~~~~~~~~

The class :py:class:`compass.ocean.tests.utility.extrap_woa.RemapTopography`
defines a step for conservatively remapping a topography dataset (see
:ref:`dev_ocean_framework_remap_topography`) from a very high resolution
grid to the WOA 2023 0.25 degree grid.  The topography and masks are then used
to aid in the extrapolation process.  Because of the very high resolution
source grid, at least 360 cores are needed to make the mapping file in this
step.

extrap_step
~~~~~~~~~~~

The class :py:class:`compass.ocean.tests.utility.extrap_woa.ExtrapStep`
defines a step for extrapolating the data into invalid regions.  Because we
will use the dataset to interpolate potential temperature and salinity at the
centers of MPAS grid cells, we want there to be valid data not just where WOA
2023 has it defined but also in the cavities below Antarctic ice shelves, in a
buffer region into land, and below the bathymetry.  This way, the interpolation
will be sure never to be contaminated with NaNs or other invalid values.

The ice-shelf cavities provide a particular challenge.  No temperature or
salinity data is available for them in WOA 2023 but they cover vast areas so
it is not safe to simply extrapolate from the nearest point available in
the WOA dataset.  When we have done this in the past, properties deep in the
Filchner-Ronne ice-shelf cavity come from the Bellinghshausen Sea, which is
geographically close by but topographically disconnected from the cavity.  To
avoid this problem, we perform extrapolation in 4 steps.

1. Extrapolate horizontally from WOA 2023 data into regions that the topography
   dataset indicates are ocean.  This includes anywhere where the top of a WOA
   layer is above the bathymetry that is not covered the grounded antarctic ice
   sheet.  This means that we extrapolate not only into ice-shelf cavities but
   also into the ice-shelves above them.  Past experience has shown this to be
   a helpful approach.

2. Extrapolate vertically downward from the surface to the seafloor.  This
   fills in regions that are horizontally blocked by topography from any valid
   WOA data (e.g. the deep interiors of ice-shelf cavities) but which are still
   either in the ocean or in floating ice shelves.

3. Extrapolate horizontally from the results of step 2 to fill in land,
   grounded ice sheet areas, and below the bathymetry.  here, we will likely
   only ever use the data in a small halo around the "valid" ocean region when
   we interpolate it to MPAS-Ocean meshes but we fill in everywhere "just in
   case".

4. Finally, we extrapolate vertically downward one last time all the way to the
   bottom layer of the dataset.  This won't do anything unless there happen to
   be layers with no valid WOA data at all (which is not presently the case).

The resulting file is ready to be placed in compass' initial condition database
(see :ref:`dev_step_input_download` for details on databases).

create_salin_restoring
----------------------
The class :py:class:`compass.ocean.tests.utility.create_salin_restoring.CreateSalinRestoring`
defines a test case for creating a monthly average sea surface salinity dataset based on the
`WOA 2023 <https://www.ncei.noaa.gov/products/world-ocean-atlas>`_ data.  It also extrapolates
the twelve months of data into ice-shelf cavities and across continents.

combine
~~~~~~~

The class :py:class:`compass.ocean.tests.utility.create_salin_restoring.Combine`
defines a step to download and combine January through December sea surface salinity into a single
file that serves as the base dataset for salinity restoring in forced ocean sea-ice cases (FOSI).
The surface level of WOA 2023 data is utilized.

The reference date for each month of data is assumed to be the 15th of each month.  In a simulation,
this implies that for a model start time of January 1, the salinity is restored to the average of
the December and January sea surface salinities.

extrap
~~~~~~

The class :py:class:`compass.ocean.tests.utility.create_salin_restoring.Extrap`
defines a step to extrapolate the combined January through December sea surface salinities
into missing ocean regions such as ice-shelf cavities and across continents.  Since this is only
extrapolation of surface values, masks are not utilized.

remap
~~~~~

The class :py:class:`compass.ocean.tests.utility.create_salin_restoring.Remap`
defines a step to remap the extrapolated data to a cubed-sphere grid at ne300
(~10 km) resolution. The cubed-sphere grid is much more favorable to remapping
to MPAS meshes, particularly at the poles and with significant smoothing.

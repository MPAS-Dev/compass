import numpy as np

from alphaBetaLab.alphaBetaLab.abEstimateAndSave import (
    abEstimateAndSaveTriangularEtopo1,
    triMeshSpecFromMshFile,
)
# importing from alphaBetaLab the needed components
from alphaBetaLab.alphaBetaLab.abOptionManager import abOptions

# from compass.mesh import QuasiUniformSphericalMeshStep


class WavesUostFiles():
    """
    A step for creating the unresolved obstacles file for wave mesh
    """
    # pass

    def __init__(self, test_case):
        """
        Create a new step
        """
        super().__init__(test_case, name='uost_files')

    def run(self):
        """
        Run this step
        """
        super().run()

        dirs = np.linspace(0, 2 * np.pi, 36)
        nfreq = 50   # ET NOTE: this should be flexible
        minfrq = .035
        if (nfreq == 50):
            frqfactor = 1.07
        elif (nfreq == 36):
            frqfactor = 1.10
        elif (nfreq == 25):
            frqfactor = 1.147
            # PRINT SOME ERROR MESSAGE FOR NON-supoorted spectral reslution

        freqs = [minfrq * (frqfactor ** i) for i in range(1, nfreq + 1)]

        # definition of the spatial mesh
        gridname = 'glo_unst'
        mshfile = 'global.msh'
        triMeshSpec = triMeshSpecFromMshFile(mshfile)

        # path of the etopo1 bathymetry
        # etopoFilePath = '/users/sbrus/scratch4/ \
        # WW3_unstructured/GEBCO_2019.nc'
        etopoFilePath = './etopo1_180.nc'

        # output directory
        outputDestDir = './output/'

        # number of cores for parallel computing
        nParWorker = 1

        # this option indicates that the computation
        # should be skipped for cells smaller than 3 km
        minSizeKm = 3
        opt = abOptions(minSizeKm=minSizeKm)

        # instruction to do the computation and save the output
        # abEstimateAndSaveTriangularGebco(
        # dirs, freqs, gridname, triMeshSpec,
        # etopoFilePath, outputDestDir, nParWorker, abOptions=opt)
        abEstimateAndSaveTriangularEtopo1(
            dirs, freqs, gridname, triMeshSpec, etopoFilePath,
            outputDestDir, nParWorker, abOptions=opt)

    # def build_cell_width_lat_lon(self):
    #    """
    #    Create cell width array for this mesh on a regular latitude-longitude
    #    grid

    #    Returns
    #    -------
    #    cellWidth : numpy.array
    #        m x n array of cell width in km

    #    lon : numpy.array
    #        longitude in degrees (length n and between -180 and 180)

    #    lat : numpy.array
    #        longitude in degrees (length m and between -90 and 90)
    #    """

    #    dlon = 10.
    #    dlat = 0.1
    #    nlon = int(360. / dlon) + 1
    #    nlat = int(180. / dlat) + 1
    #    lon = np.linspace(-180., 180., nlon)
    #    lat = np.linspace(-90., 90., nlat)

    #    cellWidthVsLat = mdt.EC_CellWidthVsLat(lat)
    #    cellWidth = np.outer(cellWidthVsLat, np.ones([1, lon.size]))

    #    return cellWidth, lon, lat

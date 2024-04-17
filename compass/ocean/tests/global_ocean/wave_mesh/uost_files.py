import numpy as np

from alphaBetaLab.alphaBetaLab.abEstimateAndSave import (
    abEstimateAndSaveTriangularEtopo1,
    triMeshSpecFromMshFile,
)
# importing from alphaBetaLab the needed components
from alphaBetaLab.alphaBetaLab.abOptionManager import abOptions
from compass import Step


class WavesUostFiles(Step):
    """
    A step for creating the unresolved obstacles file for wave mesh
    """
    def __init__(self, test_case, wave_culled_mesh,
                 name='uost_files', subdir=None):

        super().__init__(test_case=test_case, name=name, subdir=subdir)

        # other things INIT should do?
        culled_mesh_path = wave_culled_mesh.path
        self.add_input_file(
            filename='wave_mesh_culled.msh',
            work_dir_target=f'{culled_mesh_path}/wave_mesh_culled.msh')

        self.add_input_file(
            filename='etopo1_180.nc',
            target='etopo1_180.nc',
            database='bathymetry_database')

    def run(self):
        """
        Create unresolved obstacles for wave mesh and spectral resolution
        """

        dirs = np.linspace(0, 2 * np.pi, 36)
        nfreq = 50   # ET NOTE: this should be flexible
        minfrq = .035
        if (nfreq == 50):
            frqfactor = 1.07
        elif (nfreq == 36):
            frqfactor = 1.10
        elif (nfreq == 25):
            frqfactor = 1.147
        else:
            print("ERROR: Spectral resolution not supported.")
            print("Number of wave freqencies must be 25, 36, or 50.")

        freqs = [minfrq * (frqfactor ** i) for i in range(1, nfreq + 1)]

        # definition of the spatial mesh
        gridname = 'glo_unst'  # SB NOTE: should be flexible
        mshfile = 'wave_mesh_culled.msh'
        triMeshSpec = triMeshSpecFromMshFile(mshfile)

        # path of the etopo1 bathymetry
        # etopoFilePath = '/users/sbrus/scratch4/ \
        # WW3_unstructured/GEBCO_2019.nc'
        etopoFilePath = 'etopo1_180.nc'

        # output directory
        outputDestDir = 'output/'

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

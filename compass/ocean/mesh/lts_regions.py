import math
import subprocess as sp

import netCDF4 as nc
import numpy as np
import xarray
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from compass.step import Step


class LTSRegionsStep(Step):
    """
    A step for adding LTS regions to a global MPAS-Ocean mesh

    Attributes
    ----------
    cull_mesh_step : CullMeshStep
        The culled mesh step containing input files to this step
    """
    def __init__(self, test_case, cull_mesh_step,
                 name='lts_regions', subdir='cull_mesh'):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.mesh.Mesh
            The test case this step belongs to

        cull_mesh_step : cull_mesh_step
            The culled mesh step containing input files to this step

        name : str, optional
            the name of the step

        subdir : str, optional
            the subdirectory for the step.  The default is ``name``
        """
        super().__init__(test_case, name=name, subdir=subdir,
                         cpus_per_task=None, min_cpus_per_task=None)
        self.cull_mesh_step = cull_mesh_step

    def setup(self):
        """
        Set up the test case in the work directory, including downloading any
        dependencies.
        """
        super().setup()

        config = self.cull_mesh_step.config
        self.cpus_per_task = config.getint('spherical_mesh',
                                           'cull_mesh_cpus_per_task')
        self.min_cpus_per_task = config.getint('spherical_mesh',
                                               'cull_mesh_min_cpus_per_task')

    def run(self):
        """
        Run this step of the test case
        """

        fine_region = get_fine_region(region='east_us_coast')

        label_mesh(fine_region, mesh='culled_mesh.nc',
                   graph_info='culled_graph.info', num_interface=2)


def get_fine_region(region):

    # [lat, lon] points defining the fine region
    westernAtlanticRegionPts = np.array([[0.481, -1.737 + 2 * math.pi],
                                         [0.311, -1.701 + 2 * math.pi],
                                         [0.234, -1.508 + 2 * math.pi],
                                         [0.148, -1.430 + 2 * math.pi],
                                         [0.151, -1.397 + 2 * math.pi],
                                         [0.163, -1.383 + 2 * math.pi],
                                         [0.120, -1.320 + 2 * math.pi],
                                         [0.077, -0.921 + 2 * math.pi],
                                         [0.199, -0.784 + 2 * math.pi],
                                         [0.496, -0.750 + 2 * math.pi],
                                         [0.734, -0.793 + 2 * math.pi],
                                         [0.826, -0.934 + 2 * math.pi],
                                         [0.871, -1.001 + 2 * math.pi],
                                         [0.897, -0.980 + 2 * math.pi],
                                         [0.914, -1.012 + 2 * math.pi],
                                         [0.850, -1.308 + 2 * math.pi],
                                         [0.743, -1.293 + 2 * math.pi],
                                         [0.638, -1.781 + 2 * math.pi],
                                         [0.481, -1.737 + 2 * math.pi]])

    eastUSCoastRegionPts = np.array([[0.532, 4.862],
                                     [0.520, 4.946],
                                     [0.523, 5.018],
                                     [0.548, 5.082],
                                     [0.596, 5.131],
                                     [0.639, 5.159],
                                     [0.690, 5.175],
                                     [0.731, 5.168],
                                     [0.760, 5.147],
                                     [0.777, 5.148],
                                     [0.790, 5.181],
                                     [0.813, 5.048],
                                     [0.556, 4.775]])

    delawareCoastRegionPts = np.array([[0.598, 4.929],
                                       [0.591, 4.953],
                                       [0.591, 4.973],
                                       [0.596, 4.991],
                                       [0.607, 5.001],
                                       [0.620, 5.005],
                                       [0.633, 5.005],
                                       [0.642, 4.999],
                                       [0.683, 5.028],
                                       [0.689, 5.046],
                                       [0.697, 5.057],
                                       [0.709, 5.064],
                                       [0.719, 5.063],
                                       [0.726, 5.059],
                                       [0.727, 5.058],
                                       [0.727, 5.053],
                                       [0.730, 5.050],
                                       [0.745, 5.011],
                                       [0.654, 4.883]])

    if region == 'east_us_coast':
        fine_region = Polygon(eastUSCoastRegionPts)
    elif region == 'western_atlantic':
        fine_region = Polygon(westernAtlanticRegionPts)
    elif region == 'delaware_coast':
        fine_region = Polygon(delawareCoastRegionPts)
    else:
        print('Desired region is undefined')
        quit()

    return fine_region


def label_mesh(fine_region, mesh, graph_info, num_interface):   # noqa: C901

    # read in mesh data
    ds = xarray.open_dataset(mesh)
    nCells = ds['nCells'].size
    nEdges = ds['nEdges'].size
    areaCell = ds['areaCell'].values
    cellsOnEdge = ds['cellsOnEdge'].values
    edgesOnCell = ds['edgesOnCell'].values
    latCell = ds['latCell']
    lonCell = ds['lonCell']

    # start by setting all cells to coarse
    LTSRgn = [2] * nCells

    # check each cell, if in the fine region, label as fine
    print('Labeling fine cells...')
    for iCell in range(0, nCells):
        cellPt = Point(latCell[iCell], lonCell[iCell])
        if fine_region.contains(cellPt):
            LTSRgn[iCell] = 1

    # first layer of cells with label 5
    changedCells = [[], []]
    for iEdge in range(0, nEdges):
        cell1 = cellsOnEdge[iEdge, 0] - 1
        cell2 = cellsOnEdge[iEdge, 1] - 1

        if (cell1 != -1 and cell2 != -1):
            if (LTSRgn[cell1] == 1 and LTSRgn[cell2] == 2):

                LTSRgn[cell2] = 5
                changedCells[0].append(cell2)

            elif (LTSRgn[cell1] == 2 and LTSRgn[cell2] == 1):

                LTSRgn[cell1] = 5
                changedCells[0].append(cell1)

    # second and third layer of cells with label 5
    # only looping over cells changed during loop for previous layer
    # at the end of this loop, changedCells[0] will have the list of cells
    # sharing edegs with the coarse cells
    print('Labeling interface-adjacent fine cells...')
    for i in range(0, 2):  # this loop creates 2 layers
        changedCells[(i + 1) % 2] = []

        for iCell in changedCells[i % 2]:
            edges = edgesOnCell[iCell]
            for iEdge in edges:
                if iEdge != 0:
                    cell1 = cellsOnEdge[iEdge - 1, 0] - 1
                    cell2 = cellsOnEdge[iEdge - 1, 1] - 1

                    if (cell1 != -1 and cell2 != -1):
                        if (LTSRgn[cell1] == 5 and LTSRgn[cell2] == 2):

                            LTSRgn[cell2] = 5
                            changedCells[(i + 1) % 2].append(cell2)

                        elif (LTSRgn[cell1] == 2 and LTSRgn[cell2] == 5):

                            LTSRgn[cell1] = 5
                            changedCells[(i + 1) % 2].append(cell1)

    # n layers of interface region with label 4
    print('Labeling interface cells...')
    for i in range(0, num_interface):
        changedCells[(i + 1) % 2] = []

        for iCell in changedCells[i % 2]:
            edges = edgesOnCell[iCell]
            for iEdge in edges:
                if iEdge != 0:
                    cell1 = cellsOnEdge[iEdge - 1, 0] - 1
                    cell2 = cellsOnEdge[iEdge - 1, 1] - 1

                    if (cell1 != -1 and cell2 != -1):
                        # for the first layer, need to check neighbors are
                        # 5 and 2
                        # for further layers, need to check neighbors are
                        # 3 and 2
                        if (i == 0):
                            if (LTSRgn[cell1] == 5 and LTSRgn[cell2] == 2):

                                LTSRgn[cell2] = 3
                                changedCells[(i + 1) % 2].append(cell2)

                            elif (LTSRgn[cell1] == 2 and LTSRgn[cell2] == 5):

                                LTSRgn[cell1] = 3
                                changedCells[(i + 1) % 2].append(cell1)

                        else:
                            if (LTSRgn[cell1] == 3 and LTSRgn[cell2] == 2):

                                LTSRgn[cell2] = 3
                                changedCells[(i + 1) % 2].append(cell2)

                            elif (LTSRgn[cell1] == 2 and LTSRgn[cell2] == 3):

                                LTSRgn[cell1] = 3
                                changedCells[(i + 1) % 2].append(cell1)

    changedCells[0] = changedCells[num_interface % 2]

    # n layers of interface region with label 3
    for i in range(0, num_interface):
        changedCells[(i + 1) % 2] = []

        for iCell in changedCells[i % 2]:
            edges = edgesOnCell[iCell]
            for iEdge in edges:
                if iEdge != 0:
                    cell1 = cellsOnEdge[iEdge - 1, 0] - 1
                    cell2 = cellsOnEdge[iEdge - 1, 1] - 1

                    if (cell1 != -1 and cell2 != -1):
                        # for the first layer, need to check neighbors are
                        # 5 and 2
                        # for further layers, need to check neighbors are
                        # 3 and 2
                        if (i == 0):
                            if (LTSRgn[cell1] == 3 and LTSRgn[cell2] == 2):

                                LTSRgn[cell2] = 4
                                changedCells[(i + 1) % 2].append(cell2)

                            elif (LTSRgn[cell1] == 2 and LTSRgn[cell2] == 3):

                                LTSRgn[cell1] = 4
                                changedCells[(i + 1) % 2].append(cell1)
                            # END if
                        else:
                            if (LTSRgn[cell1] == 4 and LTSRgn[cell2] == 2):

                                LTSRgn[cell2] = 4
                                changedCells[(i + 1) % 2].append(cell2)

                            elif (LTSRgn[cell1] == 2 and LTSRgn[cell2] == 4):

                                LTSRgn[cell1] = 4
                                changedCells[(i + 1) % 2].append(cell1)

    # create lts_mesh.nc

    print('Adding LTSRegion to ' + mesh + '...')

    # open mesh nc file to be appended
    meshNC = nc.Dataset(mesh, 'a', format='NETCDF4_64BIT_OFFSET')

    try:
        # try to get LTSRegion and assign new value
        ltsRgnNC = meshNC.variables['LTSRegion']
        ltsRgnNC[:] = LTSRgn[:]
    except KeyError:
        # create new variable
        nCellsNC = meshNC.dimensions['nCells'].name
        ltsRgnsNC = meshNC.createVariable('LTSRegion', np.int32, (nCellsNC,))

        # set new variable
        ltsRgnsNC[:] = LTSRgn[:]

    meshNC.close()

    shCommand = 'paraview_vtk_field_extractor.py --ignore_time \
                 -d maxEdges=0 -v allOnCells -f ' + mesh + ' \
                 -o lts_mesh_vtk'
    sp.call(shCommand.split())

    # label cells in graph.info

    print('Weighting ' + graph_info + '...')

    fineCells = 0
    coarseCells = 0

    newf = ""
    with open(graph_info, 'r') as f:
        lines = f.readlines()
        # this is to have fine, coarse and interface be separate for METIS
        # newf += lines[0].strip() + " 010 3 \n"

        # this is to have fine, and interface be together for METIS
        newf += lines[0].strip() + " 010 2 \n"
        for iCell in range(1, len(lines)):
            if (LTSRgn[iCell - 1] == 1 or LTSRgn[iCell - 1] == 5):  # fine

                # newf+= "0 1 0 " + lines[iCell].strip() + "\n"
                newf += "0 1 " + lines[iCell].strip() + "\n"
                fineCells = fineCells + 1

            elif (LTSRgn[iCell - 1] == 2):  # coarse
                # newf+= "1 0 0 " + lines[iCell].strip() + "\n"
                newf += "1 0 " + lines[iCell].strip() + "\n"
                coarseCells = coarseCells + 1

            else:  # interface 1 and 2
                # newf+= "0 0 1 " + lines[iCell].strip() + "\n"
                newf += "0 1 " + lines[iCell].strip() + "\n"
                coarseCells = coarseCells + 1

    with open(graph_info, 'w') as f:
        f.write(newf)

    maxArea = max(areaCell)
    minArea = min(areaCell)
    maxWidth = 2 * np.sqrt(maxArea / math.pi) / 1000
    minWidth = 2 * np.sqrt(minArea / math.pi) / 1000
    areaRatio = maxArea / minArea
    widthRatio = maxWidth / minWidth
    numberRatio = coarseCells / fineCells

    txt = 'number of fine cells = {}\n'.format(fineCells)
    txt += 'number of coarse cells = {}\n'.format(coarseCells)
    txt += 'ratio of coarse to fine cells = {}\n'.format(numberRatio)
    txt += 'ratio of largest to smallest cell area = {}\n'.format(areaRatio)
    txt += 'ratio of largest to smallest cell width = {}\n'.format(widthRatio)
    txt += 'number of interface layers = {}\n'.format(num_interface)

    print(txt)

    with open('lts_mesh_info.txt', 'w') as f:
        f.write(txt)

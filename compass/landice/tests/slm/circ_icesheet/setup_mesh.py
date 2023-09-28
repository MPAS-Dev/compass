import os
import shutil

import numpy as np
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.planar_hex import make_planar_hex_mesh
from mpas_tools.scrip.from_mpas import scrip_from_mpas
from netCDF4 import Dataset as NetCDFFile

from compass.model import make_graph_file
from compass.step import Step


class SetupMesh(Step):
    """
    A step for creating a mesh with initial conditions plus forcing file for
    circular icesheet test

    Attributes
    ----------
    """

    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='setup_mesh')

        self.add_output_file(filename='graph.info')
        self.add_output_file(filename='landice_grid.nc')

    # no setup() method is needed

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        config = self.config
        section = config['circ_icesheet']

        nx = section.getint('nx')
        ny = section.getint('ny')
        dc = section.getfloat('dc')
        print('calling the mesh creation function')
        dsMesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc,
                                      nonperiodic_x=True,
                                      nonperiodic_y=True)
        dsMesh = cull(dsMesh, logger=logger)
        dsMesh = convert(dsMesh, logger=logger)
        write_netcdf(dsMesh, 'mpas_grid.nc')

        levels = 3
        args = ['create_landice_grid_from_generic_MPAS_grid.py',
                '-i', 'mpas_grid.nc',
                '-o', 'landice_grid.nc',
                '-l', str(levels)]

        check_call(args, logger)

        make_graph_file(mesh_filename='landice_grid.nc',
                        graph_filename='graph.info')

        _setup_circsheet_initial_conditions(config, logger,
                                            filename='landice_grid.nc')

        _create_smb_forcing_file(config, logger,
                                 mali_mesh_file='landice_grid.nc',
                                 filename='smb_forcing.nc')
        logger.info('calling building mapping files')
        _build_mapping_files(config, logger,
                             mali_mesh_file='landice_grid.nc')


def _setup_circsheet_initial_conditions(config, logger, filename):
    """
    Create initial condition for circular ice sheet

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for this test case, a combination of the defaults
        for the machine, core and configuration

    logger : logging.Logger
        A logger for output from the step

    filename : str
        file to setup circ_icesheet
    """
    section = config['circ_icesheet']
    ice_type = section.get('ice_type')
    set_topo_elev = section.getboolean('set_topo_elev')
    topo_elev = section.get('topo_elev')
    r0 = section.get('r0')
    h0 = section.get('h0')
    put_origin_on_a_cell = section.getboolean('put_origin_on_a_cell')

    # Open the file, get needed dimensions
    gridfile = NetCDFFile(filename, 'r+')
    nVertLevels = len(gridfile.dimensions['nVertLevels'])
    # Get variables
    xCell = gridfile.variables['xCell']
    yCell = gridfile.variables['yCell']
    xEdge = gridfile.variables['xEdge']
    yEdge = gridfile.variables['yEdge']
    xVertex = gridfile.variables['xVertex']
    yVertex = gridfile.variables['yVertex']
    thickness = gridfile.variables['thickness']
    bedTopography = gridfile.variables['bedTopography']
    layerThicknessFractions = gridfile.variables['layerThicknessFractions']

    # Find center of domain
    x0 = xCell[:].min() + 0.5 * (xCell[:].max() - xCell[:].min())
    y0 = yCell[:].min() + 0.5 * (yCell[:].max() - yCell[:].min())
    # Calculate distance of each cell center from dome center
    r = ((xCell[:] - x0) ** 2 + (yCell[:] - y0) ** 2) ** 0.5

    if put_origin_on_a_cell:
        # Center the ice in the center of the cell that is closest to the
        # center of the domain.
        centerCellIndex = np.abs(r[:]).argmin()
        xShift = -1.0 * xCell[centerCellIndex]
        yShift = -1.0 * yCell[centerCellIndex]
        xCell[:] = xCell[:] + xShift
        yCell[:] = yCell[:] + yShift
        xEdge[:] = xEdge[:] + xShift
        yEdge[:] = yEdge[:] + yShift
        xVertex[:] = xVertex[:] + xShift
        yVertex[:] = yVertex[:] + yShift
        # Now update origin location and distance array
        x0 = 0.0
        y0 = 0.0
        r = ((xCell[:] - x0) ** 2 + (yCell[:] - y0) ** 2) ** 0.5
        print('HH==max r is', max(r))
        print('HH==min r is', min(r))
    # Assign variable values for the circular ice sheet
    # Set default value for non-circular cells
    thickness[:] = 0.0
    # Calculate the dome thickness for cells within the desired radius
    # (thickness will be NaN otherwise)
    thickness_field = thickness[0, :]
    r0 = 6000000.0 * np.sqrt(0.125)

    print('r0', r0)
    print('r', r)
    if ice_type == 'cylinder':
        logger.info('cylinder ice type is chosen')
        thickness_field[r < r0] = h0
    elif ice_type == 'dome-cism':
        thickness_field[r < r0] = h0 * (1.0 - (r[r < r0] / r0) ** 2) ** 0.5
    elif ice_type == 'dome-halfar':
        thickness_field[r < r0] = h0 * (
            1.0 - (r[r < r0] / r0) ** (4.0 / 3.0)) ** (3.0 / 7.0)
    else:
        raise ValueError('Unexpected ice_type: {}'.format(ice_type))
    thickness[0, :] = thickness_field

    # flat bed at sea level
    bedTopography[:] = 0.0
    if set_topo_elev:
        # this line will make a small shelf:
        bedTopography[:] = topo_elev

    # Setup layerThicknessFractions
    layerThicknessFractions[:] = 1.0 / nVertLevels

    gridfile.close()

    logger.info('Successfully added circular initial conditions to: {}'.format(
        filename))


def _create_smb_forcing_file(config, logger, mali_mesh_file, filename):
    """
    Create surface mass balance forcing file for circular ice sheet

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for this test case, a combination of the defaults
        for the machine, core and configuration

    logger : logging.Logger
        A logger for output from the step

    mali_mesh_file : str
        mesh file created in the current step

    filename : str
        file to setup circ_icesheet
    """
    section = config['circ_icesheet']
    r0 = section.get('r0')
    h0 = section.get('h0')
    r0 = 6000000.0 * np.sqrt(0.125)  # HH: comment this line out later

    section = config['smb_forcing']
    direction = section.get('direction')

    # other constants used in the fuction
    pi = 3.1415926535898  # pi value used in the SLM
    rhoi = 910.0  # ice density

    # Open the file, get needed dimensions
    shutil.copy(mali_mesh_file, filename)
    smbfile = NetCDFFile(filename, 'r+')

    # Get variables
    xCell = smbfile.variables['xCell']
    yCell = smbfile.variables['yCell']
    # nCells = smbfile.dimensions['nCells']
    sfcMassBal = smbfile.variables['sfcMassBal']

    # set xtime variable
    # smbfile.variables['xtime']

    # initialize SMB value everywhere in the mesh
    sfcMassBal[:, :] = 0.0

    # define the time (1 year interval)
    dt = 1
    t_array = np.arange(0, 10, dt)
    x0 = 0.0
    y0 = 0.0
    r = ((xCell[:] - x0) ** 2 + (yCell[:] - y0) ** 2) ** 0.5

    dVdt = -9.678E13
    dHdt = 20.0
    dRdt = 40000.0

    if direction == 'vertical':
        print('creating a file that prescribes vertical smb')
        # dHdt = dVdt / (pi*(r0*1000.0)**2) # change in ice thickness in m
        smb = dHdt * rhoi / (365.0 * 24 * 60 * 60)  # smb in kg/m^2/s
        indx = np.where(r <= r0)[0]

        # for loop through time
        for t in t_array:
            print((t))
            Ht = h0 + (dHdt * t * dt)  # calculate a new height each time
            print(f'new height will be: {Ht}km')
            sfcMassBal[t, indx] = smb  # assign sfcMassBal
            smbfile.variables['sfcMassBal'][t, :] = sfcMassBal[t, :]

    elif direction == 'horizontal':
        print('creating a file that prescribes horizontal smb')
        # dRdt = (abs(Rf - r0)) / len(t_array)
        smb = (10000.0) * rhoi / (365.0 * 24.0 * 60.0 * 60.0)
        print('-smb is', -smb)
        # for loop through time
        for t in t_array:
            print((t))
            # Rt = abs(np.sqrt(((dVdt * t * dt) / (pi * h0)) + (r0*1000)**2))
            # calculate a new radius at each time; for constant radius change
            Rt = (r0 - (dRdt * t * dt))
            if (np.isnan(Rt) or Rt < 0):
                Rt = 0.0
                smbfile.variables['sfcMassBal'][t, :] = 0.0
                print(f'new radius is: {Rt}km')
            else:
                indx = np.where(r >= Rt / 1000)[0]
                print('idx', indx)
                smbfile.variables['sfcMassBal'][t, indx] = -smb
                print(f'new radius is: {Rt / 1000}km')
            smbfile.variables['sfcMassBal'][t, :] = sfcMassBal[t, :]

    elif direction == 'dome-halfar':
        for t in t_array:
            Rt = (((dVdt * dt * t) * 3 / (2 * pi) +
                   (r0 * 1000) ** 3)) ** (1.0 / 3)
            print(f'new radius is: {Rt / 1000}km')
            # smb = (dVdt * t * dt) / (2 * pi * (Rt ** 2)) \
            # * rhoi / (365.0 * 24.0 * 60.0 * 60.0)i
            indx = np.where(r > r0)[0]
            smbfile.variables['sfcMassBal'][t, :] = smb
            print(f'smb is: {smb}')
        print('ice thickness will change both horizontally and vertically')

    smbfile.close()


def _build_mapping_files(self, config, logger, mali_mesh_file):
    """
    Build a mapping file if it does not exist.
    Mapping file is then used to remap the ismip6 source file in polarstero
    coordinate to unstructured mali mesh

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for a ismip6 forcing test case

    cores : int
        the number of cores for the ESMF_RegridWeightGen

    logger : logging.Logger
        A logger for output from the step

    mali_mesh_file : str, optional
        The MALI mesh file if mapping file does not exist

    method_remap : str, optional
        Remapping method used in building a mapping file
    """

    section = config['slm']
    slm_res = section.get['slm_res']
    method_mali_to_slm = section.get['mapping_method_mali_to_slm']
    method_slm_to_mali = section.get['mapping_method_slm_to_mali']

    section = config['circ_icesheet']
    dc = section.get['dc']

    mali_scripfile = 'mali_scrip.nc'
    slm_scripfile = 'slm_scrip.nc'

    # create slm scripfile
    logger.info(f'creating scripfile for the SH-degree {slm_res}'
                f'resolution SLM grid')
    args = ['ncremap',
            '-g', slm_scripfile,
            '-G',
            f'latlon={slm_res},{int(2*slm_res)}#lat_typ=gss#lat_drc=n2s']

    check_call(args, logger=self.logger)

    # create scrip files for source and destination grids
    logger.info('creating scrip file for the mali mesh')
    mali_mesh_copy = f'{mali_mesh_file}_copy'
    shutil.copy(mali_mesh_file, mali_mesh_copy)

    args = ['set_lat_lon_fields_in_planar_grid.py',
            '--file', mali_mesh_copy,
            '--proj', 'ais-bedmap2-sphere']

    check_call(args, logger=self.logger)

    scrip_from_mpas(mali_mesh_copy, mali_scripfile)

    # create a mapping file using ESMF weight gen
    logger.info('Creating a mapping file...')

    mapping_file_mali_to_slm = f'mapping_file_mali{int(dc/1000)}_to' \
                               f'_slm{slm_res}_{method_mali_to_slm}.nc'
    mapping_file_slm_to_mali = f'mapping_file_slm{slm_res}_to' \
                               f'_mali{int(dc/1000)}_{method_slm_to_mali}.nc'

    parallel_executable = config.get("parallel", "parallel_executable")
    # split the parallel executable into constituents in case it includes flags
    args = parallel_executable.split(' ')
    args.extend(['ESMF_RegridWeightGen',
                 '-s', mali_scripfile,
                 '-d', slm_scripfile,
                 '-w', mapping_file_mali_to_slm,
                 '-m', method_mali_to_slm,
                 '-i', '-64bit_offset', '--netcdf4',
                 '--src_regional'])

    check_call(args, logger)

    args.extend(['ESMF_RegridWeightGen',
                 '-s', slm_scripfile,
                 '-d', mali_scripfile,
                 '-w', mapping_file_slm_to_mali,
                 '-m', method_slm_to_mali,
                 '-i', '-64bit_offset', '--netcdf4',
                 '--dst_regional'])

    check_call(args, logger)

    # remove the temporary scripfiles once the mapping file is generated
    logger.info('Removing the temporary mesh and scripfiles...')
    os.remove(slm_scripfile)
    os.remove(mali_scripfile)
    os.remove(mali_mesh_copy)

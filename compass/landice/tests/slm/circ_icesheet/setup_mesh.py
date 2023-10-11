import os
import shutil

import netCDF4
import numpy as np
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.planar_hex import make_planar_hex_mesh
from mpas_tools.scrip.from_mpas import scrip_from_mpas
from mpas_tools.translate import center, translate
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

    def __init__(self, test_case, name='setup_mesh'):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name=name)

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
        # call the mesh creation function
        dsMesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc,
                                      nonperiodic_x=True,
                                      nonperiodic_y=True)
        dsMesh = cull(dsMesh, logger=logger)
        dsMesh = convert(dsMesh, logger=logger)
        # translating the mesh center to x=0 & y=0
        center(dsMesh)
        # shift the center to a quarter or radius
        r0 = section.getfloat('r0')
        shift = r0
        print(f'shifting the center by {shift} meters')
        translate(dsMesh, shift, shift)
        # write to a file
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
        file to setup circular icesheet
    """
    section = config['circ_icesheet']
    ice_type = section.get('ice_type')
    set_topo_elev = section.getboolean('set_topo_elev')
    topo_elev = section.get('topo_elev')
    r0 = section.getfloat('r0')
    h0 = section.getfloat('h0')
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

    # Assign variable values for the circular ice sheet
    # Set default value for non-circular cells
    thickness[:] = 0.0
    # Calculate the dome thickness for cells within the desired radius
    # (thickness will be NaN otherwise)
    thickness_field = thickness[0, :]

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
        file to setup circular icesheet
    """
    section = config['circ_icesheet']
    r0 = section.getfloat('r0')
    h0 = section.getfloat('h0')

    section = config['smb_forcing']
    direction = section.get('direction')
    start_year = int(section.get('start_year'))
    end_year = int(section.get('end_year'))
    dt_year = int(section.get('dt_year'))
    dHdt = section.getfloat('dHdt')
    dRdt = section.getfloat('dRdt')

    # other constants used in the fuction
    # pi = 3.1415926535898  # pi value used in the SLM
    rhoi = 910.0  # ice density

    # Open the file, get needed dimensions
    shutil.copy(mali_mesh_file, filename)
    smbfile = NetCDFFile(filename, 'r+')

    # Get variables
    xCell = smbfile.variables['xCell']
    yCell = smbfile.variables['yCell']
    nCells = smbfile.dimensions['nCells'].size

    # time interval for the forcing
    t_array = np.arange(start_year, end_year, dt_year)
    nt = len(t_array)

    # initialize SMB value everywhere in the mesh
    sfcMassBal = np.zeros((nt, nCells))

    # Find center of domain
    x0 = xCell[:].min() + 0.5 * (xCell[:].max() - xCell[:].min())
    y0 = yCell[:].min() + 0.5 * (yCell[:].max() - yCell[:].min())
    # calculate the radius of circ ice sheet
    r = ((xCell[:] - x0) ** 2 + (yCell[:] - y0) ** 2) ** 0.5

    if direction == 'vertical':
        logger.info('creating a file that prescribes vertical smb')
        smb = dHdt * rhoi / (365.0 * 24 * 60 * 60)
        logger.info(f'prescribed smb is {smb} kg/m2/s')
        indx = np.where(r <= r0)[0]
        # for loop through time
        for t in range(len(t_array)):
            Ht = h0 + (dHdt * (t + 1))  # calculate a new height each time
            logger.info(f'At time {start_year + dt_year}, \
                        new height will be {Ht} km')
            sfcMassBal[t, indx] = smb  # assign sfcMassBal
        smbfile.variables['sfcMassBal'][:, :] = sfcMassBal[:, :]

    elif direction == 'horizontal':
        logger.info('creating a file that prescribes horizontal smb')
        smb = -1 * ((10000.0) * rhoi / (365.0 * 24.0 * 60.0 * 60.0))
        print(f'prescribed smb is {smb} kg/m2/s')
        # for loop through time
        for t in range(len(t_array)):
            # calculate a new radius at each time
            Rt = r0 + (dRdt * (t + 1))
            if (Rt < 0):
                Rt = 0.0
                sfcMassBal[t, :] = 0.0
                print(f'At time {start_year+dt_year*t}, \
                      new radius will be {Rt} km')
            else:
                indx = np.where(r >= Rt)[0]
                sfcMassBal[t, indx] = smb
                print(f'At time {start_year+dt_year*t}, \
                      new radius will be {Rt/1000} km')
        smbfile.variables['sfcMassBal'][:, :] = sfcMassBal[:, :]

    # add xtime variable
    smbfile.createDimension('StrLen', 64)
    xtime = smbfile.createVariable('xtime', 'S1', ('Time', 'StrLen'))

    # initialize SMB value everywhere in the mesh
    xtime_str = []
    for t_index in range(len(t_array)):
        yr = start_year + (t_index * dt_year)
        xtime_str = f'{int(yr)}-01-01_00:00:00'.ljust(64)
        xtime_char = netCDF4.stringtochar(np.array([xtime_str], 'S64'),
                                          encoding='utf-8')
        xtime[t_index, :] = xtime_char

    smbfile.close()


def _build_mapping_files(config, logger, mali_mesh_file):
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
    slm_res = section.get('slm_res')
    method_mali_to_slm = section.get('mapping_method_mali_to_slm')
    method_slm_to_mali = section.get('mapping_method_slm_to_mali')

    section = config['circ_icesheet']
    dc = section.getfloat('dc')

    mali_scripfile = f'mali{int(dc/1000)}km_scripfile.nc'
    slm_scripfile = f'slm{slm_res}_scripfile.nc'

    # create slm scripfile
    logger.info(f'creating scripfile for the SH-degree {slm_res} '
                f'resolution SLM grid')
    args = ['ncremap',
            '-g', slm_scripfile,
            '-G',
            f'latlon={slm_res},{2*int(slm_res)}#lat_typ=gss#lat_drc=n2s']

    check_call(args, logger=logger)

    # adjust the lat-lon values
    args = ['set_lat_lon_fields_in_planar_grid.py',
            '--file', mali_mesh_file,
            '--proj', 'ais-bedmap2-sphere']

    check_call(args, logger=logger)

    # create scrip files for source and destination grids
    logger.info('creating scrip file for the mali mesh')
    scrip_from_mpas(mali_mesh_file, mali_scripfile)

    # create a mapping file using ESMF weight gen
    logger.info('Creating a mapping file...')

    parallel_executable = config.get("parallel", "parallel_executable")
    # split the parallel executable into constituents in case it includes flags
    args = parallel_executable.split(' ')
    args.extend(['ESMF_RegridWeightGen',
                 '-s', mali_scripfile,
                 '-d', slm_scripfile,
                 '-w', 'mapping_file_mali_to_slm.nc',
                 '-m', method_mali_to_slm,
                 '-i', '-64bit_offset', '--netcdf4',
                 '--src_regional'])

    check_call(args, logger)

    args = parallel_executable.split(' ')
    args.extend(['ESMF_RegridWeightGen',
                 '-s', slm_scripfile,
                 '-d', mali_scripfile,
                 '-w', 'mapping_file_slm_to_mali.nc',
                 '-m', method_slm_to_mali,
                 '-i', '-64bit_offset', '--netcdf4',
                 '--dst_regional'])

    check_call(args, logger)

    # remove the temporary scripfiles once the mapping file is generated
    logger.info('Removing the temporary mesh and scripfiles...')
    os.remove(slm_scripfile)
    os.remove(mali_scripfile)

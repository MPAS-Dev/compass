import subprocess
from datetime import datetime
import numpy
import xarray
import os
import shutil

from mpas_tools.io import write_netcdf


def get_e3sm_mesh_names(config, levels):
    """
    Get short and long E3SM mesh name from config options and the given number
    of vertical levels (typically taken from an initial condition or restart
    file).

    Parameters
    ----------
    config : configparser.ConfigParser
        Configuration options for this testcase, a combination of the defaults
        for the machine, core and configuration

    levels : int
        The number of vertical levels

    Returns
    -------
    short_mesh_name : str
        The short E3SM name of the ocean and sea-ice mesh

    long_mesh_name : str
        The long E3SM name of the ocean and sea-ice mesh
    """

    mesh_prefix = config.get('global_ocean', 'prefix')
    min_res = config.get('global_ocean', 'min_res')
    max_res = config.get('global_ocean', 'max_res')
    config.set('global_ocean', 'levels', '{}'.format(levels))
    e3sm_version = config.get('global_ocean', 'e3sm_version')
    mesh_revision = config.get('global_ocean', 'mesh_revision')

    if min_res == max_res:
        res = min_res
    else:
        res = '{}to{}'.format(min_res, max_res)

    short_mesh_name = '{}{}E{}r{}'.format(mesh_prefix, res, e3sm_version,
                                          mesh_revision)
    long_mesh_name = '{}{}kmL{}E3SMv{}r{}'.format(mesh_prefix, res, levels,
                                                  e3sm_version, mesh_revision)

    return short_mesh_name, long_mesh_name


def add_mesh_and_init_metadata(output_filenames, config, init_filename):
    """
    Add MPAS mesh and initial condition metadata to NetCDF outputs of the given
    step

    Parameters
    ----------
    output_filenames : list
        A list of output files.

    config : configparser.ConfigParser
        Configuration options for this testcase, a combination of the defaults
        for the machine, core and configuration

    init_filename : str
        The name of an initial condition file to get the number of vertical
        levels and maximum depth from
    """

    if config.getboolean('global_ocean', 'add_metadata'):
        with xarray.open_dataset(init_filename) as dsInit:
            metadata = _get_metadata(dsInit, config)

        for filename in output_filenames:
            if filename.endswith('.nc'):
                args = ['ncra']
                for key, value in metadata.items():
                    args.extend(['--glb_att_add', '{}={}'.format(key, value)])
                name, ext = os.path.splitext(filename)
                new_filename = '{}_with_metadata{}'.format(name, ext)
                args.extend([filename, new_filename])
                subprocess.check_call(args)
                shutil.move(new_filename, filename)


def _get_metadata(dsInit, config):
    """ add metadata to a given dataset """

    author = config.get('global_ocean', 'author')
    if author == 'autodetect':
        author = subprocess.check_output(
            ['git', 'config', 'user.name']).decode("utf-8").strip()
        config.set('global_ocean', 'author', author)

    email = config.get('global_ocean', 'email')
    if email == 'autodetect':
        email = subprocess.check_output(
            ['git', 'config', 'user.email']).decode("utf-8").strip()
        config.set('global_ocean', 'email', email)

    creation_date = config.get('global_ocean', 'creation_date')
    if creation_date == 'autodetect':
        now = datetime.now()
        creation_date = now.strftime("%y%m%d")
        config.set('global_ocean', 'creation_date', creation_date)

    max_depth = dsInit.bottomDepth.max().values
    # round to the nearest 0.1 m
    max_depth = numpy.round(max_depth, 1)
    config.set('global_ocean', 'max_depth', '{}'.format(max_depth))

    mesh_prefix = config.get('global_ocean', 'prefix')
    min_res = config.get('global_ocean', 'min_res')
    max_res = config.get('global_ocean', 'max_res')
    levels = dsInit.sizes['nVertLevels']
    config.set('global_ocean', 'levels', '{}'.format(levels))
    e3sm_version = config.get('global_ocean', 'e3sm_version')
    mesh_revision = config.get('global_ocean', 'mesh_revision')
    pull_request = config.get('global_ocean', 'pull_request')

    short_name, long_name = get_e3sm_mesh_names(config, levels)

    descriptions = dict()

    for prefix in ['mesh', 'init', 'bathy', 'bgc', 'wisc']:
        option = '{}_description'.format(prefix)
        if config.has_option('global_ocean', option):
            description = config.get('global_ocean', option)
            description = ' '.join(
                [line.strip() for line in description.split('\n')])
            descriptions[prefix] = description

    prefix = 'MPAS_Mesh_{}'.format(mesh_prefix)

    metadata = {'MPAS_Mesh_Short_Name': short_name,
                'MPAS_Mesh_Long_Name': long_name,
                'MPAS_Mesh_Prefix': mesh_prefix,
                'MPAS_Mesh_E3SM_Version': e3sm_version,
                'MPAS_Mesh_Pull_Request': pull_request,
                '{}_Revision'.format(prefix): mesh_revision,
                '{}_Version_Author'.format(prefix): author,
                '{}_Version_Author_E-mail'.format(prefix): email,
                '{}_Version_Creation_Date'.format(prefix): creation_date,
                '{}_Minimum_Resolution_km'.format(prefix): min_res,
                '{}_Maximum_Resolution_km'.format(prefix): max_res,
                '{}_Maximum_Depth_m'.format(prefix): '{}'.format(max_depth),
                '{}_Number_of_Levels'.format(prefix): '{}'.format(levels),
                'MPAS_Mesh_Description': descriptions['mesh'],
                'MPAS_Mesh_Bathymetry': descriptions['bathy'],
                'MPAS_Initial_Condition': descriptions['init']}

    if 'wisc' in descriptions:
        metadata['MPAS_Mesh_Ice_Shelf_Cavities'] = descriptions['wisc']

    if 'bgc' in descriptions:
        metadata['MPAS_Mesh_Biogeochemistry'] = descriptions['bgc']

    packages = {'compass': 'compass', 'JIGSAW': 'jigsaw',
                'JIGSAW-Python': 'jigsawpy', 'MPAS-Tools': 'mpas_tools',
                'NCO': 'nco', 'ESMF': 'esmf',
                'geometric_features': 'geometric_features',
                'Metis': 'metis', 'pyremap': 'pyremap'}

    for name in packages:
        package = packages[name]
        metadata['MPAS_Mesh_{}_Version'.format(name)] = \
            _get_conda_package_version(package)

    return metadata


def _get_conda_package_version(package):
    conda = subprocess.check_output(['conda', 'list', package]).decode("utf-8")
    lines = conda.split('\n')
    for line in lines:
        parts = line.split()
        if parts[0] == package:
            return parts[1]

    raise ValueError('Package {} not found in the conda environment'.format(
        package))

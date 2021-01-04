import os


def get_init_sudbdir(mesh_name, initial_condition, with_bgc):
    """
    Get the subdirectory specific to the initial condition that all testscases
    (other than mesh) are under

    Parameters
    ----------
    mesh_name : str
        The name of the mesh

    initial_condition : {'PHC', 'EN4_1900'}
        The initial condition to build

    with_bgc : bool
        Whether to include BGC fields in the initial condition

    Returns
    -------
    init_subdir : str
        The subdirectory

    """
    if with_bgc:
        init_subdir = '{}/{}_BGC'.format(mesh_name, initial_condition)
    else:
        init_subdir = '{}/{}'.format(mesh_name, initial_condition)
    return init_subdir


def get_mesh_relative_path(step):
    """
    get the relative path to the subdirectory with the mesh name in the
    global_ocean configuration

    Parameters
    ----------
    step : dict
        The dictionary describing the step

    Returns
    -------
    path : str
        The relative path to the subdirectory with the mesh name
    """

    # build a path that has a '..' for each directory in subdir
    subdir = os.path.join(step['testcase_subdir'], step['subdir'])
    # skip the mesh directory, which is common to all tests
    subdirs = subdir.split('/')[1:]
    path = '/'.join(['..' for _ in subdirs])
    return path


def get_initial_condition_relative_path(step):
    """
    get the relative path to the subdirectory with the initial condition name
    (``PHC``, ``EN4_1900``, etc.) in the global_ocean configuration

    Parameters
    ----------
    step : dict
        The dictionary describing the step

    Returns
    -------
    path : str
        The relative path to the subdirectory with the initial condition name
    """

    # build a path that has a '..' for each directory in subdir
    subdir = os.path.join(step['testcase_subdir'], step['subdir'])
    # skip the mesh and init directories, which is common to all but the mesh
    # tests
    subdirs = subdir.split('/')[2:]
    path = '/'.join(['..' for _ in subdirs])
    return path

from importlib.util import find_spec


def get_model_configuration_name(config):
    """
    Get the configured model configuration name.

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for a test case

    Returns
    -------
    str
        The selected model configuration name
    """
    section = 'ensemble_generator'
    option = 'model_configuration'

    if not config.has_section(section):
        raise ValueError(
            f"Missing required config section '{section}' for ensemble "
            "generator configuration selection.")

    if not config.has_option(section, option):
        raise ValueError(
            f"Missing required config option '{option}' in section "
            f"'{section}'.")

    configuration = config.get(section, option).strip()
    if configuration == '':
        raise ValueError('model_configuration cannot be empty.')

    return configuration


def get_spinup_configuration_package(config):
    """
    Get the package containing spinup ensemble resources.

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for a test case

    Returns
    -------
    str
        Package path for spinup resources
    """
    configuration = get_model_configuration_name(config)
    return ('compass.landice.tests.ensemble_generator.configurations.'
            f'{configuration}.spinup')


def get_branch_configuration_package(config):
    """
    Get the package containing branch ensemble resources.

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for a test case

    Returns
    -------
    str
        Package path for branch resources
    """
    configuration = get_model_configuration_name(config)
    return ('compass.landice.tests.ensemble_generator.configurations.'
            f'{configuration}.branch')


def add_configuration_file(config, package, filename):
    """
    Add a configuration file from a selected configuration package.

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for a test case

    package : str
        The package containing the requested configuration file

    filename : str
        The configuration filename to add from the package
    """
    if find_spec(package) is None:
        raise ValueError(
            f"Model configuration package '{package}' was not found.")

    config.add_from_package(package, filename, exception=True)

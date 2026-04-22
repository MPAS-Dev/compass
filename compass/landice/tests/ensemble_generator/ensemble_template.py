def get_ensemble_template_name(config):
    """
    Get the configured ensemble template name.

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for a test case

    Returns
    -------
    str
        The selected ensemble template name
    """
    section = 'ensemble_generator'
    option = 'ensemble_template'

    if not config.has_section(section):
        raise ValueError(
            f"Missing required config section '{section}' for ensemble "
            "generator configuration selection.")

    if not config.has_option(section, option):
        raise ValueError(
            f"Missing required config option '{option}' in section "
            f"'{section}'.")

    template = config.get(section, option).strip()
    if template == '':
        raise ValueError('ensemble_template cannot be empty.')

    return template


def get_spinup_template_package(config):
    """
    Get the package containing spinup ensemble template resources.

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for a test case

    Returns
    -------
    str
        Package path for spinup resources
    """
    template = get_ensemble_template_name(config)
    return ('compass.landice.tests.ensemble_generator.ensemble_templates.'
            f'{template}.spinup')


def get_branch_template_package(config):
    """
    Get the package containing branch ensemble template resources.

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for a test case

    Returns
    -------
    str
        Package path for branch resources
    """
    template = get_ensemble_template_name(config)
    return ('compass.landice.tests.ensemble_generator.ensemble_templates.'
            f'{template}.branch')

def configure(config, check_model_options):
    """
    A shared function for configuring options for all ismip6 forcing
    test cases

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for a ismip6 forcing test case

    check_model_options : bool
        Whether we check ``model``, ``scenario``, ``period_endyear``

    """

    section = 'ismip6_ais'
    options = ['base_path_ismip6', 'base_path_mali', 'mali_mesh_name',
               'mali_mesh_file']
    if check_model_options:
        options = options + ['model', 'scenario', 'period_endyear']

    for option in options:
        value = config.get(section=section, option=option)
        if value == "NotAvailable":
            raise ValueError(f"You need to supply a user config file, which "
                             f"should contain the {section} "
                             f"section with the {option} option")

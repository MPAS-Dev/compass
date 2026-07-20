def configure(config):
    """
    A shared function for configuring options for all ismip7 forcing
    test cases

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for an ismip7 forcing test case
    """

    section = "ismip7"
    options = ["ice_sheet", "base_path_ismip7", "base_path_mali",
               "mali_mesh_name", "mali_mesh_file", "output_base_path",
               "model", "scenario"]

    for option in options:
        value = config.get(section=section, option=option)
        if value == "NotAvailable":
            raise ValueError(f"You need to supply a user config file, which "
                             f"should contain the {section} "
                             f"section with the {option} option")

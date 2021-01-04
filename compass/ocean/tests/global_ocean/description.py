def get_description(mesh_name, initial_condition, with_bgc, time_integrator,
                    description):
    """
    Get a complete description of a global ocean testcase

    Parameters
    ----------
     mesh_name : str
        The name of the mesh

    initial_condition : {'PHC', 'EN4_1900'}
        The initial condition to build

    with_bgc : bool
        Whether to include BGC variables in the initial condition

    time_integrator : {'split_explicit', 'RK4'}
        The time integrator to use for the run

    description : str
        The part of the description specific to this testcase

    Returns
    -------
    description : str
        The full description

    """

    if with_bgc:
        initial_condition = '{} with BGC'.format(initial_condition)
    description = 'global ocean {} - {} - {} {}'.format(
        mesh_name, initial_condition, time_integrator, description)

    return description

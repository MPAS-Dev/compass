from compass.ocean.tests.global_ocean.metadata import \
    get_author_and_email_from_git


def configure_global_ocean(test_case, mesh, init=None):
    """
    Modify the configuration options for this test case

    Parameters
    ----------
    test_case : compass.TestCase
        The test case to configure

    mesh : compass.ocean.tests.global_ocean.mesh.Mesh
        The test case that produces the mesh for this run

    init : compass.ocean.tests.global_ocean.init.Init, optional
        The test case that produces the initial condition for this run
    """
    config = test_case.config
    mesh_step = mesh.mesh_step
    config.add_from_package(mesh_step.package, mesh_step.mesh_config_filename,
                            exception=True)

    if mesh.with_ice_shelf_cavities:
        config.set('global_ocean', 'prefix', '{}wISC'.format(
            config.get('global_ocean', 'prefix')))

    # add a description of the initial condition
    if init is not None:
        initial_condition = init.initial_condition
        descriptions = {'PHC': 'Polar science center Hydrographic '
                               'Climatology (PHC)',
                        'EN4_1900':
                            "Met Office Hadley Centre's EN4 dataset from 1900"}
        config.set('global_ocean', 'init_description',
                   descriptions[initial_condition])

    # a description of the bathymetry
    config.set('global_ocean', 'bathy_description',
               'Bathymetry is from GEBCO 2019, combined with BedMachine '
               'Antarctica around Antarctica.')

    if init is not None and init.with_bgc:
        # todo: this needs to be filled in!
        config.set('global_ocean', 'bgc_description',
                   '<<<Missing>>>')

    if mesh.with_ice_shelf_cavities:
        config.set('global_ocean', 'wisc_description',
                   'Includes cavities under the ice shelves around Antarctica')

    get_author_and_email_from_git(config)

from compass.ocean.tests.global_ocean.metadata import \
    get_author_and_email_from_git


def configure_tides(test_case, mesh):
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

    get_author_and_email_from_git(config)

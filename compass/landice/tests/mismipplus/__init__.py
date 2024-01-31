import os

from compass.landice.tests.mismipplus.smoke_test import SmokeTest
from compass.landice.tests.mismipplus.spin_up import SpinUp
from compass.testgroup import TestGroup


class MISMIPplus(TestGroup):
    """
    A test group for MISMIP+ test cases.
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.landice.Landice
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='mismipplus')

        self.add_test_case(SmokeTest(test_group=self))

        self.add_test_case(SpinUp(test_group=self))


def configure(step, config, resolution):
    """
    This function will create a common and consitent directory structure
    for all test cases within this test group. The direcotry structure will
    based on the requested resolution, enabling the user to run the testcases
    multiple times at different resolutions. ``compass setup`` will need to be
    run for each desired resolution.

    Note: this function does not return anything. Instead it modifies the
          ``step`` and ``config``, which are passed to it.

    Parameters
    ----------
    step : compass.step

    config : compass.config.CompassConfigParser
        Configuration options for the test case, including custom configuration
        options passed via the ``-f`` flag at the time of ``compass setup``

    resolution : float
        The requested (nominal) resolution. Directory strcture will include
        this value in it. See comment in configuration file for more details
        about what "nominal" resolution means.
    """

    # format resolution for creating subdirectory structure
    resolution_key = f'{resolution:4.0f}m'
    step.subdir = f'{resolution_key}/{step.name}'

    # set the path attribute, based on the subdir attribute set above.
    step.path = os.path.join(step.mpas_core.name,
                             step.test_group.name,
                             step.test_case.subdir,
                             step.subdir)

    # NOTE: we do not set the `step.work_dir` attribute, since it will be set
    # by `compass setup`` by joining the work dir provided through the command
    # line interface and the `step.path` set above.

    # store the resolution (at the time of `compass setup`) as an attribute.
    # This is needed to prevent the changing of resolution between
    # `compass setup` and `compas run`, which could result in a mesh having
    # a different resolution than the direcotry it sits in.
    step.resolution = resolution

    comment = ("Nominal cell spacing (m) at the time of `compass setup`. \n"
               "NOTE: the resolution can not be changed after this point. \n"
               "To run another resolution, rerun `compass setup` with the \n"
               "desired nominal resolution. Resolution is \"nominal\" since \n"
               "the true cell spacing will be determined such that the cell \n"
               "center to cell center length of the entire domain in the y \n"
               "direction is exactly the required y domain length (80 km).")

    config.set('mesh', 'resolution', str(resolution), comment=comment)

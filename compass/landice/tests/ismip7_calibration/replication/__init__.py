from compass.landice.tests.ismip7_calibration.configure import check_options
from compass.landice.tests.ismip7_calibration.replication.replicate import (
    Replicate,
    check_published,
)
from compass.testcase import TestCase
from compass.validate import compare_variables


class Replication(TestCase):
    """
    A test case that reproduces the published ISMIP7 quadratic calibration on
    the ISMIP 8 km grid, through compass's own code path.

    The protocol reports, for the quadratic local parameterization on the
    ISMIP 8 km grid (Sect. 4.3.1 and Fig. 5), 5th / 50th / 95th percentiles
    of K = 4.75e-5 / 8.5e-5 / 1.375e-4.  Reproducing those with compass's
    area-weighted, mesh-agnostic terms driving the vendored objective
    function validates the whole of the parameter-selection stage before any
    MALI run exists, and is the regression test that the unstructured
    generalization must not break.

    This test case deliberately uses the **published inputs**, including the
    spatially varying 8 km salinity fields.  It replicates a published
    calculation, so it must not be changed to match the constant-salinity
    choice the MALI calibration makes; it validates the parameter-selection
    machinery, not MALI's physics.
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.ismip7_calibration.Ismip7Calibration
            The test group that this test case belongs to
        """  # noqa: E501
        name = 'replication'
        super().__init__(test_group=test_group, name=name, subdir=name)

        self.add_step(Replicate(test_case=self))

    def configure(self):
        """
        Check that the ISMIP7 dataset path has been supplied
        """
        check_options(self.config, ['base_path_ismip7'])

    def validate(self):
        """
        Check the percentiles against the published values, and against a
        baseline if one was provided
        """
        filename = 'replicate/replication_8km.nc'
        check_published(filename, self.logger)
        compare_variables(test_case=self,
                          variables=['p5', 'median', 'p95', 'mode',
                                     'min_p1'],
                          filename1=filename)

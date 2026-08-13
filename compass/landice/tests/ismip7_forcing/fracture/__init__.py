from compass.landice.tests.ismip7_forcing.configure import (
    configure as configure_testgroup,
)
from compass.landice.tests.ismip7_forcing.fracture.process_excess_melt import (
    ProcessExcessMelt,
)
from compass.landice.tests.ismip7_forcing.fracture.process_lake_properties import (  # noqa: E501
    ProcessLakeProperties,
)
from compass.landice.tests.ismip7_forcing.fracture.process_shelf_collapse import (  # noqa: E501
    ProcessShelfCollapse,
)
from compass.testcase import TestCase


class Fracture(TestCase):
    """
    A test case for processing ISMIP7 fracture forcing data.
    Implements the surface-melt-driven ice shelf collapse pathways:

    * Path A (``process_excess_melt``): excess meltwater after firn air
      content depletion.
    * Path B (``process_lake_properties``): supraglacial lake mean depth
      and area fraction from Grau et al. (2025).
    * Path C (``process_shelf_collapse``): the ice shelf collapse mask, in
      which a floating grid cell is flagged as collapsed when excess
      meltwater exceeds 72.5 mm/yr for 10 consecutive years.
    """

    def __init__(self, test_group):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.ismip7_forcing.Ismip7Forcing
            The test group that this test case belongs to
        """
        name = "fracture"
        subdir = name
        super().__init__(test_group=test_group, name=name, subdir=subdir)

        self.add_step(ProcessExcessMelt(test_case=self))
        self.add_step(ProcessLakeProperties(test_case=self))
        self.add_step(ProcessShelfCollapse(test_case=self))

    def configure(self):
        """
        Configures test case
        """
        configure_testgroup(config=self.config)

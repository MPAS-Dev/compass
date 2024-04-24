from compass.landice.tests.ismip6_GrIS_forcing.create_mapping_files import (
    CreateMappingFiles,
)
from compass.landice.tests.ismip6_GrIS_forcing.file_finders import (
    atmosphereFileFinder,
    oceanFileFinder,
)
from compass.landice.tests.ismip6_GrIS_forcing.process_forcing import (
    ProcessForcing,
)
from compass.testcase import TestCase


class ForcingGen(TestCase):
    """
    A TestCase for remapping (i.e. interpolating) ISMIP6 GrIS forcing files
    onto a MALI mesh

    Attributes
    ----------

    mali_mesh_scrip : str
        filepath to the scrip file describing the MALI mesh

    ismip6_GrIS_scrip : str
        filepath to the scrip file describing the grid the ISMIP6 GrIS
        forcing data is on. (Note: All forcing files are on the same grid,
        so only need one scrip file for all forcing files)

    remapping_weights : str
        filepath to the `ESMF_RegirdWeightGen` generated mapping file

    experiments : dict

    """

    def __init__(self, test_group):
        """
        Parameters
        ----------
        test_group : compass.landice.tests...
            The test group that this test case belongs to
        """
        name = "forcing_gen"
        super().__init__(test_group=test_group, name=name)

        # filenames for remapping files, stored at the testcase level so they
        # will be accesible to all steps in the testcase
        self.mali_mesh_scrip = "mali_mesh.scrip.nc"
        self.ismip6_GrIS_scrip = "ismip6_GrIS.scrip.nc"
        self.remapping_weights = "ismip6_2_mali.weights.nc"
        # place holder for file finders that will be initialized in `configure`
        self.__atmFF = None
        self.__ocnFF = None

        # precusssory step that builds scrip file for mali mesh,
        # and generates a common weights file to be used in remapping
        self.add_step(CreateMappingFiles(test_case=self))

        # step that deals with racmo, do all I need to do is remap and average?

        # add steps that re-maps and processes downscaled GCM data for each
        # experiment.
        self.add_step(ProcessForcing(test_case=self))

    def configure(self):

        config = self.config
        # add ouputdir path to the remapping files

        # get the list of requested experiments
        expr2run = config.getlist("ISMIP6_GrIS_Forcing", "experiments")
        # get the dictionary of experiments, as defined in the yaml file
        all_exprs = self.test_group.experiments
        # get subset of dictionaries, based on requested expriments
        self.experiments = {e: all_exprs[e] for e in expr2run}

        archive_fp = config.get("ISMIP6_GrIS_Forcing", "archive_fp")

        # initalize the oceanFileFinder
        self.__ocnFF = oceanFileFinder(archive_fp)
        self.__atmFF = atmosphereFileFinder(archive_fp)  # , workdir=workdir)

    def findForcingFiles(self, GCM, scenario, variable):
        """
        """

        if variable in ["basin_runoff", "thermal_forcing"]:
            forcing_fp = self.__ocnFF.get_filename(GCM, scenario, variable)

        if variable in ["aSMB", "aST", "dSMBdz", "dSTdz"]:
            forcing_fp = self.__atmFF.get_filename(GCM, scenario, variable)

        return forcing_fp

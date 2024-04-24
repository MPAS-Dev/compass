import glob
import os
import shutil

import xarray as xr
from mpas_tools.logging import check_call
from mpas_tools.scrip.from_mpas import scrip_from_mpas
from pyproj import Proj
from pyremap.descriptor import ProjectionGridDescriptor

from compass.step import Step


class CreateMappingFiles(Step):
    """
    """

    def __init__(self, test_case):
        """
        """
        name = "create_mapping_files"
        subdir = "mapping_files"

        super().__init__(test_case=test_case, name=name, subdir=subdir,
                         cpus_per_task=128, min_cpus_per_task=1)

        # initalize the attributes with empty values for now, attributes will
        # be propely in `setup` method so user specified config options can
        # be parsed
        self.mali_mesh = None

    def setup(self):
        """
        """

        # parse user specified parameters from the config
        config = self.config
        section = config["ISMIP6_GrIS_Forcing"]

        # read the mesh path/name from the config file
        mali_mesh = section.get("MALI_mesh_fp")

        # make sure the mesh path/name is valid and accessible
        if not os.path.exists(mali_mesh):
            raise FileNotFoundError(f"{mali_mesh}")

        # add mesh name as an attribute and an input file to the step
        self.mali_mesh = mali_mesh
        self.add_input_file(filename=self.mali_mesh)

        # loop over mapping file `test_case` attributes and add the current
        # steps `work_dir` to the absolute path. This will ensure other steps
        # will be able to access the mapping files created here
        for attr in ["mali_mesh_scrip",
                     "ismip6_GrIS_scrip",
                     "remapping_weights"]:
            # get the attribute values
            file_name = getattr(self.test_case, attr)
            # combine filename w/ the testcases work dir
            full_path = os.path.join(self.work_dir, file_name)
            # update the attribute value to include the full path
            setattr(self.test_case, attr, full_path)

        # Add the testcase attributes as output files; since they'll be
        # generated as part of this step
        self.add_output_file(filename=self.test_case.mali_mesh_scrip)
        self.add_output_file(filename=self.test_case.ismip6_GrIS_scrip)
        self.add_output_file(filename=self.test_case.remapping_weights)

    def run(self):
        """
        """

        # unpack the filepaths stored as attributes of the test_case
        mali_mesh_scrip = self.test_case.mali_mesh_scrip
        ismip6_GrIS_scrip = self.test_case.ismip6_GrIS_scrip
        remapping_weights = self.test_case.remapping_weights

        # create scrip file describing MALI mesh that forcing variables
        # will be interpolated onto
        scrip_from_mpas(self.mali_mesh, mali_mesh_scrip)

        # get a list of the projection experiments requesed.
        # (i.e. experiments names that use ISMIP6 forcing)
        proj_exprs = list(filter(lambda x: "Exp" in x,
                                 self.test_case.experiments.keys()))

        if proj_exprs:
            # b/c all GrIS forcing files are on the same grid, it doesn't
            # matter what expr we use; so just use the first suitable candiate
            expr = proj_exprs[0]

        expr_params = self.test_case.experiments[expr]

        GCM = expr_params["GCM"]
        scenario = expr_params["Scenario"]
        variable = "thermal_forcing"
        # get the filepath for any given forcing file (since all files are on
        # the same grid), in order to generate a scrip file for the grid
        forcing_fp = self.test_case.findForcingFiles(GCM, scenario, variable)

        # WGS 84 / NSIDC Sea Ice Polar Stereographic North Projection
        epsg_3413 = Proj("EPSG:3413")

        # open up forcing grid; get x and y coordinates as numpy arrays
        with xr.open_dataset(forcing_fp) as ds:
            x = ds.x.values
            y = ds.y.values

        # create pyremap `MeshDescriptor` obj for the ISMIP6 GrIS forcing grid
        forcingDescriptor = ProjectionGridDescriptor.create(
            projection=epsg_3413, x=x, y=y, meshName=forcing_fp)

        # write scrip file describing ISMIP6 GrIS forcing grid
        forcingDescriptor.to_scrip(ismip6_GrIS_scrip)

        # generate mapping weights between ismip6 forcing data and the MALI
        # mesh, using `ESMF_RegridWeightGen`
        args = ['srun', '-n', '128', 'ESMF_RegridWeightGen',
                '-s', ismip6_GrIS_scrip,
                '-d', mali_mesh_scrip,
                '-w', remapping_weights,
                '--method', 'conserve',
                "--netcdf4",
                # "--no_log",
                "--dst_regional", "--src_regional", '--ignore_unmapped']

        check_call(args, logger=self.logger)

        # `ESMF_RegridWeightGen` will generate a log file for each processor,
        # which really clutters up the directory. So, if log files are created
        # move them to their own `logs` directory
        log_files = glob.glob("PET*.RegridWeightGen.Log")

        # check if glob returns an empty list
        if log_files:
            # make the log directory
            os.mkdir("logs/")
            # copy the log files to the log directory; using list comprehension
            _ = [shutil.move(f, "logs/") for f in log_files]

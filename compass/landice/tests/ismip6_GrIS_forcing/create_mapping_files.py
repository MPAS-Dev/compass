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
        self.racmo_grid = None

    def setup(self):
        """
        """

        # parse user specified parameters from the config
        config = self.config
        forcing_section = config["ISMIP6_GrIS_Forcing"]
        smb_ref_section = config["smb_ref_climatology"]

        # read the mesh path/name from the config file
        mali_mesh = forcing_section.get("MALI_mesh_fp")

        # make sure the mesh path/name is valid and accessible
        if not os.path.exists(mali_mesh):
            raise FileNotFoundError(f"{mali_mesh}")

        # add mesh name as an attribute and an input file to the step
        self.mali_mesh = mali_mesh
        self.add_input_file(filename=self.mali_mesh)

        #
        racmo_directory = smb_ref_section.get("racmo_directory")
        # this filename should probably just be hardcoded.....
        racmo_grid_fn = smb_ref_section.get("racmo_grid_fn")
        # combine the directory and filename
        racmo_grid_fp = os.path.join(racmo_directory, racmo_grid_fn)

        # make sure the combined filename exists
        if not os.path.exists(racmo_grid_fp):
            # check if the parent directory exists
            if not os.path.exists(racmo_directory):
                raise FileNotFoundError(f"{racmo_directory} does not exist")
            # the parent directory exists but the forcing file does not
            else:
                raise FileNotFoundError(f"{racmo_grid_fp} does not exist")

        # add the racmo grid as an attribute and as an input file
        self.racmo_grid = racmo_grid_fp
        self.add_input_file(filename=racmo_grid_fp)

        # loop over mapping file `test_case` attributes and add the current
        # steps `work_dir` to the absolute path. This will ensure other steps
        # will be able to access the mapping files created here
        for attr in ["mali_mesh_scrip",
                     "racmo_gis_scrip",
                     "ismip6_gis_scrip",
                     "racmo_2_mali_weights",
                     "ismip6_2_mali_weights"]:
            # get the attribute values
            file_name = getattr(self.test_case, attr)
            # combine filename w/ the testcases work dir
            full_path = os.path.join(self.work_dir, file_name)
            # update the attribute value to include the full path
            setattr(self.test_case, attr, full_path)

        # Add the testcase attributes as output files; since they'll be
        # generated as part of this step
        self.add_output_file(filename=self.test_case.mali_mesh_scrip)
        self.add_output_file(filename=self.test_case.racmo_gis_scrip)
        self.add_output_file(filename=self.test_case.ismip6_gis_scrip)
        self.add_output_file(filename=self.test_case.racmo_2_mali_weights)
        self.add_output_file(filename=self.test_case.ismip6_2_mali_weights)

    def run(self):
        """
        """

        # unpack the filepaths stored as attributes of the test_case
        mali_mesh_scrip = self.test_case.mali_mesh_scrip
        racmo_gis_scrip = self.test_case.racmo_gis_scrip
        ismip6_gis_scrip = self.test_case.ismip6_gis_scrip
        racmo_2_mali_weights = self.test_case.racmo_2_mali_weights
        ismip6_2_mali_weights = self.test_case.ismip6_2_mali_weights

        # create scrip file describing MALI mesh that forcing variables
        # will be interpolated onto
        scrip_from_mpas(self.mali_mesh, mali_mesh_scrip)

        # WGS 84 / NSIDC Sea Ice Polar Stereographic North Projection
        epsg_3413 = Proj("EPSG:3413")

        # make scrip file for racmo grid and mapping weights for racmo->mpas
        self.make_forcing_descriptor_and_weights(self.racmo_grid,
                                                 racmo_gis_scrip,
                                                 mali_mesh_scrip,
                                                 epsg_3413,
                                                 racmo_2_mali_weights)

        # finding the right ismip6 forcing file is complicated,
        # so let's call a helper function to do that.
        ismip6_forcing_fp = self._find_ismip6_forcing_files()

        # make scrip file for ismip6 grid and mapping weights for ismip6->mpas
        self.make_forcing_descriptor_and_weights(ismip6_forcing_fp,
                                                 ismip6_gis_scrip,
                                                 mali_mesh_scrip,
                                                 epsg_3413,
                                                 ismip6_2_mali_weights)

    def make_forcing_descriptor_and_weights(self,
                                            forcing_file_fp,
                                            forcing_scrip_fp,
                                            mali_mesh_scrip_fp,
                                            forcing_proj,
                                            weights_file_fp):

        # open up forcing grid; get x and y coordinates as numpy arrays
        with xr.open_dataset(forcing_file_fp) as ds:
            x = ds.x.values
            y = ds.y.values

        # create pyremap `MeshDescriptor` obj for the forcing grid
        forcingDescriptor = ProjectionGridDescriptor.create(
            projection=forcing_proj, x=x, y=y, mesh_name=forcing_file_fp)

        # write scrip file describing forcing grid
        forcingDescriptor.to_scrip(forcing_scrip_fp)

        # generate mapping weights between forcing data and the MALI mesh
        args = ['srun', '-n', '128', 'ESMF_RegridWeightGen',
                '-s', forcing_scrip_fp,
                '-d', mali_mesh_scrip_fp,
                '-w', weights_file_fp,
                '--method', 'bilinear',
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
            # split the output weights filename so that only the basename w/
            # no extension is left, in order to clean up the log files
            base_name = os.path.splitext(os.path.basename(weights_file_fp))[0]
            dir_name = f"{base_name}.Logs/"

            # check if there is already a Log files directory, if so wipe it
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)

            # make the log directory
            os.mkdir(dir_name)

            # copy the log files to the log directory; using list comprehension
            _ = [shutil.move(f, dir_name) for f in log_files]

    def _find_ismip6_forcing_files(self):
        """
        """
        # get a list of all projection experiments requesed.
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

        return forcing_fp

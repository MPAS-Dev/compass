import os
import shutil

from compass.landice.ismip7.ice_sheet_params import get_params
from compass.landice.ismip7.mapping import build_mapping_file
from compass.step import Step


class BuildMappingFile(Step):
    """
    A step for building the ESMF mapping file (regridding weights) from
    the ISMIP7 atmosphere grid to the MALI mesh. This step runs
    ESMF_RegridWeightGen with the full esmf_ntasks allocation, so it can
    be run on a separate node allocation from the processing steps (which
    run on a single node).

    The mapping file can be reused across scenarios, since it depends only on
    the ice sheet, mesh name, and remapping method. If mapping_files_path is
    provided in the config, this step will symlink any existing weight file
    from that directory and skip the build.
    """

    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.landice.tests.ismip7_forcing.atmosphere.Atmosphere
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name="build_mapping_file")

    def setup(self):
        """
        Set up this step of the test case
        """
        config = self.config
        section = config["ismip7"]
        base_path_mali = section.get("base_path_mali")
        mali_mesh_file = section.get("mali_mesh_file")

        self.add_input_file(filename=mali_mesh_file,
                            target=os.path.join(base_path_mali,
                                                mali_mesh_file))

        # Request the full esmf_ntasks allocation for this step
        self.ntasks = section.getint("esmf_ntasks")
        self.min_tasks = self.ntasks

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        config = self.config
        params = get_params(config)

        section = config["ismip7"]
        base_path_ismip7 = section.get("base_path_ismip7")
        mali_mesh_name = section.get("mali_mesh_name")
        mali_mesh_file = section.get("mali_mesh_file")
        ice_sheet = section.get("ice_sheet")
        output_base_path = section.get("output_base_path")

        # Check if user provided a path to reuse existing weight files
        mapping_files_path = section.get("mapping_files_path")

        section = config["ismip7_atmosphere"]
        method_remap = section.get("method_remap")

        # Construct the canonical mapping file name
        mapping_file = (f"map_ismip7_{ice_sheet}_atm_to_"
                        f"{mali_mesh_name}_{method_remap}.nc")

        # If mapping_files_path is provided, symlink any existing file
        if mapping_files_path != "NotAvailable":
            source_file = os.path.join(mapping_files_path, mapping_file)
            if os.path.exists(source_file):
                logger.info(f"Symlinking existing mapping file from "
                            f"{mapping_files_path}")
                if os.path.exists(mapping_file):
                    os.remove(mapping_file)
                os.symlink(source_file, mapping_file)

        # Use any atmosphere file as the grid template for building the
        # mapping file. We'll use the first year's acabf (SMB) file.
        ismip7_section = config["ismip7"]
        base_path_ismip7 = ismip7_section.get("base_path_ismip7")
        mali_mesh_name = ismip7_section.get("mali_mesh_name")
        mali_mesh_file = ismip7_section.get("mali_mesh_file")
        ice_sheet = ismip7_section.get("ice_sheet")
        output_base_path = ismip7_section.get("output_base_path")
        mapping_files_path = ismip7_section.get("mapping_files_path")

        atmosphere_section = config["ismip7_atmosphere"]
        method_remap = atmosphere_section.get("method_remap")

        prefix = params["prefix"]
        resolution = params["atm_resolution"]
        version = params["atm_version"]

        model = (ismip7_section.get("model")
                 if params["atm_model"] is None
                 else params["atm_model"])

        scenario = ismip7_section.get("scenario")

        input_path = os.path.join(base_path_ismip7, "acabf", version)
        # Use a simple pattern that will match the first available file
        file_pattern = (f"acabf_{prefix}_{model}_{scenario}_"
                        f"SDBN1-{resolution}_{version}_*.nc")
        ismip7_grid_file = os.path.join(input_path, file_pattern)

        import glob
        grid_files = sorted(glob.glob(ismip7_grid_file))
        if not grid_files:
            raise FileNotFoundError(
                f"No atmosphere grid file found matching pattern:\n"
                f"  {ismip7_grid_file}")
        ismip7_grid_file = grid_files[0]
        logger.info(f"Using grid template: "
                    f"{os.path.basename(ismip7_grid_file)}")

        # Build the mapping file (build_mapping_file will skip if it already
        # exists, e.g., from the symlink above or a previous failed run)
        build_mapping_file(config, logger, ismip7_grid_file, mapping_file,
                           mali_mesh_file=mali_mesh_file,
                           method_remap=method_remap)

        # Copy the mapping file to output_base_path/mapping_files/ for reuse
        # in future runs
        mapping_files_dir = os.path.join(output_base_path, "mapping_files")
        if not os.path.exists(mapping_files_dir):
            os.makedirs(mapping_files_dir)

        dst = os.path.join(mapping_files_dir, mapping_file)
        # Only copy if it's a real file (not a symlink we just created)
        if not os.path.islink(mapping_file):
            logger.info(f"Copying mapping file to {mapping_files_dir} "
                        f"for reuse")
            shutil.copy(mapping_file, dst)
        else:
            logger.info(f"Mapping file is a symlink; not copying to "
                        f"{mapping_files_dir}")

        logger.info("Done building atmosphere mapping file.")

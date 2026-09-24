import glob
import os
import shutil

from compass.landice.ismip7.ice_sheet_params import get_params
from compass.landice.ismip7.mapping import build_mapping_file
from compass.step import Step


class BuildMappingFile(Step):
    """
    A step for building the ESMF mapping file (regridding weights) from the
    ISMIP7 ocean grid to the MALI mesh. This step runs ESMF_RegridWeightGen
    with the full esmf_ntasks allocation, so it can be run on a separate node
    allocation from the processing steps (which run on a single node).

    The mapping file can be reused across scenarios and ocean choices, since
    it depends only on the ice sheet, mesh name, and remapping method. If
    mapping_files_path is provided in the config, this step will symlink any
    existing weight file from that directory and skip the build.
    """

    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.landice.tests.ismip7_forcing.ocean_thermal.\
OceanThermal
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
        model = section.get("model")
        scenario = section.get("scenario")

        # Check if user provided a path to reuse existing weight files
        mapping_files_path = section.get("mapping_files_path")

        section = config["ismip7_ocean_thermal"]
        method_remap = section.get("method_remap")

        # Construct the canonical mapping file name
        mapping_file = (f"map_ismip7_{ice_sheet}_ocean_to_"
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

        # Use any ocean thermal forcing file as the grid template for building
        # the mapping file. Look for the first available thermal_forcing file.
        prefix = params['prefix']
        version = params['ocean_version']
        ocean_grid = params['ocean_grid']

        # For OCX scenarios, the directory structure differs
        if params.get('ocean_choice_layout', False):
            # AIS OCX: files are in subdirectories (main/cold/warm/vary)
            # Use 'main' as the default choice for building the mapping file
            ocean_choice = 'main'
            input_path = os.path.join(
                base_path_ismip7, "ocean", ocean_choice, version)
            file_pattern = (f"tf_{prefix}_OCX_{ocean_grid}_{ocean_choice}_"
                            f"{version}_*.nc")
        else:
            # Standard ESM scenario or GrIS OCX
            if params['ocean_model'] is not None:
                # OCX (GrIS): use the ocean_model token
                ocean_model = params['ocean_model']
                input_path = os.path.join(
                    base_path_ismip7, "ocean", "tf", version)
                file_pattern = (f"tf_{prefix}_{ocean_model}_"
                                f"{ocean_grid}_{version}_*.nc")
            else:
                # Standard ESM scenario
                input_path = os.path.join(
                    base_path_ismip7, "ocean", version)
                file_pattern = (f"tf_{prefix}_{model}_"
                                f"{scenario}_{ocean_grid}_{version}_*.nc")

        ismip7_grid_files = sorted(
            glob.glob(os.path.join(input_path, file_pattern)))
        if not ismip7_grid_files:
            raise FileNotFoundError(
                f"No ocean thermal forcing file found matching pattern:\n"
                f"  {os.path.join(input_path, file_pattern)}")

        ismip7_grid_file = ismip7_grid_files[0]
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

        logger.info("Done building ocean mapping file.")

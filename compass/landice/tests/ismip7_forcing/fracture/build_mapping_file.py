import glob
import os
import shutil

from compass.landice.ismip7.mapping import build_mapping_file
from compass.step import Step


class BuildMappingFile(Step):
    """
    A step for building ESMF mapping files (regridding weights) from the
    ISMIP7 fracture grid to the MALI mesh. This step runs ESMF_RegridWeightGen
    with the full esmf_ntasks allocation, so it can be run on a separate node
    allocation from the processing steps (which run on a single node).

    Up to three mapping files may be built, one for each enabled fracture
    pathway (shelf collapse, excess melt, lake properties), since each may
    use a different remapping method. The mapping files can be reused across
    scenarios. If mapping_files_path is provided in the config, this step will
    symlink any existing weight files from that directory and skip building
    those that already exist.
    """

    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.landice.tests.ismip7_forcing.fracture.Fracture
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

        section = config["ismip7"]
        base_path_ismip7 = section.get("base_path_ismip7")
        mali_mesh_name = section.get("mali_mesh_name")
        mali_mesh_file = section.get("mali_mesh_file")
        ice_sheet = section.get("ice_sheet")
        output_base_path = section.get("output_base_path")

        # Check if user provided a path to reuse existing weight files
        mapping_files_path = section.get("mapping_files_path")

        section = config["ismip7_fracture"]
        version = section.get("version")

        # Three fracture pathways, each with its own remapping method
        fracture_paths = [
            ("shelf_collapse", "method_remap_shelf_collapse",
             "ice_shelf_collapse_mask_*.nc"),
            ("excess_melt", "method_remap_excess_melt",
             "excess_meltwaterinput_*.nc"),
            ("lake_properties", "method_remap_lake_properties",
             "lake_properties_*.nc"),
        ]

        # Build a mapping file for each enabled pathway
        for pathway_name, method_key, file_pattern in fracture_paths:
            method_remap = section.get(method_key)

            # Skip if method is None
            if method_remap.lower() == "none":
                logger.info(f"{method_key} is None; skipping {pathway_name} "
                            f"mapping file.")
                continue

            # Construct the canonical mapping file name
            mapping_file = (f"map_ismip7_{ice_sheet}_fracture_to_"
                            f"{mali_mesh_name}_{method_remap}.nc")

            # If mapping_files_path is provided, symlink any existing file
            if mapping_files_path != "NotAvailable":
                source_file = os.path.join(mapping_files_path, mapping_file)
                if os.path.exists(source_file):
                    logger.info(f"Symlinking existing mapping file for "
                                f"{pathway_name} from {mapping_files_path}")
                    if os.path.exists(mapping_file):
                        os.remove(mapping_file)
                    os.symlink(source_file, mapping_file)

            # Use the pathway's fracture file as the grid template
            input_path = os.path.join(base_path_ismip7, "fracture", version)
            grid_files = sorted(
                glob.glob(os.path.join(input_path, file_pattern)))
            if not grid_files:
                raise FileNotFoundError(
                    f"No {pathway_name} file found matching pattern:\n"
                    f"  {os.path.join(input_path, file_pattern)}")

            ismip7_grid_file = grid_files[0]
            logger.info(f"Building {pathway_name} mapping file using grid "
                        f"template: {os.path.basename(ismip7_grid_file)}")

            # Build the mapping file (build_mapping_file will skip if it
            # already exists, e.g., from the symlink above or a previous
            # failed run)
            build_mapping_file(config, logger, ismip7_grid_file, mapping_file,
                               mali_mesh_file=mali_mesh_file,
                               method_remap=method_remap)

            # Copy the mapping file to output_base_path/mapping_files/ for
            # reuse in future runs
            mapping_files_dir = os.path.join(output_base_path, "mapping_files")
            if not os.path.exists(mapping_files_dir):
                os.makedirs(mapping_files_dir)

            dst = os.path.join(mapping_files_dir, mapping_file)
            # Only copy if it's a real file (not a symlink we just created)
            if not os.path.islink(mapping_file):
                logger.info(f"Copying {pathway_name} mapping file to "
                            f"{mapping_files_dir} for reuse")
                shutil.copy(mapping_file, dst)
            else:
                logger.info(f"{pathway_name} mapping file is a symlink; not "
                            f"copying to {mapping_files_dir}")

        logger.info("Done building fracture mapping files.")

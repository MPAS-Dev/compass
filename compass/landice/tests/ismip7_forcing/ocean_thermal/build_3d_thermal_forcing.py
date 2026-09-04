import os
from pathlib import Path

from compass.landice.tests.ismip7_forcing.ice_sheet_params import get_params
from compass.landice.tests.ismip7_forcing.ocean_thermal import greenland_3d
from compass.step import Step


class BuildGreenland3dThermalForcing(Step):
    """
    A step that builds regional 3-D Greenland ocean thermal forcing from the
    2-D forcing produced by ProcessThermalForcing. GrIS only; gated by the
    ``process_ocean_thermal_3d`` config option. The 3-D-specific parameters
    are supplied through a JSON config file (``config_file``); the mesh, 2-D
    forcing, output, and diagnostics paths are injected from the compass
    config so the step auto-chains from the 2-D forcing.
    """

    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.landice.tests.ismip7_forcing.ocean_thermal.OceanThermal
            The test case this step belongs to
        """  # noqa: E501
        super().__init__(test_case=test_case,
                         name="build_3d_thermal_forcing")

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        logger = self.logger
        section = config["ismip7"]

        if not section.getboolean("process_ocean_thermal_3d"):
            logger.info("process_ocean_thermal_3d is false; skipping 3-D "
                        "Greenland thermal forcing.")
            return

        ice_sheet = section.get("ice_sheet")
        if ice_sheet != "gis":
            raise ValueError(
                "process_ocean_thermal_3d is only supported for the Greenland "
                "Ice Sheet (ice_sheet = gis); the Antarctic pathway already "
                "produces 3-D thermal forcing directly.")

        base_path_mali = section.get("base_path_mali")
        mali_mesh_file = section.get("mali_mesh_file")
        mali_mesh_name = section.get("mali_mesh_name")
        model = section.get("model")
        scenario = section.get("scenario")
        output_base_path = section.get("output_base_path")

        # Mirror ProcessThermalForcing._run_scenario forcing_group and ocean
        # source so we find the 2-D forcing it just wrote.
        params = get_params(config)
        if params["ocean_model"] is not None:
            forcing_group = scenario
            source = params["ocean_model"]
        else:
            forcing_group = f"{model}_{scenario}"
            source = model

        ocean_section = config["ismip7_ocean_thermal"]
        start_year = ocean_section.getint("start_year")
        end_year = ocean_section.getint("end_year")

        ocean_dir = os.path.join(output_base_path, forcing_group,
                                 "ocean_thermal_forcing")
        forcing_2d = os.path.join(
            ocean_dir,
            f"{mali_mesh_name}_2dThermalForcing_{source}_{scenario}_"
            f"{start_year}-{end_year}.nc")
        output_file = os.path.join(
            ocean_dir,
            f"{mali_mesh_name}_3dThermalForcing_{source}_{scenario}_"
            f"{start_year}-{end_year}.nc")

        json_path = config.get("ismip7_ocean_thermal_3d", "config_file")
        if json_path == "NotAvailable":
            raise ValueError(
                "You need to supply the [ismip7_ocean_thermal_3d] config_file "
                "option (path to the 3-D forcing JSON config) when "
                "process_ocean_thermal_3d is true")

        overrides = {
            "mesh_file": Path(os.path.join(base_path_mali, mali_mesh_file)),
            "forcing_2d_file": Path(forcing_2d),
            "output_file": Path(output_file),
            "diagnostics_directory": Path(
                os.path.join(self.work_dir, "diagnostics_3d")),
        }
        cfg = greenland_3d.Config.from_json(Path(json_path),
                                            overrides=overrides)
        greenland_3d.run(cfg, logger)

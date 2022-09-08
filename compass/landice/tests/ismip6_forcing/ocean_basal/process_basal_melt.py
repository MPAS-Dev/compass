import os
import subprocess
import xarray as xr
from compass.landice.tests.ismip6_forcing.ocean_basal.create_mapfile \
    import build_mapping_file
from mpas_tools.io import write_netcdf
from compass.step import Step


class ProcessBasalMelt(Step):
    """
    A step for processing (combine, remap and rename) the ISMIP6 basalmelt
    data
    """

    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.landice.tests.ismip6_forcing.ocean_basal.OceanBasal
            The test case this step belongs to
        """
        super().__init__(test_case=test_case, name='process_basal_melt')

    def setup(self):
        """
        Set up this step of the test case
        """
        config = self.config
        section = config['ismip6_ais']
        base_path = section.get('base_path')
        mali_mesh_file = section.get('mali_mesh_file')
        section = config['ismip6_ais_ocean_basal']
        basin_file = section.get('basin_file')
        coeff_file = section.get('coeff_file')
        self.add_input_file(filename=basin_file,
                            target=os.path.join(base_path,
                                                basin_file))
        self.add_input_file(filename=coeff_file,
                            target=os.path.join(base_path,
                                                coeff_file))
        self.add_input_file(filename=mali_mesh_file,
                            target=os.path.join(base_path, mali_mesh_file))
        self.add_output_file(filename=f"output_basin_and_{coeff_file}")

    def run(self):
        """
        Run this step of the test case
        """
        # logger = self.logger
        config = self.config
        section = config['ismip6_ais']
        mali_mesh_name = section.get('mali_mesh_name')
        mali_mesh_file = section.get('mali_mesh_file')
        section = config['ismip6_ais_ocean_basal']
        basin_file = section.get('basin_file')
        coeff_file = section.get('coeff_file')
        method_remap = section.get('method_remap')
        output_file = f"output_basin_and_{coeff_file}"

        # combine, interpolate and rename the basin file and deltaT0_gamma0
        # ismip6 input files
        combined_file_temp = "combined.nc"  # temporary file names
        remapped_file_temp = "remapped.nc"

        # call the function that combines data
        # logger.info = ('calling combine_ismip6_inputfiles')
        self.combine_ismip6_inputfiles(basin_file, coeff_file,
                                       combined_file_temp)

        # call the function that reads in, remap and rename the file.
        print("Calling a remapping function...")
        self.remap_ismip6BasalMelt_to_mali(combined_file_temp,
                                           remapped_file_temp, mali_mesh_name,
                                           mali_mesh_file, method_remap)

        # call the function that renames the ismip6 variables to MALI variables
        print("Renaming the ismip6 variables to mali variable names...")
        self.rename_ismip6BasalMelt_to_mali_vars(remapped_file_temp,
                                                 output_file)

        print("Remapping and renamping process done successfully. "
              "Removing the temporary files 'combined.nc' and 'remapped.nc'")

        # remove the temporary combined file
        os.remove(combined_file_temp)
        os.remove(remapped_file_temp)

    def combine_ismip6_inputfiles(self, basin_file, coeff_gamma0_deltaT_file,
                                  combined_file_temp):
        """
        Combine ismip6 input files before regridding onto mali mesh

        Parameters
        ----------
        basin_file : str
            imbie2 basin numbers in ismip6 grid
        coeff_gamma0_deltaT_file : str
            uniform melt rate coefficient (gamma0) and temperature
            correction per basin
        combined_file_temp : str
            temporary output file that has all the variables combined
        """

        ds_basin = xr.open_dataset(basin_file, engine="netcdf4")
        ds = xr.open_dataset(coeff_gamma0_deltaT_file, engine="netcdf4")

        ds["ismip6shelfMelt_basin"] = ds_basin.basinNumber
        write_netcdf(ds, combined_file_temp)

    def remap_ismip6BasalMelt_to_mali(self, input_file, output_file,
                                      mali_mesh_name, mali_mesh_file,
                                      method_remap):
        """
        Remap the input ismip6 basal melt data onto mali mesh

        Parameters
        ----------
        input_file: str
            temporary output file that has all the variables combined
            combined_file_temp generated in the above function
        output_file : str
            ismip6 data remapped on mali mesh
        mali_mesh_name : str
            name of the mali mesh used to name mapping files
        mali_mesh_file : str, optional
            The MALI mesh file if mapping file does not exist
        method_remap : str, optional
            Remapping method used in building a mapping file
        """

        # check if mapfile exists
        mapping_file = f"map_ismip6_8km_to_{mali_mesh_name}_{method_remap}.nc"

        if not os.path.exists(mapping_file):
            # build a mapping file if it doesn't already exist
            build_mapping_file(input_file, mapping_file, mali_mesh_file,
                               method_remap)
        else:
            print("Mapping file exists. Remapping the input data...")

        # remap the input data
        args = ["ncremap",
                "-i", input_file,
                "-o", output_file,
                "-m", mapping_file]

        subprocess.check_call(args)

    def rename_ismip6BasalMelt_to_mali_vars(self, remapped_file_temp,
                                            output_file):
        """
        Rename variables in the remapped ismip6 input data
        to the ones that MALI uses.

        Parameters
        ----------
        remapped_file_temp : str
            temporary ismip6 data remapped on mali mesh
        output_file : str
            remapped ismip6 data renamed on mali mesh
        """

        # open dataset in 20 years chunk
        ds = xr.open_dataset(remapped_file_temp, chunks=dict(time=20))

        # build dictionary and rename the ismip6 dimension and variables
        ismip6_to_mali_dims = dict(
            ncol="nCells")
        ds = ds.rename(ismip6_to_mali_dims)

        ismip6_to_mali_vars = dict(
            deltaT_basin="ismip6shelfMelt_deltaT",
            gamma0="ismip6shelfMelt_gamma0")

        # drop unnecessary variables on the regridded data on MALI mesh
        ds = ds.rename(ismip6_to_mali_vars)
        ds = ds.drop_vars(["lat_vertices", "lon_vertices",
                           "lat", "lon", "area"])

        # write to a new netCDF file
        write_netcdf(ds, output_file)
        ds.close()

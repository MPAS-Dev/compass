import xarray as xr
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call
from mpas_tools.mesh.cull import write_culled_dataset, write_map_culled_to_base

from compass.step import Step


class Cull(Step):
    """
    A step for culling MPAS-Ocean and -Seaice restart files to exclude
    ice-shelf cavities
    """

    def __init__(self, test_case):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.utility.cull_restarts.CullRestarts
            The test case this step belongs to
        """
        super().__init__(test_case, name='cull', cpus_per_task=128,
                         min_cpus_per_task=1)
        self.add_output_file(filename='unculled_mesh.nc')
        self.add_output_file(filename='mesh_no_isc.nc')
        self.add_output_file(filename='culled_graph.info')
        self.add_output_file(filename='no_isc_to_culled_map.nc')
        self.add_output_file(filename='culled_ocean_restart.nc')
        self.add_output_file(filename='culled_seaice_restart.nc')

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        logger = self.logger

        section = config['cull_restarts']
        ocean_restart_filename = section['ocean_restart_filename']
        if ocean_restart_filename == '<<<Missing>>>':
            raise ValueError('It looks like you did not set the '
                             'ocean_restart_filename config option.')

        seaice_restart_filename = section['seaice_restart_filename']
        if seaice_restart_filename == '<<<Missing>>>':
            raise ValueError('It looks like you did not set the '
                             'seaice_restart_filename config option.')

        culled_mesh_filename = 'mesh_no_isc.nc'
        map_filename = 'no_isc_to_culled_map.nc'

        mesh_vars = [
            'areaCell', 'cellsOnCell', 'edgesOnCell', 'indexToCellID',
            'latCell', 'lonCell', 'meshDensity', 'nEdgesOnCell',
            'verticesOnCell', 'xCell', 'yCell', 'zCell', 'angleEdge',
            'cellsOnEdge', 'dcEdge', 'dvEdge', 'edgesOnEdge',
            'indexToEdgeID', 'latEdge', 'lonEdge', 'nEdgesOnCell',
            'nEdgesOnEdge', 'verticesOnEdge', 'weightsOnEdge', 'xEdge',
            'yEdge', 'zEdge', 'areaTriangle', 'cellsOnVertex', 'edgesOnVertex',
            'indexToVertexID', 'kiteAreasOnVertex', 'latVertex',
            'lonVertex', 'xVertex', 'yVertex', 'zVertex']
        ds_unculled_mesh = xr.open_dataset(ocean_restart_filename)
        ds_unculled_mesh = ds_unculled_mesh[mesh_vars + ['landIceMask']]
        # cull cells where landIceMask == 1
        ds_unculled_mesh = ds_unculled_mesh.rename({'landIceMask': 'cullCell'})

        write_netcdf(ds_unculled_mesh, 'unculled_mesh.nc')

        args = ['MpasCellCuller.x', 'unculled_mesh.nc', culled_mesh_filename]
        # also produces culled_graph.info
        check_call(args=args, logger=logger)

        write_map_culled_to_base(base_mesh_filename=ocean_restart_filename,
                                 culled_mesh_filename=culled_mesh_filename,
                                 out_filename=map_filename,
                                 workers=self.cpus_per_task)

        in_filename = ocean_restart_filename
        out_filename = 'culled_ocean_restart.nc'

        write_culled_dataset(in_filename=in_filename,
                             out_filename=out_filename,
                             base_mesh_filename=ocean_restart_filename,
                             culled_mesh_filename=culled_mesh_filename,
                             map_culled_to_base_filename=map_filename,
                             logger=logger)

        in_filename = seaice_restart_filename
        out_filename = 'culled_seaice_restart.nc'

        write_culled_dataset(in_filename=in_filename,
                             out_filename=out_filename,
                             base_mesh_filename=ocean_restart_filename,
                             culled_mesh_filename=culled_mesh_filename,
                             map_culled_to_base_filename=map_filename,
                             logger=logger)

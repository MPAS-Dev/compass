import os
from datetime import datetime

import xarray as xr

from compass.io import symlink
from compass.step import Step


class FilesForE3SMStep(Step):
    """
    A superclass for steps in the FilesForE3SM test case

    Attributes
    ----------
    mesh_short_name : str
        The E3SM short name of the mesh

    creation_date : str
        The creation date of the mesh in YYYYMMDD format

    ocean_inputdata_dir : str
        The relative path to the ocean inputdata directory for this mesh

    seaice_inputdata_dir : str
        The relative path to the sea-ice inputdata directory for this mesh

    with_ice_shelf_cavities : bool
        Whether the mesh includes ice-shelf cavities

    ocean_mesh_dir : str
        The relative path to the ocean inputdata directory for all meshes

    seaice_mesh_dir : str
        The relative path to the sea-ice inputdata directory for all meshes

    mesh_vars : list
        A list of the variables that belong to the MPAS horizontal mesh
    """

    def __init__(self, test_case, name='files_for_e3sm_step', subdir=None,
                 cpus_per_task=1, min_cpus_per_task=1, ntasks=1, min_tasks=1,
                 **kwargs):
        """
        Create a new test case

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        name : str
            the name of the test case

        subdir : str, optional
            the subdirectory for the step.  The default is ``name``

        cpus_per_task : int, optional
            the number of cores per task the step would ideally use.  If
            fewer cores per node are available on the system, the step will
            run on all available cores as long as this is not below
            ``min_cpus_per_task``

        min_cpus_per_task : int, optional
            the number of cores per task the step requires.  If the system
            has fewer than this number of cores per node, the step will fail

        ntasks : int, optional
            the number of tasks the step would ideally use.  If too few
            cores are available on the system to accommodate the number of
            tasks and the number of cores per task, the step will run on
            fewer tasks as long as as this is not below ``min_tasks``

        min_tasks : int, optional
            the number of tasks the step requires.  If the system has too
            few cores to accommodate the number of tasks and cores per task,
            the step will fail

        """
        super().__init__(test_case=test_case, name=name, subdir=subdir,
                         cpus_per_task=cpus_per_task,
                         min_cpus_per_task=min_cpus_per_task, ntasks=ntasks,
                         min_tasks=min_tasks, **kwargs)

        self.mesh_short_name = None
        self.creation_date = None
        self.ocean_inputdata_dir = None
        self.seaice_inputdata_dir = None
        self.with_ice_shelf_cavities = None

        self.ocean_mesh_dir = \
            '../assembled_files/inputdata/share/meshes/mpas/ocean'

        self.seaice_mesh_dir = \
            '../assembled_files/inputdata/share/meshes/mpas/sea-ice'

        self.mesh_vars = [
            'areaCell', 'cellsOnCell', 'edgesOnCell', 'indexToCellID',
            'latCell', 'lonCell', 'meshDensity', 'nEdgesOnCell',
            'verticesOnCell', 'xCell', 'yCell', 'zCell', 'angleEdge',
            'cellsOnEdge', 'dcEdge', 'dvEdge', 'edgesOnEdge',
            'indexToEdgeID', 'latEdge', 'lonEdge', 'nEdgesOnCell',
            'nEdgesOnEdge', 'verticesOnEdge', 'weightsOnEdge', 'xEdge',
            'yEdge', 'zEdge', 'areaTriangle', 'cellsOnVertex', 'edgesOnVertex',
            'indexToVertexID', 'kiteAreasOnVertex', 'latVertex',
            'lonVertex', 'xVertex', 'yVertex', 'zVertex']

    def setup(self):
        """
        setup input files based on config options
        """
        config = self.config
        self.add_input_file(filename='README', target='../README')

        for prefix in ['base_mesh', 'initial_state', 'restart']:
            filename = config.get('files_for_e3sm',
                                  f'ocean_{prefix}_filename')
            if filename != 'autodetect':
                filename = os.path.normpath(os.path.join(
                    self.test_case.work_dir, filename))
                self.add_input_file(filename='base_mesh.nc',
                                    target=filename)

        with_ice_shelf_cavities = config.get('files_for_e3sm',
                                             'with_ice_shelf_cavities')
        if with_ice_shelf_cavities != 'autodetect':
            self.with_ice_shelf_cavities = \
                (with_ice_shelf_cavities.lower() == 'true')
        super().setup()

    def run(self):  # noqa: C901
        """
        Run this step of the testcase
        """
        config = self.config
        for prefix in ['base_mesh', 'initial_state', 'restart']:
            if not os.path.exists(f'{prefix}.nc'):
                filename = config.get('files_for_e3sm',
                                      f'ocean_{prefix}_filename')
                if filename == 'autodetect':
                    raise ValueError(f'No file was provided in the '
                                     f'ocean_{prefix}_filename config option.')
                filename = os.path.normpath(os.path.join(
                    self.test_case.work_dir, filename))
                if not os.path.exists(filename):
                    raise FileNotFoundError(
                        f'The ocean file given in ocean_{prefix}_filename '
                        f'could not be found.')
                if filename != f'{prefix}.nc':
                    symlink(filename, f'{prefix}.nc')

        mesh_short_name = config.get('files_for_e3sm', 'mesh_short_name')
        creation_date = config.get('global_ocean', 'creation_date')
        with xr.open_dataset('restart.nc') as ds:
            if 'MPAS_Mesh_Short_Name' in ds.attrs:
                if mesh_short_name == 'autodetect':
                    mesh_short_name = ds.attrs['MPAS_Mesh_Short_Name']
                if creation_date == 'autodetect':
                    # search for the creation date attribute
                    for attr in ds.attrs:
                        if attr.startswith('MPAS_Mesh') and \
                                attr.endswith('Version_Creation_Date'):
                            creation_date = ds.attrs[attr]
                            # convert to the desired format
                            try:
                                date = datetime.strptime(creation_date,
                                                         '%m/%d/%Y %H:%M:%S')
                                creation_date = date.strftime("%Y%m%d")
                            except ValueError:
                                # creation date isn't in this old format, so
                                # assume it's already YYYYMMDD
                                pass
                            break

        if mesh_short_name == 'autodetect':
            raise ValueError(
                'No mesh short name provided in "mesh_short_name" config '
                'option and none found in MPAS_Mesh_Short_Name attribute.')

        if creation_date == 'autodetect':
            now = datetime.now()
            creation_date = now.strftime("%Y%m%d")
            config.set('global_ocean', 'creation_date', creation_date)

        if self.with_ice_shelf_cavities is None:
            with_ice_shelf_cavities = self.config.get(
                'files_for_e3sm', 'with_ice_shelf_cavities')
            if with_ice_shelf_cavities == 'autodetect':
                self.with_ice_shelf_cavities = 'wISC' in mesh_short_name
            else:
                self.with_ice_shelf_cavities = \
                    (with_ice_shelf_cavities.lower() == 'true')

        self.mesh_short_name = mesh_short_name
        self.creation_date = creation_date

        self.ocean_inputdata_dir = \
            f'../assembled_files/inputdata/ocn/mpas-o/{mesh_short_name}'

        self.seaice_inputdata_dir = \
            f'../assembled_files/inputdata/ice/mpas-seaice/{mesh_short_name}'

        for dest_dir in [self.ocean_inputdata_dir, self.seaice_inputdata_dir,
                         self.ocean_mesh_dir, self.seaice_mesh_dir]:
            try:
                os.makedirs(dest_dir)
            except FileExistsError:
                pass

        super().run()

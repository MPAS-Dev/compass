import os

import numpy as np
import xarray as xr
from mpas_tools.io import write_netcdf

from compass.io import symlink
from compass.ocean.tests.global_ocean.files_for_e3sm.files_for_e3sm_step import (  # noqa: E501
    FilesForE3SMStep,
)
from compass.ocean.vertical import (
    compute_cell_mask,
    compute_ssh_from_layer_thickness,
    compute_zmid_from_layer_thickness,
)


class OceanMesh(FilesForE3SMStep):
    """
    A step for creating an MPAS-Ocean mesh from variables from an MPAS-Ocean
    initial state file
    """
    def __init__(self, test_case):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_ocean.files_for_e3sm.FilesForE3SM
            The test case this step belongs to
        """  # noqa: E501

        super().__init__(test_case, name='ocean_mesh')

        # for now, we won't define any outputs because they include the mesh
        # short name, which is not known at setup time.  Currently, this is
        # safe because no other steps depend on the outputs of this one.

    def run(self):
        """
        Run this step of the testcase
        """
        super().run()

        dest_filename = f'{self.mesh_short_name}.{self.creation_date}.nc'

        with xr.open_dataset('initial_state.nc') as ds:

            keep_vars = self.mesh_vars + [
                'refBottomDepth', 'vertCoordMovementWeights',
                'bottomDepth', 'maxLevelCell',
                'layerThickness', 'restingThickness'
            ]

            if 'minLevelCell' in ds:
                keep_vars.append('minLevelCell')

            if self.with_ice_shelf_cavities:
                keep_vars = keep_vars + [
                    'landIceMask', 'landIceDraft', 'landIceFraction',
                    'landIceFloatingMask', 'landIceFloatingFraction'
                ]

            ds = ds[keep_vars]
            ds.load()

            for attr in list(ds.attrs):
                # drop config options from global attributes
                if attr.startswith('config_'):
                    ds.attrs.pop(attr)

            ref_bot_depth = ds.refBottomDepth.values
            interfaces = np.append([0], ref_bot_depth)

            if 'minLevelCell' not in ds:
                ds['minLevelCell'] = xr.ones_like(ds.maxLevelCell)
                ds.minLevelCell.attrs['long_name'] = \
                    'Index to the first active ocean cell in each column.'

            ds['refTopDepth'] = ('nVertLevels', interfaces[0:-1])
            ds.refTopDepth.attrs['units'] = 'm'
            ds.refTopDepth.attrs['long_name'] = \
                "Reference depth of ocean for each vertical level. Used in " \
                "'z-level' type runs."
            ds['refZMid'] = ('nVertLevels',
                             -0.5 * (interfaces[1:] + interfaces[0:-1]))
            ds.refZMid.attrs['units'] = 'm'
            ds.refZMid.attrs['long_name'] = \
                'Reference mid z-coordinate of ocean for each vertical ' \
                'level. This has a negative value.'
            ds['refLayerThickness'] = ('nVertLevels',
                                       interfaces[1:] - interfaces[0:-1])
            ds.refLayerThickness.attrs['units'] = 'm'
            ds.refLayerThickness.attrs['long_name'] = \
                'Reference layer thickness of ocean for each vertical level.'

            ds['cellMask'] = compute_cell_mask(
                ds.minLevelCell - 1, ds.maxLevelCell - 1,
                ds.sizes['nVertLevels'])
            ds.cellMask.attrs['long_name'] = \
                'Mask on cells that determines if computations should be ' \
                'done on cells.'
            ds['ssh'] = compute_ssh_from_layer_thickness(
                ds.layerThickness, ds.bottomDepth, ds.cellMask)
            ds.ssh.attrs['units'] = 'm'
            ds.ssh.attrs['long_name'] = 'sea surface height'
            ds['zMid'] = compute_zmid_from_layer_thickness(
                ds.layerThickness, ds.ssh, ds.cellMask)
            ds.zMid.attrs['units'] = 'm'
            ds.zMid.attrs['long_name'] = \
                'z-coordinate of the mid-depth of the layer'

            write_netcdf(ds, dest_filename)

        symlink(os.path.abspath(dest_filename),
                f'{self.ocean_mesh_dir}/{dest_filename}')

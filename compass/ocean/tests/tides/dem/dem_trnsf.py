
import argparse

import netCDF4 as nc
import numpy as np

from compass.ocean.tests.tides.dem.dem_names import names

# Authors: Darren Engwirda


def dem_trnsf(base_mesh, part_mesh):  # noqa: C901
    """
    Transfer elevation & ice+ocn-thickness data from a full
    sphere MPAS mesh onto a partial sub-mesh resulting from
    a culling operation, or equiv.

    The sub-mesh is expected to be a piece of the base mesh
    such that (some) cells match 1:1.

    """

    base = nc.Dataset(base_mesh, "r")
    part = nc.Dataset(part_mesh, "r+")

    # -- make a vector of cell-centre positions to match against

    xpos = np.vstack((
        np.vstack((
            base["xCell"][:],
            base["yCell"][:],
            base["zCell"][:])).T,
        np.vstack((
            part["xCell"][:],
            part["yCell"][:],
            part["zCell"][:])).T
    ))

    xidx = np.hstack((
        np.arange(
            0, base.dimensions["nCells"].size),
        np.arange(
            0, part.dimensions["nCells"].size)
    ))

    # -- culling shouldn't introduce fp round-off - but truncate
    # -- anyway...

    xpos = np.round(xpos, decimals=9)

    # -- use stable sorting to bring matching cell xyz (and idx)
    # -- into "ascending" order

    imap = np.argsort(xpos[:, 2], kind="stable")
    xpos = xpos[imap, :]
    xidx = xidx[imap]
    imap = np.argsort(xpos[:, 1], kind="stable")
    xpos = xpos[imap, :]
    xidx = xidx[imap]
    imap = np.argsort(xpos[:, 0], kind="stable")
    xpos = xpos[imap, :]
    xidx = xidx[imap]

    diff = xpos[1:, :] - xpos[:-1, :]

    same = np.argwhere(np.logical_and.reduce((
        diff[:, 0] == 0,
        diff[:, 1] == 0,
        diff[:, 2] == 0))).ravel()

    # -- cell inew in part matches iold in base - re-index elev.
    # -- data-sets

    inew = xidx[same + 1]
    iold = xidx[same + 0]

    if ("bed_elevation" in base.variables.keys()):
        if ("bed_elevation" not in part.variables.keys()):
            part.createVariable("bed_elevation", "f4", ("nCells"))

        btmp = np.asarray(
            base["bed_elevation"][:], dtype=np.float32)

        ncel = part.dimensions["nCells"].size

        ptmp = np.zeros(ncel, dtype=np.float32)
        ptmp[inew] = btmp[iold]

        part["bed_elevation"].units = "m"
        part["bed_elevation"].long_name = names.bed_elevation
        part["bed_elevation"][:] = ptmp

    if ("bed_slope" in base.variables.keys()):
        if ("bed_slope" not in part.variables.keys()):
            part.createVariable("bed_slope", "f4", ("nCells"))

        btmp = np.asarray(
            base["bed_slope"][:], dtype=np.float32)

        ncel = part.dimensions["nCells"].size

        ptmp = np.zeros(ncel, dtype=np.float32)
        ptmp[inew] = btmp[iold]

        part["bed_slope"].units = "deg"
        part["bed_slope"].long_name = names.bed_slope_deg
        part["bed_slope"][:] = ptmp

    if ("bed_dz_dx" in base.variables.keys()):
        if ("bed_dz_dx" not in part.variables.keys()):
            part.createVariable("bed_dz_dx", "f4", ("nCells"))

        btmp = np.asarray(
            base["bed_dz_dx"][:], dtype=np.float32)

        ncel = part.dimensions["nCells"].size

        ptmp = np.zeros(ncel, dtype=np.float32)
        ptmp[inew] = btmp[iold]

        part["bed_dz_dx"].long_name = names.bed_dz_dx
        part["bed_dz_dx"][:] = ptmp

    if ("bed_dz_dy" in base.variables.keys()):
        if ("bed_dz_dy" not in part.variables.keys()):
            part.createVariable("bed_dz_dy", "f4", ("nCells"))

        btmp = np.asarray(
            base["bed_dz_dy"][:], dtype=np.float32)

        ncel = part.dimensions["nCells"].size

        ptmp = np.zeros(ncel, dtype=np.float32)
        ptmp[inew] = btmp[iold]

        part["bed_dz_dy"].long_name = names.bed_dz_dy
        part["bed_dz_dy"][:] = ptmp

    if ("ocn_thickness" not in part.variables.keys()):
        if ("ocn_thickness" not in part.variables.keys()):
            part.createVariable("ocn_thickness", "f4", ("nCells"))

        btmp = np.asarray(
            base["ocn_thickness"][:], dtype=np.float32)

        ncel = part.dimensions["nCells"].size

        ptmp = np.zeros(ncel, dtype=np.float32)
        ptmp[inew] = btmp[iold]

        part["ocn_thickness"].units = "m"
        part["ocn_thickness"].long_name = names.ocn_thickness
        part["ocn_thickness"][:] = ptmp

    if ("ice_thickness" not in part.variables.keys()):
        if ("ice_thickness" not in part.variables.keys()):
            part.createVariable("ice_thickness", "f4", ("nCells"))

        btmp = np.asarray(
            base["ice_thickness"][:], dtype=np.float32)

        ncel = part.dimensions["nCells"].size

        ptmp = np.zeros(ncel, dtype=np.float32)
        ptmp[inew] = btmp[iold]

        part["ice_thickness"].units = "m"
        part["ice_thickness"].long_name = names.ice_thickness
        part["ice_thickness"][:] = ptmp

    if ("bed_elevation_profile" in base.variables.keys()):
        if ("bed_elevation_profile" not in part.variables.keys()):
            part.createVariable("bed_elevation_profile",
                                "f4", ("nCells", "nProfiles"))

        btmp = np.asarray(
            base["bed_elevation_profile"][:, :], dtype=np.float32)

        ncel = part.dimensions["nCells"].size
        nprf = part.dimensions["nProfiles"].size

        ptmp = np.zeros((ncel, nprf), dtype=np.float32)
        ptmp[inew, :] = btmp[iold, :]

        part["bed_elevation_profile"].units = "m"
        part["bed_elevation_profile"].long_name = names.bed_elevation_profile
        part["bed_elevation_profile"][:, :] = ptmp

    if ("ocn_thickness_profile" in base.variables.keys()):
        if ("ocn_thickness_profile" not in part.variables.keys()):
            part.createVariable("ocn_thickness_profile",
                                "f4", ("nCells", "nProfiles"))

        btmp = np.asarray(
            base["ocn_thickness_profile"][:, :], dtype=np.float32)

        ncel = part.dimensions["nCells"].size
        nprf = part.dimensions["nProfiles"].size

        ptmp = np.zeros((ncel, nprf), dtype=np.float32)
        ptmp[inew, :] = btmp[iold, :]

        part["ocn_thickness_profile"].units = "m"
        part["ocn_thickness_profile"].long_name = names.ocn_thickness_profile
        part["ocn_thickness_profile"][:, :] = ptmp

    if ("ice_thickness_profile" in base.variables.keys()):
        if ("ice_thickness_profile" not in part.variables.keys()):
            part.createVariable("ice_thickness_profile",
                                "f4", ("nCells", "nProfiles"))

        btmp = np.asarray(
            base["ice_thickness_profile"][:, :], dtype=np.float32)

        ncel = part.dimensions["nCells"].size
        nprf = part.dimensions["nProfiles"].size

        ptmp = np.zeros((ncel, nprf), dtype=np.float32)
        ptmp[inew, :] = btmp[iold, :]

        part["ice_thickness_profile"].units = "m"
        part["ice_thickness_profile"].long_name = names.ice_thickness_profile
        part["ice_thickness_profile"][:, :] = ptmp

    base.close()
    part.close()

    return


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--base-mesh", dest="base_mesh", type=str,
        required=True, help="Name of (full) MPAS mesh.")

    parser.add_argument(
        "--part-mesh", dest="part_mesh", type=str,
        required=True, help="Name of culled MPAS mesh.")

    args = parser.parse_args()
    base_mesh = args.base_mesh
    part_mesh = args.part_mesh

    dem_trnsf(base_mesh, part_mesh)

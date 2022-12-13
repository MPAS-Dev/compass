
import numpy as np
import netCDF4 as nc
import argparse

# Authors: Darren Engwirda


def dem_trnsf(base_mesh, part_mesh):
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

    xpos = np.round(xpos, decimals=8)

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

    if ("bed_elevation" not in base.variables.keys() or
            "ocn_thickness" not in base.variables.keys() or
            "ice_thickness" not in base.variables.keys()):
        raise Exception("Base does not contain elev. data!")

    if ("bed_elevation" not in part.variables.keys()):
        part.createVariable("bed_elevation", "f8", ("nCells"))

    if ("ocn_thickness" not in part.variables.keys()):
        part.createVariable("ocn_thickness", "f8", ("nCells"))

    if ("ice_thickness" not in part.variables.keys()):
        part.createVariable("ice_thickness", "f8", ("nCells"))

    part["bed_elevation"][inew] = base["bed_elevation"][iold]
    part["ocn_thickness"][inew] = base["ocn_thickness"][iold]
    part["ice_thickness"][inew] = base["ice_thickness"][iold]

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

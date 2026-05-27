
import argparse
import os

import netCDF4 as nc
import numpy as np
from scipy.ndimage import gaussian_filter

from compass.ocean.tests.tides.dem.dem_names import names

# Authors: Darren Engwirda

RSPH = 6371220.


def circ_dist(xa, ya, xb, yb, rs=1.):
    """
    Calculate geodesic-length of great circles [PA, PB] on a
    sphere of radius RS.

    """

    dlon = .5 * (xa - xb)
    dlat = .5 * (ya - yb)

    dist = 2. * rs * np.arcsin(np.sqrt(
        np.sin(dlat) ** 2 +
        np.sin(dlon) ** 2 * np.cos(ya) * np.cos(yb)
    ))

    return dist


def cell_dzdx(xlon, ylat, vals, rsph):
    """
    Eval. the lon.- and lat.-aligned elevation gradients and
    their magnitude via a D8 stencil.

    """

    print("Building slopes...")

    cols = xlon.size - 2
    rows = ylat.size - 2

    xmid = .5 * (xlon[:-1:] + xlon[1::])
    ymid = .5 * (ylat[:-1:] + ylat[1::])

    ridx, icol = np.ogrid[:rows + 1, :cols + 1]

    # dz/dx is poorly conditioned at poles
    beta = (np.tanh((ymid + 87.5) * 2.5) -
            np.tanh((ymid - 87.5) * 2.5)) * 0.5
    beta = np.reshape(beta, (beta.size, 1))

    filt = np.asarray(vals, dtype=np.float32)

    part = rows // 20
    fbot = gaussian_filter(filt[+0:part * +1, :],
                           sigma=(4., cols / 512.), mode=("reflect", "wrap"))

    ftop = gaussian_filter(filt[19 * part:-1, :],
                           sigma=(4., cols / 512.), mode=("reflect", "wrap"))

    filt[+0:part * +1, :] = fbot
    filt[19 * part:-1, :] = ftop

    vals *= beta  # careful with mem. alloc.
    beta = (+1. - beta)
    vals += beta * filt
    # vals = beta * vals + (1. - beta) * filt

    del filt
    del ftop
    del fbot

    xmid = xmid * np.pi / 180.
    ymid = ymid * np.pi / 180.

    dzds = np.zeros((
        rows + 1, cols + 1), dtype=np.float32)

    dzdx = np.zeros((
        rows + 1, cols + 1), dtype=np.float32)
    dzdy = np.zeros((
        rows + 1, cols + 1), dtype=np.float32)

    indx = np.asarray(np.round(
        np.linspace(-1, rows, 32)), dtype=np.int64)

    print("* process tiles:")

    for tile in range(0, indx.size - 1):

        head = indx[tile + 0] + 1
        tail = indx[tile + 1] + 1

        slab = tail - head + 0

        irow = ridx[head:tail]

        zdel = np.zeros((
            slab + 0, cols + 1, 8), dtype=np.float32)

        xdel = np.zeros((
            slab + 0, cols + 1, 1), dtype=np.float32)
        ydel = np.zeros((
            slab + 0, cols + 1, 1), dtype=np.float32)

        zero = 0
        wcol = icol - 1
        wcol[wcol < zero] = cols
        ecol = icol + 1
        ecol[ecol > cols] = zero
        nrow = irow + 1
        nrow[nrow > rows] = rows
        srow = irow - 1
        srow[srow < zero] = zero

        # -- index D4 neighbours

        xdel[:, :, 0] = \
            vals[irow, ecol] - vals[irow, wcol]

        dist = circ_dist(xmid[ecol], ymid[irow],
                         xmid[wcol], ymid[irow])

        dist += 1.E-12
        xdel[:, :, 0] /= dist * rsph

        ydel[:, :, 0] = \
            vals[nrow, icol] - vals[srow, icol]

        dist = circ_dist(xmid[icol], ymid[nrow],
                         xmid[icol], ymid[srow])

        dist += 1.E-12
        ydel[:, :, 0] /= dist * rsph

        # -- index D8 neighbours =
        # -- NN, EE, SS, WW, NE, SE, SW, NW

        zdel[:, :, 0] = np.abs(
            vals[nrow, icol] - vals[irow, icol])
        zdel[:, :, 1] = np.abs(
            vals[irow, ecol] - vals[irow, icol])
        zdel[:, :, 2] = np.abs(
            vals[srow, icol] - vals[irow, icol])
        zdel[:, :, 3] = np.abs(
            vals[irow, wcol] - vals[irow, icol])

        zdel[:, :, 4] = np.abs(
            vals[nrow, ecol] - vals[irow, icol])
        zdel[:, :, 5] = np.abs(
            vals[srow, ecol] - vals[irow, icol])
        zdel[:, :, 6] = np.abs(
            vals[srow, wcol] - vals[irow, icol])
        zdel[:, :, 7] = np.abs(
            vals[nrow, wcol] - vals[irow, icol])

        dist = circ_dist(xmid[icol], ymid[irow],
                         xmid[icol], ymid[nrow])

        dist += 1.E-12
        zdel[:, :, 0] /= dist * rsph

        dist = circ_dist(xmid[icol], ymid[irow],
                         xmid[ecol], ymid[irow])

        dist += 1.E-12
        zdel[:, :, 1] /= dist * rsph

        dist = circ_dist(xmid[icol], ymid[irow],
                         xmid[icol], ymid[srow])

        dist += 1.E-12
        zdel[:, :, 2] /= dist * rsph

        dist = circ_dist(xmid[icol], ymid[irow],
                         xmid[wcol], ymid[irow])

        dist += 1.E-12
        zdel[:, :, 3] /= dist * rsph

        dist = circ_dist(xmid[icol], ymid[irow],
                         xmid[ecol], ymid[nrow])

        dist += 1.E-12
        zdel[:, :, 4] /= dist * rsph

        dist = circ_dist(xmid[icol], ymid[irow],
                         xmid[ecol], ymid[srow])

        dist += 1.E-12
        zdel[:, :, 5] /= dist * rsph

        dist = circ_dist(xmid[icol], ymid[irow],
                         xmid[wcol], ymid[srow])

        dist += 1.E-12
        zdel[:, :, 6] /= dist * rsph

        dist = circ_dist(xmid[icol], ymid[irow],
                         xmid[wcol], ymid[nrow])

        dist += 1.E-12
        zdel[:, :, 7] /= dist * rsph

        dzds[irow, icol] = \
            np.sqrt(np.mean(zdel**2, axis=2))

        dzdx[irow, icol] = xdel[:, :, 0]
        dzdy[irow, icol] = ydel[:, :, 0]

        del zdel
        del xdel
        del ydel

        print("* compute local D8 slope:",
              tile, "of", indx.size - 1)

    dzds[+1, :] = dzds[+2, :]
    dzdx[+1, :] = dzdx[+2, :]
    dzdy[+1, :] = dzdy[+2, :]

    dzds[-1, :] = dzds[-2, :]
    dzdx[-1, :] = dzdx[-2, :]
    dzdy[-1, :] = dzdy[-2, :]

    return dzds, dzdx, dzdy


def blend_front(e1st, i1st, e2nd, halo, sdev):
    """
    Create a mask of linear weights to 'blend' two elev. fun
    at the ice sheet/shelf front:

        ELEV = (1.-MASK) * E1ST + (0.+MASK) * E2ND

    Elev. data assoc. with the 1st data-set is preserved in
    pixels where ice-thickness is non-zero. The two datasets
    are blended over a distance of approx. HALO pixels. The
    mask is additionally smoothed via a Gaussian filter with
    standard-deviation sigma=SDEV.

    """

    mask = np.full(
        e1st.shape, halo + 1, dtype=np.float32)

    nidx = np.full(
        e1st.shape[0], False, dtype=bool)
    sidx = np.full(
        e1st.shape[0], False, dtype=bool)

    bnds = np.asarray(np.round(
        np.linspace(0, e1st.shape[0], 5)), dtype=np.uint32)

    sidx[bnds[0]:bnds[1]:] = True
    nidx[bnds[3]:bnds[4]:] = True

    print("* blending SOUTH.")

    part = mask[sidx, :]

    npos = np.arange(+1, part.shape[0] + 1)
    epos = np.arange(-1, part.shape[1] - 1)
    spos = np.arange(-1, part.shape[0] - 1)
    wpos = np.arange(+1, part.shape[1] + 1)

    Y = part.shape[0]
    X = part.shape[1]

    npos[npos >= +Y] = Y - 1
    spos[spos <= -1] = 0
    epos[epos <= -1] = X - 1
    wpos[wpos >= +X] = 0

    part[i1st[sidx, :] > 0] = 0.

    for inum in range(halo):

        part = np.minimum(part, part[npos, :] + 1.)
        part = np.minimum(part, part[spos, :] + 1.)
        part = np.minimum(part, part[:, epos] + 1.)
        part = np.minimum(part, part[:, wpos] + 1.)

        print("* compute local blending:",
              inum, "of", halo)

    mask[sidx, :] = part

    print("* blending NORTH.")

    part = mask[nidx, :]

    npos = np.arange(+1, part.shape[0] + 1)
    epos = np.arange(-1, part.shape[1] - 1)
    spos = np.arange(-1, part.shape[0] - 1)
    wpos = np.arange(+1, part.shape[1] + 1)

    Y = part.shape[0]
    X = part.shape[1]

    npos[npos >= +Y] = Y - 1
    spos[spos <= -1] = 0
    epos[epos <= -1] = X - 1
    wpos[wpos >= +X] = 0

    part[i1st[nidx, :] > 0] = 0.

    for inum in range(halo):

        part = np.minimum(part, part[npos, :] + 1.)
        part = np.minimum(part, part[spos, :] + 1.)
        part = np.minimum(part, part[:, epos] + 1.)
        part = np.minimum(part, part[:, wpos] + 1.)

        print("* compute local blending:",
              inum, "of", halo)

    mask[nidx, :] = part

    print("* blending final.")

    mask /= float(halo + 1.00)

    mask = gaussian_filter(mask, sigma=sdev, mode="wrap")
    mask = mask ** 1.50
    mask[i1st >= 1] = 0.

    return np.asarray(mask, dtype=np.float32)


def rtopo_60sec(elev_path, save_path):

    """
    Create a zipped and pixel centred version of RTopo 2.0.4
    (60 arc-sec) to support remapping of elevation data.

    """

    print("Making RTopo-2.0.4 (60 arc-sec) pixel...")

    data = nc.Dataset(os.path.join(
        elev_path, "RTopo-2.0.4_1min_data.nc"), "r")

    data.set_auto_maskandscale(False)  # quiet valid_min/max

    xpos = np.asarray(data["lon"][:], dtype=np.float64)
    ypos = np.asarray(data["lat"][:], dtype=np.float64)

    elev = np.asarray(
        data["bedrock_topography"][:], dtype=np.float32)
    surf = np.asarray(
        data["surface_elevation"][:], dtype=np.float32)
    base = np.asarray(
        data["ice_base_topography"][:], dtype=np.float32)

    elev = (elev[:-1:, :-1:] + elev[+1::, :-1:] +
            elev[:-1:, +1::] + elev[+1::, +1::]) / 4.

    surf = (surf[:-1:, :-1:] + surf[+1::, :-1:] +
            surf[:-1:, +1::] + surf[+1::, +1::]) / 4.

    base = (base[:-1:, :-1:] + base[+1::, :-1:] +
            base[:-1:, +1::] + base[+1::, +1::]) / 4.

    elev = np.asarray(np.round(elev), dtype=np.int16)
    surf = np.asarray(np.round(surf), dtype=np.int16)
    base = np.asarray(np.round(base), dtype=np.int16)

    iceh = surf - base
    iceh[base == 0] = 0

    ocnh = np.maximum(0, base - elev)

    root = nc.Dataset(
        os.path.join(
            save_path, "RTopo_2_0_4_60sec_pixel.nc"),
        "w", format="NETCDF4")
    root.description = "A zipped RTopo-2.0.4 (60 arc-sec) " + \
        "data-set, pixel centred and compressed to int16_t."
    root.source = "RTopo-2.0.4_1min_data.nc"
    root.references = "doi.pangaea.de/10.1594/PANGAEA.905295"
    root.createDimension("num_lon", elev.shape[1] + 1)
    root.createDimension("num_col", elev.shape[1])
    root.createDimension("num_lat", elev.shape[0] + 1)
    root.createDimension("num_row", elev.shape[0])

    data = root.createVariable("lon", "f8", ("num_lon"))
    data.units = "degrees_east"
    data[:] = xpos
    data = root.createVariable("lat", "f8", ("num_lat"))
    data.units = "degrees_north"
    data[:] = ypos
    data = root.createVariable(
        "bed_elevation", "i2", ("num_row", "num_col"))
    data.units = "m"
    data[:, :] = elev
    data = root.createVariable(
        "ice_thickness", "i2", ("num_row", "num_col"))
    data.units = "m"
    data[:, :] = iceh
    data = root.createVariable(
        "ocn_thickness", "i2", ("num_row", "num_col"))
    data.units = "m"
    data[:, :] = ocnh

    root.close()


def rtopo_30sec(elev_path, save_path):
    """
    Create a zipped and pixel centred version of RTopo 2.0.4
    (30 arc-sec) to support remapping of elevation data.

    """

    print("Making RTopo-2.0.4 (30 arc-sec) pixel...")

    data = nc.Dataset(os.path.join(
        elev_path,
        "RTopo-2.0.4_30sec_bedrock_topography.nc"), "r")

    data.set_auto_maskandscale(False)  # quiet valid_min/max

    xpos = np.asarray(data["lon"][:], dtype=np.float64)
    ypos = np.asarray(data["lat"][:], dtype=np.float64)

    elev = np.asarray(
        data["bedrock_topography"][:], dtype=np.float32)

    data = nc.Dataset(os.path.join(
        elev_path,
        "RTopo-2.0.4_30sec_surface_elevation.nc"), "r")

    data.set_auto_maskandscale(False)  # quiet valid_min/max

    surf = np.asarray(
        data["surface_elevation"][:], dtype=np.float32)

    data = nc.Dataset(os.path.join(
        elev_path,
        "RTopo-2.0.4_30sec_ice_base_topography.nc"), "r")

    data.set_auto_maskandscale(False)  # quiet valid_min/max

    base = np.asarray(
        data["ice_base_topography"][:], dtype=np.float32)

    elev = (elev[:-1:, :-1:] + elev[+1::, :-1:] +
            elev[:-1:, +1::] + elev[+1::, +1::]) / 4.

    surf = (surf[:-1:, :-1:] + surf[+1::, :-1:] +
            surf[:-1:, +1::] + surf[+1::, +1::]) / 4.

    base = (base[:-1:, :-1:] + base[+1::, :-1:] +
            base[:-1:, +1::] + base[+1::, +1::]) / 4.

    elev = np.asarray(np.round(elev), dtype=np.int16)
    surf = np.asarray(np.round(surf), dtype=np.int16)
    base = np.asarray(np.round(base), dtype=np.int16)

    iceh = surf - base
    iceh[base == 0] = 0

    ocnh = np.maximum(0, base - elev)

    root = nc.Dataset(
        os.path.join(
            save_path, "RTopo_2_0_4_30sec_pixel.nc"),
        "w", format="NETCDF4")
    root.description = "A zipped RTopo-2.0.4 (30 arc-sec) " + \
        "data-set, pixel centred and compressed to int16_t."
    root.source = "RTopo-2.0.4_30sec_data.nc"
    root.references = "doi.pangaea.de/10.1594/PANGAEA.905295"
    root.createDimension("num_lon", elev.shape[1] + 1)
    root.createDimension("num_col", elev.shape[1])
    root.createDimension("num_lat", elev.shape[0] + 1)
    root.createDimension("num_row", elev.shape[0])

    data = root.createVariable("lon", "f8", ("num_lon"))
    data.units = "degrees_east"
    data[:] = xpos
    data = root.createVariable("lat", "f8", ("num_lat"))
    data.units = "degrees_north"
    data[:] = ypos
    data = root.createVariable(
        "bed_elevation", "i2", ("num_row", "num_col"))
    data.units = "m"
    data[:, :] = elev
    data = root.createVariable(
        "ice_thickness", "i2", ("num_row", "num_col"))
    data.units = "m"
    data[:, :] = iceh
    data = root.createVariable(
        "ocn_thickness", "i2", ("num_row", "num_col"))
    data.units = "m"
    data[:, :] = ocnh

    root.close()


def rtopo_15sec(elev_path, save_path):
    """
    Create a zipped and pixel centred version of RTopo 2.0.4
    (15 arc-sec) to support remapping of elevation data.

    """

    print("Making RTopo-2.0.4 (15 arc-sec) pixel...")

    data = nc.Dataset(os.path.join(
        elev_path,
        "RTopo-2.0.4_30sec_bedrock_topography.nc"), "r")

    data.set_auto_maskandscale(False)  # quiet valid_min/max

    elev = np.asarray(
        data["bedrock_topography"][:], dtype=np.float32)

    zmid = (elev[:-1:, :-1:] + elev[+1::, :-1:] +
            elev[:-1:, +1::] + elev[+1::, +1::]) / 4.

    zlhs = (elev[:-1:, :-1:] + elev[+1::, :-1:]) / 2.
    zrhs = (elev[:-1:, +1::] + elev[+1::, +1::]) / 2.
    zbot = (elev[:-1:, :-1:] + elev[:-1:, +1::]) / 2.
    ztop = (elev[+1::, :-1:] + elev[+1::, +1::]) / 2.

    znew = np.zeros((43200, 86400), dtype=np.float32)
    znew[0::2, 0::2] = (
        elev[:-1:, :-1:] + zbot + zmid + zlhs) / 4.
    znew[0::2, 1::2] = (
        zbot + elev[:-1:, +1::] + zrhs + zmid) / 4.
    znew[1::2, 1::2] = (
        zmid + zrhs + elev[+1::, +1::] + ztop) / 4.
    znew[1::2, 0::2] = (
        zlhs + zmid + ztop + elev[+1::, :-1:]) / 4.

    del zmid
    del zlhs
    del zrhs
    del zbot
    del ztop

    elev = np.asarray(np.round(znew), dtype=np.int16)
    data.close()

    data = nc.Dataset(os.path.join(
        elev_path,
        "RTopo-2.0.4_30sec_surface_elevation.nc"), "r")

    data.set_auto_maskandscale(False)  # quiet valid_min/max

    surf = np.asarray(
        data["surface_elevation"][:], dtype=np.float32)

    zmid = (surf[:-1:, :-1:] + surf[+1::, :-1:] +
            surf[:-1:, +1::] + surf[+1::, +1::]) / 4.

    zlhs = (surf[:-1:, :-1:] + surf[+1::, :-1:]) / 2.
    zrhs = (surf[:-1:, +1::] + surf[+1::, +1::]) / 2.
    zbot = (surf[:-1:, :-1:] + surf[:-1:, +1::]) / 2.
    ztop = (surf[+1::, :-1:] + surf[+1::, +1::]) / 2.

    znew = np.zeros((43200, 86400), dtype=np.float32)
    znew[0::2, 0::2] = (
        surf[:-1:, :-1:] + zbot + zmid + zlhs) / 4.
    znew[0::2, 1::2] = (
        zbot + surf[:-1:, +1::] + zrhs + zmid) / 4.
    znew[1::2, 1::2] = (
        zmid + zrhs + surf[+1::, +1::] + ztop) / 4.
    znew[1::2, 0::2] = (
        zlhs + zmid + ztop + surf[+1::, :-1:]) / 4.

    del zmid
    del zlhs
    del zrhs
    del zbot
    del ztop

    surf = np.asarray(np.round(znew), dtype=np.int16)
    data.close()

    data = nc.Dataset(os.path.join(
        elev_path,
        "RTopo-2.0.4_30sec_ice_base_topography.nc"), "r")

    data.set_auto_maskandscale(False)  # quiet valid_min/max

    base = np.asarray(
        data["ice_base_topography"][:], dtype=np.float32)

    zmid = (base[:-1:, :-1:] + base[+1::, :-1:] +
            base[:-1:, +1::] + base[+1::, +1::]) / 4.

    zlhs = (base[:-1:, :-1:] + base[+1::, :-1:]) / 2.
    zrhs = (base[:-1:, +1::] + base[+1::, +1::]) / 2.
    zbot = (base[:-1:, :-1:] + base[:-1:, +1::]) / 2.
    ztop = (base[+1::, :-1:] + base[+1::, +1::]) / 2.

    znew = np.zeros((43200, 86400), dtype=np.float32)
    znew[0::2, 0::2] = (
        base[:-1:, :-1:] + zbot + zmid + zlhs) / 4.
    znew[0::2, 1::2] = (
        zbot + base[:-1:, +1::] + zrhs + zmid) / 4.
    znew[1::2, 1::2] = (
        zmid + zrhs + base[+1::, +1::] + ztop) / 4.
    znew[1::2, 0::2] = (
        zlhs + zmid + ztop + base[+1::, :-1:]) / 4.

    del zmid
    del zlhs
    del zrhs
    del zbot
    del ztop

    base = np.asarray(np.round(znew), dtype=np.int16)
    data.close()

    iceh = surf - base
    iceh[base == 0] = 0

    ocnh = np.maximum(0, base - elev)

    xpos = np.linspace(
        -180., +180., elev.shape[1] + 1, dtype=np.float64)
    ypos = np.linspace(
        -90.0, +90.0, elev.shape[0] + 1, dtype=np.float64)

    root = nc.Dataset(
        os.path.join(
            save_path, "RTopo_2_0_4_15sec_pixel.nc"),
        "w", format="NETCDF4")
    root.description = "A zipped RTopo-2.0.4 (15 arc-sec) " + \
        "data-set, pixel centred and compressed to int16_t."
    root.source = "RTopo-2.0.4_30sec_data.nc"
    root.references = "doi.pangaea.de/10.1594/PANGAEA.905295"
    root.createDimension("num_lon", elev.shape[1] + 1)
    root.createDimension("num_col", elev.shape[1])
    root.createDimension("num_lat", elev.shape[0] + 1)
    root.createDimension("num_row", elev.shape[0])

    data = root.createVariable("lon", "f8", ("num_lon"))
    data.units = "degrees_east"
    data[:] = xpos
    data = root.createVariable("lat", "f8", ("num_lat"))
    data.units = "degrees_north"
    data[:] = ypos
    data = root.createVariable(
        "bed_elevation", "i2", ("num_row", "num_col"))
    data.units = "m"
    data[:, :] = elev

    data = root.createVariable(
        "ice_thickness", "i2", ("num_row", "num_col"))
    data.units = "m"
    data[:, :] = iceh
    data = root.createVariable(
        "ocn_thickness", "i2", ("num_row", "num_col"))
    data.units = "m"
    data[:, :] = ocnh

    root.close()


def gebco_60sec(elev_path, save_path):
    """
    Create a zipped and pixel centred version of GEBCO[2023]
    (15 arc-sec) at 60 arc-sec.

    """

    print("Making GEBCO[2023] (60 arc-sec) pixel...")

    data = nc.Dataset(os.path.join(
        elev_path, "GEBCO_2023_sub_ice_topo.nc"), "r")

    elev = np.asarray(
        data["elevation"][:], dtype=np.float32)

    halo = 4
    z_60 = np.zeros((10800, 21600), dtype=np.float32)

    for ipos in range(halo):
        for jpos in range(halo):
            iend = elev.shape[0] - halo + ipos + 1
            jend = elev.shape[1] - halo + jpos + 1
            z_60 += elev[ipos:iend:halo, jpos:jend:halo]

    elev = np.asarray(
        np.round(z_60 / float(halo ** 2)), dtype=np.int16)

    xpos = np.linspace(
        -180., +180., elev.shape[1] + 1, dtype=np.float64)
    ypos = np.linspace(
        -90.0, +90.0, elev.shape[0] + 1, dtype=np.float64)

    root = nc.Dataset(
        os.path.join(
            save_path, "GEBCO_v2023_60sec_pixel.nc"),
        "w", format="NETCDF4")
    root.createDimension("num_lon", elev.shape[1] + 1)
    root.createDimension("num_col", elev.shape[1])
    root.createDimension("num_lat", elev.shape[0] + 1)
    root.createDimension("num_row", elev.shape[0])

    data = root.createVariable("lon", "f8", ("num_lon"))
    data.units = "degrees_east"
    data[:] = xpos
    data = root.createVariable("lat", "f8", ("num_lat"))
    data.units = "degrees_north"
    data[:] = ypos
    data = root.createVariable(
        "bed_elevation", "i2", ("num_row", "num_col"))
    data.units = "m"
    data[:, :] = elev

    root.close()


def gebco_30sec(elev_path, save_path):
    """
    Create a zipped and pixel centred version of GEBCO[2023]
    (15 arc-sec) at 30 arc-sec.

    """

    print("Making GEBCO[2023] (30 arc-sec) pixel...")

    data = nc.Dataset(os.path.join(
        elev_path, "GEBCO_2023_sub_ice_topo.nc"), "r")

    elev = np.asarray(
        data["elevation"][:], dtype=np.float32)

    halo = 2
    z_30 = np.zeros((21600, 43200), dtype=np.float32)

    for ipos in range(halo):
        for jpos in range(halo):
            iend = elev.shape[0] - halo + ipos + 1
            jend = elev.shape[1] - halo + jpos + 1
            z_30 += elev[ipos:iend:halo, jpos:jend:halo]

    elev = np.asarray(
        np.round(z_30 / float(halo ** 2)), dtype=np.int16)

    xpos = np.linspace(
        -180., +180., elev.shape[1] + 1, dtype=np.float64)
    ypos = np.linspace(
        -90.0, +90.0, elev.shape[0] + 1, dtype=np.float64)

    root = nc.Dataset(
        os.path.join(
            save_path, "GEBCO_v2023_30sec_pixel.nc"),
        "w", format="NETCDF4")
    root.createDimension("num_lon", elev.shape[1] + 1)
    root.createDimension("num_col", elev.shape[1])
    root.createDimension("num_lat", elev.shape[0] + 1)
    root.createDimension("num_row", elev.shape[0])

    data = root.createVariable("lon", "f8", ("num_lon"))
    data.units = "degrees_east"
    data[:] = xpos
    data = root.createVariable("lat", "f8", ("num_lat"))
    data.units = "degrees_north"
    data[:] = ypos
    data = root.createVariable(
        "bed_elevation", "i2", ("num_row", "num_col"))
    data.units = "m"
    data[:, :] = elev

    root.close()


def gebco_15sec(elev_path, save_path):
    """
    Create a zipped and pixel centred version of GEBCO[2023]
    (15 arc-sec) at 15 arc-sec.

    """

    print("Making GEBCO[2023] (15 arc-sec) pixel...")

    data = nc.Dataset(os.path.join(
        elev_path, "GEBCO_2023_sub_ice_topo.nc"), "r")

    elev = np.asarray(
        data["elevation"][:], dtype=np.int16)

    xpos = np.linspace(
        -180., +180., elev.shape[1] + 1, dtype=np.float64)
    ypos = np.linspace(
        -90.0, +90.0, elev.shape[0] + 1, dtype=np.float64)

    root = nc.Dataset(
        os.path.join(
            save_path, "GEBCO_v2023_15sec_pixel.nc"),
        "w", format="NETCDF4")
    root.createDimension("num_lon", elev.shape[1] + 1)
    root.createDimension("num_col", elev.shape[1])
    root.createDimension("num_lat", elev.shape[0] + 1)
    root.createDimension("num_row", elev.shape[0])

    data = root.createVariable("lon", "f8", ("num_lon"))
    data.units = "degrees_east"
    data[:] = xpos
    data = root.createVariable("lat", "f8", ("num_lat"))
    data.units = "degrees_north"
    data[:] = ypos
    data = root.createVariable(
        "bed_elevation", "i2", ("num_row", "num_col"))
    data.units = "m"
    data[:, :] = elev

    root.close()


def rtopo_gebco_60sec(elev_path, save_path):
    """
    Create a zipped and pixel centred 'blend' of RTopo 2.0.4
    and GEBCO[2023] at 60 arc-sec.

    """

    print("Making RTopo-GEBCO (60 arc-sec) blend...")

    data = nc.Dataset(os.path.join(
        elev_path, "RTopo_2_0_4_60sec_pixel.nc"), "r")

    e1st = np.asarray(
        data["bed_elevation"][:], dtype=np.int16)

    i1st = np.asarray(
        data["ice_thickness"][:], dtype=np.int16)

    o1st = np.asarray(
        data["ocn_thickness"][:], dtype=np.int16)

    data = nc.Dataset(os.path.join(
        elev_path, "GEBCO_v2023_60sec_pixel.nc"), "r")

    e2nd = np.asarray(
        data["bed_elevation"][:], dtype=np.int16)

    mask = blend_front(e1st, i1st, e2nd, halo=10, sdev=1.0)

    elev = np.asarray(np.round(
        (1. - mask) * e1st + mask * e2nd), dtype=np.int16)

    iceh = i1st
    ocnh = o1st
    ocnh[i1st == 0] = np.maximum(0, -elev[i1st == 0])

    xpos = np.linspace(
        -180., +180., elev.shape[1] + 1, dtype=np.float64)
    ypos = np.linspace(
        -90.0, +90.0, elev.shape[0] + 1, dtype=np.float64)

    root = nc.Dataset(
        os.path.join(save_path,
                     "RTopo_2_0_4_GEBCO_v2023_60sec_pixel.nc"),
        "w", format="NETCDF4")
    root.description = "Blend of RTopo-2.0.4 (60 arc-sec) " + \
        "and GEBCO[2023] (15 arc-sec) - pixel centred and " + \
        "compressed to int16_t. RTopo data used under ice " + \
        "sheets/shelves. Remapped to 60 arc-sec."
    root.source = \
        "RTopo-2.0.4_1min_data.nc and GEBCO_2023.nc"
    root.references = \
        "doi.pangaea.de/10.1594/PANGAEA.905295 and " + \
        "doi.org/10.5285/a29c5465-b138-234d-e053-6c86abc040b9"
    root.createDimension("num_lon", elev.shape[1] + 1)
    root.createDimension("num_col", elev.shape[1])
    root.createDimension("num_lat", elev.shape[0] + 1)
    root.createDimension("num_row", elev.shape[0])

    data = root.createVariable("lon", "f8", ("num_lon"))
    data.units = "degrees_east"
    data[:] = xpos
    data = root.createVariable("lat", "f8", ("num_lat"))
    data.units = "degrees_north"
    data[:] = ypos
    data = root.createVariable(
        "bed_elevation", "i2", ("num_row", "num_col"))
    data.units = "m"
    data.long_name = names.bed_elevation
    data[:, :] = elev
    data = root.createVariable(
        "ice_thickness", "i2", ("num_row", "num_col"))
    data.units = "m"
    data.long_name = names.ocn_thickness
    data[:, :] = iceh
    data = root.createVariable(
        "ocn_thickness", "i2", ("num_row", "num_col"))
    data.units = "m"
    data.long_name = names.ice_thickness
    data[:, :] = ocnh

    # filt. grid-scale noise that imprints on dz/dx...
    filt = gaussian_filter(np.asarray(
        elev, dtype=np.float32), sigma=.625, mode="wrap")

    zslp, dzdx, dzdy = \
        cell_dzdx(xpos, ypos, filt, RSPH)

    data = root.createVariable(
        "bed_slope", "f4", ("num_row", "num_col"))
    data.long_name = names.bed_slope
    data[:, :] = zslp[:, :]
    data = root.createVariable(
        "bed_dz_dx", "f4", ("num_row", "num_col"))
    data.long_name = names.bed_dz_dx
    data[:, :] = dzdx[:, :]
    data = root.createVariable(
        "bed_dz_dy", "f4", ("num_row", "num_col"))
    data.long_name = names.bed_dz_dy
    data[:, :] = dzdy[:, :]

    root.close()


def rtopo_gebco_30sec(elev_path, save_path):
    """
    Create a zipped and pixel centred 'blend' of RTopo 2.0.4
    and GEBCO[2023] at 30 arc-sec.

    """

    print("Making RTopo-GEBCO (30 arc-sec) blend...")

    data = nc.Dataset(os.path.join(
        elev_path, "RTopo_2_0_4_30sec_pixel.nc"), "r")

    e1st = np.asarray(
        data["bed_elevation"][:], dtype=np.int16)

    i1st = np.asarray(
        data["ice_thickness"][:], dtype=np.int16)

    o1st = np.asarray(
        data["ocn_thickness"][:], dtype=np.int16)

    data = nc.Dataset(os.path.join(
        elev_path, "GEBCO_v2023_30sec_pixel.nc"), "r")

    e2nd = np.asarray(
        data["bed_elevation"][:], dtype=np.int16)

    mask = blend_front(e1st, i1st, e2nd, halo=10, sdev=1.0)

    elev = np.asarray(np.round(
        (1. - mask) * e1st + mask * e2nd), dtype=np.int16)

    iceh = i1st
    ocnh = o1st
    ocnh[i1st == 0] = np.maximum(0, -elev[i1st == 0])

    xpos = np.linspace(
        -180., +180., elev.shape[1] + 1, dtype=np.float64)
    ypos = np.linspace(
        -90.0, +90.0, elev.shape[0] + 1, dtype=np.float64)

    root = nc.Dataset(
        os.path.join(save_path,
                     "RTopo_2_0_4_GEBCO_v2023_30sec_pixel.nc"),
        "w", format="NETCDF4")
    root.description = "Blend of RTopo-2.0.4 (30 arc-sec) " + \
        "and GEBCO[2023] (15 arc-sec) - pixel centred and " + \
        "compressed to int16_t. RTopo data used under ice " + \
        "sheets/shelves. Remapped to 30 arc-sec."
    root.source = \
        "RTopo-2.0.4_30sec_data.nc and GEBCO_2023.nc"
    root.references = \
        "doi.pangaea.de/10.1594/PANGAEA.905295 and " + \
        "doi.org/10.5285/a29c5465-b138-234d-e053-6c86abc040b9"
    root.createDimension("num_lon", elev.shape[1] + 1)
    root.createDimension("num_col", elev.shape[1])
    root.createDimension("num_lat", elev.shape[0] + 1)
    root.createDimension("num_row", elev.shape[0])

    data = root.createVariable("lon", "f8", ("num_lon"))
    data.units = "degrees_east"
    data[:] = xpos
    data = root.createVariable("lat", "f8", ("num_lat"))
    data.units = "degrees_north"
    data[:] = ypos
    data = root.createVariable(
        "bed_elevation", "i2", ("num_row", "num_col"))
    data.units = "m"
    data.long_name = names.bed_elevation
    data[:, :] = elev
    data = root.createVariable(
        "ice_thickness", "i2", ("num_row", "num_col"))
    data.units = "m"
    data.long_name = names.ocn_thickness
    data[:, :] = iceh
    data = root.createVariable(
        "ocn_thickness", "i2", ("num_row", "num_col"))
    data.units = "m"
    data.long_name = names.ice_thickness
    data[:, :] = ocnh

    # filt. grid-scale noise that imprints on dz/dx...
    filt = gaussian_filter(np.asarray(
        elev, dtype=np.float32), sigma=1.25, mode="wrap")

    zslp, dzdx, dzdy = \
        cell_dzdx(xpos, ypos, filt, RSPH)

    data = root.createVariable(
        "bed_slope", "f4", ("num_row", "num_col"))
    data.long_name = names.bed_slope
    data[:, :] = zslp[:, :]
    data = root.createVariable(
        "bed_dz_dx", "f4", ("num_row", "num_col"))
    data.long_name = names.bed_dz_dx
    data[:, :] = dzdx[:, :]
    data = root.createVariable(
        "bed_dz_dy", "f4", ("num_row", "num_col"))
    data.long_name = names.bed_dz_dy
    data[:, :] = dzdy[:, :]

    root.close()


def rtopo_gebco_15sec(elev_path, save_path):
    """
    Create a zipped and pixel centred 'blend' of RTopo 2.0.4
    and GEBCO[2023] at 15 arc-sec.

    """

    print("Making RTopo-GEBCO (15 arc-sec) blend...")

    data = nc.Dataset(os.path.join(
        elev_path, "RTopo_2_0_4_15sec_pixel.nc"), "r")

    elev = np.asarray(
        data["bed_elevation"][:], dtype=np.int16)

    iceh = np.asarray(
        data["ice_thickness"][:], dtype=np.int16)

    data = nc.Dataset(os.path.join(
        elev_path, "GEBCO_v2023_15sec_pixel.nc"), "r")

    e2nd = np.asarray(
        data["bed_elevation"][:], dtype=np.int16)

    mask = blend_front(elev, iceh, e2nd, halo=20, sdev=2.0)

    # careful w mem. alloc.
    del iceh
    elev = np.asarray(elev, dtype=np.float32)
    elev -= mask * elev
    elev += mask * e2nd
    del e2nd
    del mask

    elev = np.asarray(np.round(elev), dtype=np.int16)

    data = nc.Dataset(os.path.join(
        elev_path, "RTopo_2_0_4_15sec_pixel.nc"), "r")

    iceh = np.asarray(
        data["ice_thickness"][:], dtype=np.int16)

    ocnh = np.asarray(
        data["ocn_thickness"][:], dtype=np.int16)

    ocnh[iceh == 0] = np.maximum(0, -elev[iceh == 0])

    xpos = np.linspace(
        -180., +180., elev.shape[1] + 1, dtype=np.float64)
    ypos = np.linspace(
        -90.0, +90.0, elev.shape[0] + 1, dtype=np.float64)

    root = nc.Dataset(
        os.path.join(save_path,
                     "RTopo_2_0_4_GEBCO_v2023_15sec_pixel.nc"),
        "w", format="NETCDF4")
    root.description = "Blend of RTopo-2.0.4 (30 arc-sec) " + \
        "and GEBCO[2023] (15 arc-sec) - pixel centred and " + \
        "compressed to int16_t. RTopo data used under ice " + \
        "sheets/shelves. Remapped to 15 arc-sec."
    root.source = \
        "RTopo-2.0.4_30sec_data.nc and GEBCO_2023.nc"
    root.references = \
        "doi.pangaea.de/10.1594/PANGAEA.905295 and " + \
        "doi.org/10.5285/a29c5465-b138-234d-e053-6c86abc040b9"
    root.createDimension("num_lon", elev.shape[1] + 1)
    root.createDimension("num_col", elev.shape[1])
    root.createDimension("num_lat", elev.shape[0] + 1)
    root.createDimension("num_row", elev.shape[0])

    data = root.createVariable("lon", "f8", ("num_lon"))
    data.units = "degrees_east"
    data[:] = xpos
    data = root.createVariable("lat", "f8", ("num_lat"))
    data.units = "degrees_north"
    data[:] = ypos
    data = root.createVariable(
        "bed_elevation", "i2", ("num_row", "num_col"))
    data.units = "m"
    data.long_name = names.bed_elevation
    data[:, :] = elev
    data = root.createVariable(
        "ice_thickness", "i2", ("num_row", "num_col"))
    data.units = "m"
    data.long_name = names.ice_thickness
    data[:, :] = iceh
    data = root.createVariable(
        "ocn_thickness", "i2", ("num_row", "num_col"))
    data.units = "m"
    data.long_name = names.ocn_thickness
    data[:, :] = ocnh

    # filt. grid-scale noise that imprints on dz/dx...
    filt = gaussian_filter(np.asarray(
        elev, dtype=np.float32), sigma=2.50, mode="wrap")

    del elev
    del ocnh
    del iceh

    zslp, dzdx, dzdy = \
        cell_dzdx(xpos, ypos, filt, RSPH)

    data = root.createVariable(
        "bed_slope", "f4", ("num_row", "num_col"))
    data.long_name = names.bed_slope
    data[:, :] = zslp
    data = root.createVariable(
        "bed_dz_dx", "f4", ("num_row", "num_col"))
    data.long_name = names.bed_dz_dx
    data[:, :] = dzdx
    data = root.createVariable(
        "bed_dz_dy", "f4", ("num_row", "num_col"))
    data.long_name = names.bed_dz_dy
    data[:, :] = dzdy

    root.close()


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--elev-path", dest="elev_path", type=str,
        required=False,
        default="", help="Path to raw DEM data-sets.")

    parser.add_argument(
        "--save-path", dest="save_path", type=str,
        required=False,
        default="", help="Path to store output data.")

    parser.parse_args()
    elev_path = parser.elev_path
    save_path = parser.save_path

    rtopo_60sec(elev_path, save_path)
    rtopo_30sec(elev_path, save_path)
    rtopo_15sec(elev_path, save_path)

    gebco_60sec(elev_path, save_path)
    gebco_30sec(elev_path, save_path)
    gebco_15sec(elev_path, save_path)

    rtopo_gebco_60sec(elev_path, save_path)
    rtopo_gebco_30sec(elev_path, save_path)
    rtopo_gebco_15sec(elev_path, save_path)

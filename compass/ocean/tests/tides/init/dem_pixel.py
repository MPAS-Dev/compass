
import os
import numpy as np
import netCDF4 as nc
from scipy.ndimage import gaussian_filter
import argparse

# Authors: Darren Engwirda


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
        e1st.shape[0], False, dtype=np.bool)
    sidx = np.full(
        e1st.shape[0], False, dtype=np.bool)

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

        print("* blending sweep:", inum)

        part = np.minimum(part, part[npos, :] + 1.)
        part = np.minimum(part, part[spos, :] + 1.)
        part = np.minimum(part, part[:, epos] + 1.)
        part = np.minimum(part, part[:, wpos] + 1.)

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

        print("* blending sweep:", inum)

        part = np.minimum(part, part[npos, :] + 1.)
        part = np.minimum(part, part[spos, :] + 1.)
        part = np.minimum(part, part[:, epos] + 1.)
        part = np.minimum(part, part[:, wpos] + 1.)

    mask[nidx, :] = part

    print("* blending final.")

    mask /= float(halo + 1.00)

    mask = gaussian_filter(mask, sigma=sdev, mode="wrap")
    mask = mask ** 1.50
    mask[i1st >= 1] = 0.

    return mask


def rtopo_60sec(elev_path, save_path):
    """
    Create a zipped and pixel centred version of RTopo 2.0.4
    (60 arc-sec) to support remapping of elevation data.

    """

    print("Making RTopo-2.0.4 (60 arc-sec) pixel...")

    data = nc.Dataset(os.path.join(
        elev_path, "RTopo-2.0.4_1min_data.nc"), "r")

    data.set_auto_maskandscale(False)  # quite valid_min/max

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

    data.set_auto_maskandscale(False)  # quite valid_min/max

    xpos = np.asarray(data["lon"][:], dtype=np.float64)
    ypos = np.asarray(data["lat"][:], dtype=np.float64)

    elev = np.asarray(
        data["bedrock_topography"][:], dtype=np.float32)

    data = nc.Dataset(os.path.join(
        elev_path,
        "RTopo-2.0.4_30sec_surface_elevation.nc"), "r")

    data.set_auto_maskandscale(False)  # quite valid_min/max

    surf = np.asarray(
        data["surface_elevation"][:], dtype=np.float32)

    data = nc.Dataset(os.path.join(
        elev_path,
        "RTopo-2.0.4_30sec_ice_base_topography.nc"), "r")

    data.set_auto_maskandscale(False)  # quite valid_min/max

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


def srtmp_60sec(elev_path, save_path):
    """
    Create a zipped and pixel centred version of SRTM15+V2.1
    (15 arc-sec) at 60 arc-sec.

    """

    print("Making SRTM15+V2.1 (60 arc-sec) pixel...")

    data = nc.Dataset(os.path.join(
        elev_path, "SRTM15+V2.1.nc"), "r")

    elev = np.asarray(data["z"][:], dtype=np.float32)

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
            save_path, "SRTM15+V2.1_60sec_pixel.nc"),
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
        "top_elevation", "i2", ("num_row", "num_col"))
    data.units = "m"
    data[:, :] = elev

    root.close()


def srtmp_30sec(elev_path, save_path):
    """
    Create a zipped and pixel centred version of SRTM15+V2.1
    (15 arc-sec) at 30 arc-sec.

    """

    print("Making SRTM15+V2.1 (30 arc-sec) pixel...")

    data = nc.Dataset(os.path.join(
        elev_path, "SRTM15+V2.1.nc"), "r")

    elev = np.asarray(data["z"][:], dtype=np.float32)

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
            save_path, "SRTM15+V2.1_30sec_pixel.nc"),
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
        "top_elevation", "i2", ("num_row", "num_col"))
    data.units = "m"
    data[:, :] = elev

    root.close()


def rtopo_srtmp_60sec(elev_path, save_path):
    """
    Create a zipped and pixel centred 'blend' of RTopo 2.0.4
    and SRTM15+V2.1 at 60 arc-sec.

    """

    print("Making RTopo-SRTM+ (60 arc-sec) blend...")

    data = nc.Dataset(os.path.join(
        elev_path, "RTopo_2_0_4_60sec_pixel.nc"), "r")

    e1st = np.asarray(
        data["bed_elevation"][:], dtype=np.int16)

    i1st = np.asarray(
        data["ice_thickness"][:], dtype=np.int16)

    o1st = np.asarray(
        data["ocn_thickness"][:], dtype=np.int16)

    data = nc.Dataset(os.path.join(
        elev_path, "SRTM15+V2.1_60sec_pixel.nc"), "r")

    e2nd = np.asarray(
        data["top_elevation"][:], dtype=np.int16)

    mask = blend_front(e1st, i1st, e2nd, halo=20, sdev=2.0)

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
                     "RTopo_2_0_4_SRTM15+V2_1_60sec_pixel.nc"),
        "w", format="NETCDF4")
    root.description = "Blend of RTopo-2.0.4 (60 arc-sec) " + \
        "and SRTM15+V2.1 (60 arc-sec) - pixel centred and " + \
        "compressed to int16_t. RTopo data used under ice " + \
        "sheets/shelves."
    root.source = \
        "RTopo-2.0.4_1min_data.nc and SRTM15+V2.1.nc"
    root.references = \
        "doi.pangaea.de/10.1594/PANGAEA.905295 and " + \
        "doi.org/10.1029/2019EA000658"
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


def rtopo_srtmp_30sec(elev_path, save_path):
    """
    Create a zipped and pixel centred 'blend' of RTopo 2.0.4
    and SRTM15+V2.1 at 30 arc-sec.

    """

    print("Making RTopo-SRTM+ (30 arc-sec) blend...")

    data = nc.Dataset(os.path.join(
        elev_path, "RTopo_2_0_4_30sec_pixel.nc"), "r")

    e1st = np.asarray(
        data["bed_elevation"][:], dtype=np.int16)

    i1st = np.asarray(
        data["ice_thickness"][:], dtype=np.int16)

    o1st = np.asarray(
        data["ocn_thickness"][:], dtype=np.int16)

    data = nc.Dataset(os.path.join(
        elev_path, "SRTM15+V2.1_30sec_pixel.nc"), "r")

    e2nd = np.asarray(
        data["top_elevation"][:], dtype=np.int16)

    mask = blend_front(e1st, i1st, e2nd, halo=40, sdev=4.0)

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
                     "RTopo_2_0_4_SRTM15+V2_1_30sec_pixel.nc"),
        "w", format="NETCDF4")
    root.description = "Blend of RTopo-2.0.4 (30 arc-sec) " + \
        "and SRTM15+V2.1 (30 arc-sec) - pixel centred and " + \
        "compressed to int16_t. RTopo data used under ice " + \
        "sheets/shelves."
    root.source = \
        "RTopo-2.0.4_30sec_data.nc and SRTM15+V2.1.nc"
    root.references = \
        "doi.pangaea.de/10.1594/PANGAEA.905295 and " + \
        "doi.org/10.1029/2019EA000658"
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
    Create a zipped and pixel centred version of GEBCO[2020]
    (15 arc-sec) at 60 arc-sec.

    """

    print("Making GEBCO[2020] (60 arc-sec) pixel...")

    data = nc.Dataset(os.path.join(
        elev_path, "GEBCO_2020.nc"), "r")

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
            save_path, "GEBCO_v2020_60sec_pixel.nc"),
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
        "top_elevation", "i2", ("num_row", "num_col"))
    data.units = "m"
    data[:, :] = elev

    root.close()


def gebco_30sec(elev_path, save_path):
    """
    Create a zipped and pixel centred version of GEBCO[2020]
    (15 arc-sec) at 30 arc-sec.

    """

    print("Making GEBCO[2020] (30 arc-sec) pixel...")

    data = nc.Dataset(os.path.join(
        elev_path, "GEBCO_2020.nc"), "r")

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
            save_path, "GEBCO_v2020_30sec_pixel.nc"),
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
        "top_elevation", "i2", ("num_row", "num_col"))
    data.units = "m"
    data[:, :] = elev

    root.close()


def rtopo_gebco_60sec(elev_path, save_path):
    """
    Create a zipped and pixel centred 'blend' of RTopo 2.0.4
    and GEBCO[2020] at 60 arc-sec.

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
        elev_path, "GEBCO_v2020_60sec_pixel.nc"), "r")

    e2nd = np.asarray(
        data["top_elevation"][:], dtype=np.int16)

    mask = blend_front(e1st, i1st, e2nd, halo=20, sdev=2.0)

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
                     "RTopo_2_0_4_GEBCO_v2020_60sec_pixel.nc"),
        "w", format="NETCDF4")
    root.description = "Blend of RTopo-2.0.4 (60 arc-sec) " + \
        "and GEBCO[2020] (15 arc-sec) - pixel centred and " + \
        "compressed to int16_t. RTopo data used under ice " + \
        "sheets/shelves. Remapped to 60 arc-sec."
    root.source = \
        "RTopo-2.0.4_1min_data.nc and GEBCO_2020.nc"
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


def rtopo_gebco_30sec(elev_path, save_path):
    """
    Create a zipped and pixel centred 'blend' of RTopo 2.0.4
    and GEBCO[2020] at 30 arc-sec.

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
        elev_path, "GEBCO_v2020_30sec_pixel.nc"), "r")

    e2nd = np.asarray(
        data["top_elevation"][:], dtype=np.int16)

    mask = blend_front(e1st, i1st, e2nd, halo=40, sdev=4.0)

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
                     "RTopo_2_0_4_GEBCO_v2020_30sec_pixel.nc"),
        "w", format="NETCDF4")
    root.description = "Blend of RTopo-2.0.4 (30 arc-sec) " + \
        "and GEBCO[2020] (15 arc-sec) - pixel centred and " + \
        "compressed to int16_t. RTopo data used under ice " + \
        "sheets/shelves. Remapped to 30 arc-sec."
    root.source = \
        "RTopo-2.0.4_30sec_data.nc and GEBCO_2020.nc"
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

    """
    SRTM15+V2.1 data seems to include high-freq 'noise'
    near coastlines, so don't use for now...

    srtmp_60sec(elev_path, save_path)
    srtmp_30sec(elev_path, save_path)

    rtopo_srtmp_60sec(elev_path, save_path)
    rtopo_srtmp_30sec(elev_path, save_path)
    """

    gebco_60sec(elev_path, save_path)
    gebco_30sec(elev_path, save_path)

    rtopo_gebco_60sec(elev_path, save_path)
    rtopo_gebco_30sec(elev_path, save_path)

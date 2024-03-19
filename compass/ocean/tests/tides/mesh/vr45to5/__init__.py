import os

import jigsawpy
import netCDF4 as nc
import numpy as np
from mpas_tools.cime.constants import constants
from mpas_tools.logging import check_call
from skimage.filters import farid, gaussian, scharr
from skimage.filters.rank import median, percentile

from compass.mesh.spherical import SphericalBaseStep


class VRTidesMesh(SphericalBaseStep):
    """
    A step for creating a variable resolution tides mesh
    """

    def __init__(self, test_case, pixel, name='base_mesh', subdir=None,
                 elev_file='RTopo_2_0_4_GEBCO_v2023_30sec_pixel.nc',
                 spac_dhdx=0.1, spac_hmin=10.0, spac_hmax=75.0, spac_hbar=60.0,
                 ncell_nwav=60, ncell_nslp=0,
                 filt_sdev=3.0, filt_halo=50, filt_plev=0.325):
        """

        """

        super().__init__(test_case=test_case, name=name, subdir=subdir)

        self.spac_dhdx = spac_dhdx
        self.spac_hmin = spac_hmin
        self.spac_hmax = spac_hmax
        self.spac_hbar = spac_hbar

        self.filt_halo = filt_halo
        self.filt_sdev = filt_sdev
        self.filt_plev = filt_plev

        self.ncell_wav = ncell_nwav
        self.ncell_slp = ncell_nslp

        self.elev_file = elev_file

        pixel_path = pixel.path

        self.add_input_file(
            filename='bathy.nc',
            work_dir_target=f'{pixel_path}/{elev_file}')

    def setup(self):
        """
        Add JIGSAW options based on config options
        """
        section = self.config['spherical_mesh']
        self.opts.mesh_file = section.get('jigsaw_mesh_filename')
        self.opts.geom_file = section.get('jigsaw_geom_filename')
        self.opts.jcfg_file = section.get('jigsaw_jcfg_filename')
        self.opts.hfun_file = section.get('jigsaw_hfun_filename')
        super().setup()

    def run(self):
        """

        """
        spac = self.build_cell_width_lat_lon()
        self.save_and_plot_cell_width(spac.xgrid, spac.ygrid, spac.value)

        self.make_jigsaw_mesh(spac)

        super().run()

    def make_jigsaw_mesh(self, spac):
        """
        Build the JIGSAW mesh.
        """
        logger = self.logger
        earth_radius = constants['SHR_CONST_REARTH']
        opts = self.opts

        jigsawpy.savemsh(opts.hfun_file, spac)

        # define JIGSAW geometry
        geom = jigsawpy.jigsaw_msh_t()
        geom.mshID = 'ELLIPSOID-MESH'
        geom.radii = earth_radius * 1e-3 * np.ones(3, float)
        jigsawpy.savemsh(opts.geom_file, geom)

        jigsawpy.savejig(opts.jcfg_file, opts)
        check_call(['jigsaw', opts.jcfg_file], logger=logger)

    def build_cell_width_lat_lon(self):
        """

        """
        spac = jigsawpy.jigsaw_msh_t()

        # ------------------------------------ define spacing pattern

        dhdx = self.spac_dhdx   # max allowable slope in spacing
        hmin = self.spac_hmin   # min mesh-spacing value (km)
        hmax = self.spac_hmax   # max mesh-spacing value (km)
        hbar = self.spac_hbar   # constant spacing value (km)

        halo = self.filt_halo   # DEM pixels per filtering radii
        sdev = self.filt_sdev   # std-dev for gaussian filter
        plev = self.filt_plev   # median-style filter percentile

        nwav = self.ncell_wav   # number of cells per wavelength
        nslp = self.ncell_slp   # number of cells per grad(elev)

        print("Loading elevation assets...")

        data = nc.Dataset("bathy.nc", "r")

        ocnh = np.asarray(
            data["ocn_thickness"][:])

        print("Computing background h(x)...")

        hmat = np.full(
            (ocnh.shape[:]), (hbar))

        if (nwav >= 1):
            hmat = np.minimum(
                hmat, self.swe_wavelength_spacing(
                    ocnh, nwav, hmin, hmax, halo, plev))

        if (nslp >= 1):
            hmat = np.minimum(
                hmat, self.elev_sharpness_spacing(
                    ocnh, nslp, hmin, hmax, halo, plev,
                    sdev))

        hmat[ocnh <= 0.] = hmax
        hmat = np.maximum(hmat, hmin)
        hmat = np.minimum(hmat, hmax)

        # -- pack h(x) data into jigsaw data-type: average pixel-to-
        # -- node, careful with periodic BC's.

        hmat = self.coarsen_spacing_pixels(hmat, down=4)

        FULL_SPHERE_RADIUS = constants["SHR_CONST_REARTH"] / 1.E+003

        spac.mshID = "ellipsoid-grid"       # use the elv. grid
        spac.radii = np.full(
            3, FULL_SPHERE_RADIUS, dtype=spac.REALS_t)

        spac.xgrid = np.linspace(
            -1. * np.pi, +1. * np.pi, hmat.shape[1] + 1)

        spac.ygrid = np.linspace(
            -.5 * np.pi, +.5 * np.pi, hmat.shape[0] + 1)

        R = hmat.shape[0]
        C = hmat.shape[1]

        spac.value = np.zeros(
            (R + 1, C + 1))

        npos = np.arange(+0, hmat.shape[0] + 1)
        epos = np.arange(-1, hmat.shape[1] - 0)
        spos = np.arange(-1, hmat.shape[0] - 0)
        wpos = np.arange(+0, hmat.shape[1] + 1)

        npos[npos >= +R] = R - 1
        spos[spos <= -1] = +0
        epos[epos <= -1] = C - 1
        wpos[wpos >= +C] = +0

        npos, epos = np.meshgrid(
            npos, epos, sparse=True, indexing="ij")
        spos, wpos = np.meshgrid(
            spos, wpos, sparse=True, indexing="ij")

        spac.value += hmat[npos, epos] * (+1. / 4.)
        spac.value += hmat[npos, wpos] * (+1. / 4.)
        spac.value += hmat[spos, epos] * (+1. / 4.)
        spac.value += hmat[spos, wpos] * (+1. / 4.)

        spac = self.limit_spacing_gradient(spac, dhdx=dhdx)

        return spac

    def limit_spacing_gradient(self, spac, dhdx):

        print("Smoothing h(x) via |dh/dx| limits...")

        opts = jigsawpy.jigsaw_jig_t()

        spac.slope = np.full(spac.value.shape, dhdx)
        opts.hfun_file = os.path.join('.', "spac_pre_smooth.msh")
        opts.jcfg_file = os.path.join('.', "opts_pre_smooth.jig")
        jigsawpy.savemsh(opts.hfun_file, spac)

        opts.verbosity = +1

        jigsawpy.cmd.marche(opts, spac)

        return spac

    def swe_wavelength_spacing(self,
                               ocnh, nwav, hmin, hmax, halo, plev,
                               T_M2=12. * 3600., grav=9.806):

        print("Computing wavelength heuristic...")

        vals = T_M2 * np.sqrt(
            grav * np.maximum(5, ocnh)) / nwav / 1000.

        vals[ocnh <= 0.] = hmax
        vals = np.maximum(vals, hmin)
        vals = np.minimum(vals, hmax)

        vals = np.asarray(vals, dtype=np.uint16)
        # vals = percentile(
        #     vals, footprint=disk(halo), mask=(ocnh>0.), p0=plev)

        return vals

    def elev_sharpness_spacing(self,
                               ocnh, nslp, hmin, hmax, halo, plev, sdev):

        print("Computing GRAD(elev) heuristic...")

        dzdx = scharr(gaussian(np.asarray(
            ocnh, dtype=np.float32), sigma=sdev, mode="wrap"))

        dzdx = np.maximum(1.E-08, dzdx)  # no divide-by-zero

        vals = np.maximum(
            5., np.abs(ocnh)) / dzdx * 2. * np.pi / nslp

        vals = np.maximum(vals, hmin)
        vals = np.minimum(vals, hmax)

        vals = np.asarray(vals, dtype=np.uint16)
        # vals = percentile(
        #     vals, footprint=disk(halo), mask=(ocnh>0.), p0=plev)

        return vals

    def coarsen_spacing_pixels(self, hmat, down):

        print("Coarsening mesh-spacing pixels...")

        rows = hmat.shape[0] // down
        cols = hmat.shape[1] // down

        htmp = np.full(
            (rows, cols), (np.amax(hmat)), dtype=hmat.dtype)

        for jpos in range(down):
            for ipos in range(down):

                iend = hmat.shape[0] - down + ipos + 1
                jend = hmat.shape[1] - down + jpos + 1

                htmp = np.minimum(htmp,
                                  hmat[ipos:iend:down, jpos:jend:down])

        return htmp

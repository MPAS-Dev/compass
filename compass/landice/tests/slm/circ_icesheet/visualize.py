import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from compass.step import Step


class Visualize(Step):
    """
    A step for visualizing the output from a dome test case

    Attributes
    ----------
    """
    def __init__(self, test_case, name='visualize', subdir=None,
                 input_dir='run_model'):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        name : str, optional
            the name of the test case

        subdir : str, optional
            the subdirectory for the step.  The default is ``name``

        input_dir : str, optional
            The input directory within the test case with a file ``output.nc``
            to visualize
        """
        super().__init__(test_case=test_case, name=name, subdir=subdir)

    def setup(self):
        """

        """
        config = self.config
        section = config['circ_icesheet']
        resolutions = section.get('resolutions').split(',')

        for res in resolutions:
            self.add_input_file(filename=f'output_res{res}km.nc',
                                target=f'../{res}km_res/run_model/'
                                'output/output.nc')

    def run(self):
        """
        Run this step of the test case
        """
        logger = self.logger
        config = self.config
        section = config['circ_icesheet']
        resolutions = section.get('resolutions').split(',')

        section = config['slm']
        coupling = section.getboolean('coupling')

        section = config['circ_icesheet_viz']
        save_images = section.getboolean('save_images')
        hide_figs = section.getboolean('hide_figs')

        # visualize run model results
        for res in resolutions:
            run_path = f'../{res}km_res/run_model/'
            logger.info(f'analyzing & visualizing outputs in path: {run_path}')
            visualize_slm_circsheet(config, logger, res)

        # calculate and plot rmse in SLC
        if coupling and len(resolutions) > 1:
            ncells_list = list()
            rmse_list = list()
            for res in resolutions:
                run_data = output_analysis(config, logger,
                                           res, run_path)
                ncells_res = run_data.ncells
                rmse_res = run_data.rmse_slc
                ncells_list.append(ncells_res)
                rmse_list.append(rmse_res)

                ncells = np.array(ncells_list)
                rmse = np.array(rmse_list)

            # plot rmse errors
            p = np.polyfit(np.log10(ncells), np.log10(rmse), 1)
            conv = abs(p[0]) * 2.0
            error_fit = ncells**p[0] * 10**p[1]

            plt.figure(1)
            plt.loglog(ncells, error_fit, 'k')
            plt.loglog(ncells, rmse, 'or')
            plt.annotate('Order of Convergence = {}'.format(np.round(conv, 3)),
                         xycoords='axes fraction', xy=(0.3, 0.95),
                         fontsize=14)
            plt.xlabel('Number of Grid Cells', fontsize=14)
            plt.ylabel('L2 Norm', fontsize=14)
            plt.title('MALI-SLM RMSE in SLC')
            if save_images:
                plt.savefig('RMSE.png', bbox_inches='tight')
                plt.clf()
            if hide_figs:
                logger.info("Plot display disabled with"
                            "hide_plot config option.")
            else:
                plt.show()


def visualize_slm_circsheet(config, logger, res):
    """
    Plot the output from a circular ice sheet test case

    Parameters
    ----------
    config : compass.config.CompassConfigParser
        Configuration options for this test case,
        a combination of the defaults for the machine,
        core and configuration

    logger : logging.Logger
        A logger for output from the step

    res : str
        resolution of MALI mesh
    """
    section = config['circ_icesheet_viz']
    save_images = section.getboolean('save_images')
    hide_figs = section.getboolean('hide_figs')

    section = config['slm']
    coupling = section.getboolean('coupling')
    slm_res = int(section.get('slm_res'))

    # get an instance of output analysis class
    run_path = f'../{res}km_res/run_model/'
    run_data = output_analysis(config, logger, res, run_path)
    yrs = run_data.yrs

    # figure 1
    fig1 = plt.figure(1, facecolor='w')
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.plot(yrs, run_data.grnd_vol_unscaled,
             label='MALI unscaled', linestyle=':', color='k')
    ax1.plot(yrs, run_data.grnd_vol, label='MALI scaled',
             linestyle='-', color='k')
    if coupling:
        ax1.plot(yrs, run_data.ice_vol_slm, label='SLM',
                 linestyle='-.', color='r')
        ax1.set_xlabel('Year')
        ax1.legend(loc='best', ncol=1, prop={'size': 10})
        ax1.set_title('Grounded Ice Mass')
        ax1.set_ylabel('Mass (Gt)')
        ax1.grid(True)
    if save_images:
        plt.savefig(f'ice_mass_mali{res}km_slm{slm_res}.png')
        fig1.clf()

    # figure 2
    fig2 = plt.figure(2, facecolor='w')
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(yrs, run_data.dgrnd_vol_unscaled,
             label='MALI unscaled', linestyle=':', color='k')
    ax2.plot(yrs, run_data.dgrnd_vol, label='MALI scaled',
             linestyle='-', color='k')
    if coupling:
        ax2.plot(yrs, run_data.dice_vol_slm, label='SLM',
                 linestyle='-.', color='r')
        ax2.set_xlabel('Year')
        ax2.legend(loc='best', ncol=1, prop={'size': 10})
        ax2.set_title('Change in Grounded Ice Mass')
        ax2.set_ylabel('Mass (Gt)')
        ax2.grid(True)
    if save_images:
        plt.savefig(f'ice_mass_change_mali{res}km_slm{slm_res}.png')
        fig2.clf()

    # figure 3
    fig3 = plt.figure(3, facecolor='w')
    ax3 = fig3.add_subplot(1, 1, 1)
    ax3.plot(yrs, run_data.SLCaf, label='MALI SLC VAF',
             linestyle=':', color='k')
    ax3.plot(yrs, run_data.SLCcorr_Aocn,
             label='MALI SLCcorr (grnded ice flooded)',
             linestyle='-.', color='k')
    ax3.plot(yrs, run_data.SLCcorr_AocnBeta,
             label='MALI SLCcorr (grnded ice NOT flooded)',
             linestyle='-', color='b')
    ax3.plot(yrs, run_data.SLCcorr_z0_AocnBeta,
             label='MALI SLCcorr+z0 (grnded ice NOT flooded)',
             linestyle='-', color='y')
    if coupling:
        ax3.plot(yrs, run_data.SLC_slm_Aocn,
                 label='SLM (grnded ice flooded)', linestyle='-.', color='r')
        ax3.plot(yrs, run_data.SLC_slm_AocnBeta,
                 label='SLM (grnded ice NOT flooded)', linestyle='-',
                 color='r')
        ax3.set_xlabel('Year')
        ax3.legend(loc='best', ncol=1, prop={'size': 10})
        ax3.set_title('Ice-sheet contribution to SLC')
        ax3.set_ylabel('Sea-level change (m)')
        ax3.grid(True)
    if save_images:
        plt.savefig(f'ice_contribution_to_SLC_mali{res}km_slm{slm_res}.png')
        fig3.clf()

    # figure 4 & 5
    if coupling:
        fig4 = plt.figure(4, facecolor='w')
        ax4 = fig4.add_subplot(1, 1, 1)
        ax4.plot(yrs, run_data.dgrnd_vol - run_data.dice_vol_slm,
                 label='MALI minus SLM', linestyle='-', color='k')
        ax4.set_xlabel('Year')
        ax4.legend(loc='best', ncol=1, prop={'size': 10})
        ax4.set_title('Difference in ice mass change')
        ax4.set_ylabel('Mass (Gt)')
        ax4.grid(True)
        if save_images:
            plt.savefig(f'diff_ice_mass_change_mali{res}km_slm{slm_res}.png')
            fig4.clf()

        fig5 = plt.figure(5, facecolor='w')
        ax5 = fig5.add_subplot(1, 1, 1)
        ax5.plot(yrs, run_data.diff_slc, label='MALI minus SLM',
                 linestyle='-', color='k')
        ax5.plot(yrs, run_data.diff_slc_z0, label='MALI minus SLM (z0)',
                 linestyle='-', color='k')
        ax5.set_xlabel('Year')
        ax5.legend(loc='best', ncol=1, prop={'size': 10})
        ax5.set_title('Difference in sea-level change')
        ax5.set_ylabel('sea-level change (m)')
        ax5.grid(True)
        if save_images:
            plt.savefig(f'diff_sea_level_change_mali{res}km_slm{slm_res}.png')
            fig5.clf()

    if hide_figs:
        logger.info("Plot display disabled with hide_plot config option.")
    else:
        plt.show()


class output_analysis:
    """
    Analyze outputs

    Attributes
    ----------
    mali_slc : float
       ice-sheet contribution to sea-level change calculated
       based on MALI outputs

    slm_slc : float
       ice-sheet contribution to sea-level change calculated
       based on SLM outputs

    rmse : float
       root mean square error between mali_slc and slm_slc
    """
    def __init__(self, config, logger, res, run_path):
        """
        Calculate sea-level change from run outputs

        Parameters
        ----------
        config : compass.config.CompassConfigParser
            Configuration options for this test case,
            a combination of the defaults for the machine,
            core and configuration

        logger : logging.Logger
            A logger for output from the step

        res : str
            Resolution of MALI mesh

        run_path : str
            Path to runs where the output file exists

        coupling : str
            Whether MALI is coupled to the SLM
        """
        self.config = config
        self.logger = logger
        self.res = res
        self.run_path = run_path

        section = config['slm']
        coupling = section.getboolean('coupling')

        if coupling:
            section = config['circ_icesheet_viz']
            Aocn_const = section.getfloat('Aocn_const')
            AocnBeta_const = section.getfloat('AocnBeta_const')

        # mali output file name
        fname_mali = f'output_res{res}km.nc'

        # read in the MALI outputs
        DS = xr.open_mfdataset(fname_mali, combine='nested',
                               concat_dim='Time',
                               decode_timedelta=False)

        # default constants
        rhoi = 910.0   # ice density in kg/m^3
        rhoo = 1000.0  # ocean density used by the SLM (MALI uses 1028.0)
        rhow = 1000.0  # fresh water density
        Aocn_const = 4.5007E+14  # area of global ocean in m2
        AocnBeta_const = 4.5007E+14  # ocean area including marine-based area

        self.ncells = DS.dims['nCells']
        cellMask = DS['cellMask']
        bed = DS['bedTopography']
        # bed0 = bed.isel(Time=0).load()
        thickness = DS['thickness']
        areaCell = DS['areaCell'][0, :].load()
        latCell = DS['latCell'][0, :].load()
        lonCell = DS['lonCell'][0, :].load() - np.pi  # correct range: [-pi pi]
        self.lat_deg = latCell * 180 / np.pi

        # calculate the area distortion (scale) factor for the
        # polar stereographic projection
        self.k = np.zeros(len(latCell),)
        k0 = (1 - np.sin(-71 * np.pi / 180)) / 2
        # center of the latitude
        lat0 = -90 * np.pi / 180
        # standard parallel in radians where distortion should be zero (k=1)
        lon0 = 0  # center of the longitude
        # expression for 'k' from p. 157
        # eqn. 21-4 in https://pubs.usgs.gov/pp/1395/report.pdf.
        # p. 142 provides a table showing 'k' for stereographic projection
        self.k[:] = 2 * k0 / (1 + (np.sin(lat0) * np.sin(latCell[:])) +
                              (np.cos(lat0) * np.cos(latCell[:]) *
                               np.cos(lonCell[:] - lon0)))

        # default MALI time steps from the MALI outputs
        self.yrs_mali = DS['daysSinceStart'].load() / 365.0 + 2015.0
        if coupling:
            # path to the SLM output data
            fpath_slm = os.path.join(run_path, 'OUTPUT_SLM/')
            config = self.config
            section = config['slm']
            time_stride = int(section.get('time_stride'))
            # reset the time indices to match the # of SLM timesteps
            nt = np.arange(0, DS.dims['Time'], time_stride, dtype='i')
            self.yrs = self.yrs_mali[nt]
            z0 = np.zeros((len(self.yrs), ))
            Aocn = np.zeros((len(self.yrs), ))
            AocnBeta = np.zeros((len(self.yrs), ))
            # get ice mass on the SLM interpolated from the MALI mesh
            fname = os.path.join(fpath_slm, 'ice_volume')
            self.ice_vol_slm = slm_outputs(fname).data * rhoi / 1.0e12
            self.dice_vol_slm = slm_outputs(fname).change_total * \
                rhoi / 1.0e12  # in Gt
            # get slc correction and ocean area from the SLM
            fname = os.path.join(fpath_slm, 'gmsle_deltaSL_Ocean_fixed')
            if os.path.exists(fname):
                logger.info(f'reading in file {fname}')
                z0 = slm_outputs(fname).change_total
                self.SLC_slm_Aocn = slm_outputs(fname).data

            fname = os.path.join(fpath_slm, 'gmsle_deltaSL_OceanBeta_fixed')
            if os.path.exists(fname):
                logger.info(f'reading in file {fname}')
                self.SLC_slm_AocnBeta = slm_outputs(fname).data

            fname = os.path.join(fpath_slm, 'ocean_area')
            if os.path.exists(fname):
                logger.info(f'reading in file {fname}')
                Aocn = slm_outputs(fname).data
                logger.info(f'area of the ocean is: {Aocn}')
            fname = os.path.join(fpath_slm, 'oceanBeta_area')
            if os.path.exists(fname):
                logger.info(f'reading in file {fname}')
                AocnBeta = slm_outputs(fname).data
                logger.info(f'area of the ocean Beta is: {AocnBeta}')
        else:
            # if not coupled, use default values for MALI outputs
            # assuming MALI output interval is 1 year
            nt = np.arange(0, DS.dims['Time'], 1, dtype='i')
            self.yrs = self.yrs_mali
            z0 = np.zeros((len(self.yrs), ))
            Aocn = np.zeros((len(self.yrs), ))
            AocnBeta = np.zeros(len(self.yrs, ))

            z0[:] = 0.0
            logger.info("'gmsle_change' file doesn't exist. "
                        "Setting z0 to zeros")
            Aocn[:] = Aocn_const
            logger.info("'ocean_area' file doesn't exist. Using "
                        f"constant ocean area defined: {Aocn_const}m2")
            AocnBeta[:] = AocnBeta_const
            logger.info("'ocean_areaBeta' file doesn't exist. Using"
                        f"constant oceanBeta area defined: {AocnBeta_const}m2")

        # calculate ice-sheet contribution to sea-level change
        # create empty arrays
        self.grnd_vol = np.zeros((len(nt), ))
        self.grnd_vol_unscaled = np.zeros((len(nt), ))
        self.dgrnd_vol = np.zeros((len(nt), ))
        self.dgrnd_vol_unscaled = np.zeros((len(nt), ))
        self.vaf = np.zeros((len(nt), ))
        self.vaf_z0 = np.zeros((len(nt), ))
        self.pov = np.zeros((len(nt), ))
        self.pov_z0 = np.zeros((len(nt), ))
        self.den = np.zeros((len(nt), ))
        self.SLEaf = np.zeros((len(nt), ))
        self.SLCaf = np.zeros((len(nt), ))
        self.SLCpov = np.zeros((len(nt), ))
        self.SLCden = np.zeros((len(nt), ))
        self.SLCcorr_Aocn = np.zeros((len(nt), ))
        self.SLEaf_AocnBeta = np.zeros((len(nt), ))
        self.SLEaf_AocnBeta = np.zeros((len(nt), ))
        self.SLCaf_AocnBeta = np.zeros((len(nt), ))
        self.SLCpov_AocnBeta = np.zeros((len(nt), ))
        self.SLCden_AocnBeta = np.zeros((len(nt), ))
        self.SLCcorr_AocnBeta = np.zeros((len(nt), ))
        self.SLEaf_z0_AocnBeta = np.zeros((len(nt), ))
        self.SLCaf_z0_AocnBeta = np.zeros((len(nt), ))
        self.SLCpov_z0_AocnBeta = np.zeros((len(nt), ))
        self.SLCcorr_z0_AocnBeta = np.zeros((len(nt), ))

        for t in np.arange(0, len(nt)):
            # get appropriate time index for the MALI output data
            if coupling:
                indx = t * time_stride
            else:
                indx = t

            bedt = bed[indx, :].load()
            cellMaskt = cellMask[indx, :].load()
            thicknesst = thickness[indx, :].load()
            areaCellt = areaCell.sum()
            logger.info(f'area of the mali domain is {areaCellt}')

            # calculation of sea-level change following Goelzer et al. 2020
            self.grnd_vol_unscaled[t] = (areaCell * grounded(cellMaskt) *
                                         thicknesst).sum() * rhoi / 1.0e12
            self.grnd_vol[t] = (areaCell * grounded(cellMaskt) *
                                thicknesst / (self.k**2)).sum() * rhoi / 1.0e12
            self.vaf[t] = (areaCell * grounded(cellMaskt) / (self.k**2) *
                           np.maximum(thicknesst + (rhoo / rhoi) *
                                      np.minimum(np.zeros((self.ncells,)),
                                                 bedt), 0.0)).sum()  # eqn. 1
            self.pov[t] = (areaCell / (self.k**2) *
                           np.maximum(0.0 * bedt, -1.0 * bedt)).sum()  # eqn.8
            self.vaf_z0[t] = (areaCell * grounded(cellMaskt) / (self.k**2) *
                              np.maximum(thicknesst + (rhoo / rhoi) *
                              np.minimum(np.zeros((self.ncells,)),
                                         bedt + z0[t]), 0.0)).sum()  # eqn. 13
            self.pov_z0[t] = (areaCell / (self.k**2) * np.maximum(0.0 * bedt,
                              -1.0 * bedt + z0[t])).sum()  # eqn. 14
            self.den[t] = (areaCell / (self.k**2) * grounded(cellMaskt) *
                           (rhoi / rhow - rhoi / rhoo)).sum()  # eqn. 10

            # SLC(m) using the ocean area where anywhere below sea level
            # is considered as ocean (even where marine-based ice exists)
            self.SLEaf[t] = (self.vaf[t] / Aocn[t]) * (rhoi / rhoo)  # eqn. 2
            self.SLCaf[t] = -1.0 * (self.SLEaf[t] - self.SLEaf[0])  # eqn. 3
            self.SLCpov[t] = -1.0 * (self.pov[t] / Aocn[t] - self.pov[0] /
                                     Aocn[t])  # eqn. 9
            self.SLCden[t] = -1.0 * (self.den[t] / Aocn[t] - self.den[0] /
                                     Aocn[t])  # eqn. 11
            self.SLCcorr_Aocn[t] = self.SLCaf[t] + self.SLCpov[t] + \
                self.SLCden[t]

            # SLC(m) including the z0 term using the ocean+iced area
            # i.e. water can't go where marine-based ice exists
            self.SLEaf_AocnBeta[t] = (self.vaf[t] / AocnBeta[t]) * \
                                     (rhoi / rhoo)
            self.SLCaf_AocnBeta[t] = -1.0 * (self.SLEaf_AocnBeta[t] -
                                             self.SLEaf_AocnBeta[0])
            self.SLCpov_AocnBeta[t] = -1.0 * (self.pov[t] / AocnBeta[t] -
                                              self.pov[0] / AocnBeta[t])
            self.SLCden_AocnBeta[t] = -1.0 * (self.den[t] / AocnBeta[t] -
                                              self.den[0] / AocnBeta[t])
            self.SLCcorr_AocnBeta[t] = self.SLCaf_AocnBeta[t] + \
                self.SLCpov_AocnBeta[t] + \
                self.SLCden_AocnBeta[t]

            # same as above but adding the eustatic sea-level change term 'z0'
            self.SLEaf_z0_AocnBeta[t] = (self.vaf_z0[t] / AocnBeta[t]) * \
                                        (rhoi / rhoo)
            self.SLCaf_z0_AocnBeta[t] = -1.0 * (self.SLEaf_z0_AocnBeta[t] -
                                                self.SLEaf_z0_AocnBeta[0])
            self.SLCpov_z0_AocnBeta[t] = -1.0 * (self.pov_z0[t] / AocnBeta[t] -
                                                 self.pov_z0[0] / AocnBeta[t])
            self.SLCcorr_z0_AocnBeta[t] = self.SLCaf_z0_AocnBeta[t] + \
                self.SLCpov_z0_AocnBeta[t] + \
                self.SLCden_AocnBeta[t]

            # get total ice mass change at each time step
            self.dgrnd_vol[t] = (self.grnd_vol[t] - self.grnd_vol[0])
            self.dgrnd_vol_unscaled[t] = (self.grnd_vol_unscaled[t] -
                                          self.grnd_vol_unscaled[0])

            DS.close()

        # calculate RMSE between MALI and SLM calculation of SLC
        if coupling:
            self.diff_slc = self.SLCcorr_AocnBeta - self.SLC_slm_AocnBeta
            self.diff_slc_z0 = self.SLCcorr_z0_AocnBeta - self.SLC_slm_AocnBeta
            self.rmse_slc = np.sqrt((self.diff_slc**2).mean())
        else:
            logger.info('MALI is not coupled to the SLM. No RMSE to compute.')
            self.rmse_slc = np.nan


def grounded(cellMask):
    # return ((cellMask&32)//32) & ~((cellMask&4)//4)
    return ((cellMask & 32) // 32) * np.logical_not((cellMask & 4) // 4)


class slm_outputs:
    """
    Read and calculate the SLM outputs

    Attributes
    ----------
    data : float
        SLM text-formatted outputs

    change_dt : float
        Change in data value between each time step

    change_total : float
        Change in data value at a time step
        with respect to the intial time step
    """
    def __init__(self, filename):
        """
        Calculate the sea-level model output file

        Parameters
        ----------
        filename : str
            Filename of SLM output data
        """
        self.fname = filename
        self.data = np.loadtxt(self.fname)
        self.change_dt = np.zeros(len(self.data),)
        self.change_total = np.zeros(len(self.data),)
        for i in range(len(self.change_dt) - 1):
            self.change_dt[i + 1] = (float(self.data[i + 1]) -
                                     float(self.data[i]))
            self.change_total[i + 1] = (float(self.data[i + 1]) -
                                        float(self.data[0]))

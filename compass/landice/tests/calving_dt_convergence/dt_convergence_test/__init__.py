from compass.testcase import TestCase
from compass.landice.tests.calving_dt_convergence.run_model import RunModel
# from compass.validate import compare_variables  # not currently used
import numpy
import netCDF4
import matplotlib.pyplot as plt
import matplotlib.cm


class DtConvergenceTest(TestCase):
    """
    A test case for running the same configuration with a series of
    values for config_adaptive_timestep_calvingCFL_fraction
    to check for convergence
    """

    def __init__(self, test_group, mesh, calving, velo):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.landice.tests.mismipplus.MISMIPplus
            The test group that this test case belongs to

        """
        name = f'calving_dt_convergence_test_{mesh}_{calving}_{velo}'
        subdir = f'{mesh}.{calving}.{velo}'
        super().__init__(test_group=test_group, name=name, subdir=subdir)

        cores = 36
        min_cores = 4

        # Do fewer runs if FO solver
        if velo == 'FO':
            self.fractions = numpy.arange(0.25, 2.3, 0.25)
        else:
            self.fractions = numpy.arange(0.2, 3.1, 0.2)

        for frac in self.fractions:
            name = f'frac{frac:.2f}'
            step = RunModel(test_case=self, name=name,
                            mesh=mesh,
                            calving=calving,
                            velo=velo,
                            calv_dt_frac=frac,
                            cores=cores, min_cores=min_cores, threads=1)
            self.add_step(step)

    # no configure() method is needed

    # no run() method is needed

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        # variables = ['thickness', 'surfaceSpeed']
        # compare_variables(test_case=self, variables=variables,
        #                   filename1='full_run/output.nc',
        #                   filename2='restart_run/output.nc')

        # plot results
        fig, ax = plt.subplots(4, figsize=(10, 7))
        ax[0].set(xlabel='year', ylabel='calving flux (kg/yr)')
        ax[1].set(xlabel='year', ylabel='cum. calving flux (kg)')
        ax[2].set(xlabel='year', ylabel='actual dt to calving dt ratio')
        ax[3].set(xlabel='fraction', ylabel='# warnings')
        colors = matplotlib.cm.jet(numpy.linspace(0, 1, len(self.fractions)))
        nWarn = numpy.zeros([len(self.fractions)])

        i = 0
        for frac in self.fractions:
            name = f'frac{frac:.2f}'
            f = netCDF4.Dataset(f'{name}/globalStats.nc', 'r')
            yr = f.variables['daysSinceStart'][:] / 365.0
            calv = f.variables['totalCalvingFlux'][:]
            deltat = f.variables['deltat'][:]
            ax[0].plot(yr[1:], calv[1:], '-', label=f'{frac:.2f}',
                       color=colors[i])

            ax[1].plot(yr[1:], (calv[1:]*deltat[1:]).cumsum(), '-',
                       color=colors[i])

            ratio = f.variables['dtCalvingCFLratio'][:]
            ax[2].plot(yr[1:], numpy.ones(yr[1:].shape) * frac, 'k:',
                       label=f'{frac:.2f}')
            ax[2].plot(yr[1:], ratio[1:], '-', label=f'{frac:.2f}',
                       color=colors[i])

            # Now count errors
            file = open(f"{name}/log.landice.0000.out", "r")
            logcontents = file.read()
            # get number of occurrences of the substring in the string
            nWarn[i] = logcontents.count("WARNING: Failed to ablate")
            ax[3].plot(frac, nWarn[i], 'ko')

            f.close()
            i += 1

        ax[0].legend(loc='best', prop={'size': 5})
        plt.savefig('calving_dt_comparison.png', dpi=150)

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from compass.step import Step


class Analysis(Step):
    """
    A step for verifying that temperature and salinity profiles for SOMA case
    from particle interpolations are consistent with initialized profiles

    Attributes
    ----------
    resolution : str
        The resolution of the test case
    """
    def __init__(self, test_case, resolution):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.ocean.tests.soma.default.Default
            The test case this step belongs to

        resolution : str
            The resolution of the test case
        """
        super().__init__(test_case=test_case, name='analysis')
        self.resolution = resolution

        self.add_input_file(
            filename='particle_output.nc',
            target='../forward/analysis_members/lagrPartTrack.0001-01-01_00.00.00.nc')

        self.add_output_file('particle_temperature_comparison.png')
        self.add_output_file('particle_salinity_comparison.png')

    def run(self):
        """
        Run this step of the test case
        """
        plt.switch_backend('Agg')

        ds = xr.open_mfdataset('particle_output.nc')

        # obtain particle data
        zLevelParticle = ds.zLevelParticle.values
        particleTemperature = ds.particleTemperature.values
        particleSalinity = ds.particleSalinity.values

        section = self.config['soma']
        alpha = section.getfloat('eos_linear_alpha')
        drho = section.getfloat('density_difference')
        t0 = section.getfloat('surface_temperature')
        s0 = section.getfloat('surface_salinity')
        gamma = section.getfloat('salinity_gradient')
        beta = section.getfloat('density_difference_linear')
        zt = section.getfloat('thermocline_depth')
        bottom_depth = section.getfloat('bottom_depth')

        z = np.linspace(0, zLevelParticle.min(), 30)

        # compute profiles
        salinity = s0 - gamma * z
        temperature = ((1 - beta) * drho * np.tanh(z / zt)
                       + beta * drho * (z / bottom_depth)) / alpha + t0

        # temperature comparison
        plt.figure()
        plt.plot(particleTemperature.ravel(), zLevelParticle.ravel(), '.',
                 label='Particle value')
        plt.plot(temperature, z, '-', label='Initialized scalar value')
        plt.ylabel('Depth (m)')
        plt.xlabel(r'Temperature $^\circ$C')
        plt.title('Particle temperature comparison')
        plt.savefig('particle_temperature_comparison.png')

        # salinity comparison
        plt.figure()
        plt.plot(particleSalinity.ravel(), zLevelParticle.ravel(), '.',
                 label='Particle value')
        plt.plot(salinity, z, '-', label='Initialized scalar value')
        plt.title('Particle salinity comparison')
        plt.ylabel('Depth (m)')
        plt.xlabel('Salinity psu')
        plt.savefig('particle_salinity_comparison.png')

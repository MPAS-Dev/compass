import time

from compass.model import run_model
from compass.step import Step
from datetime import timedelta


class Forward(Step):
    """
    A step for performing forward MPAS-Ocean runs as part of the rotation2D

    Attributes
    ----------
    resolution : int
        The resolution of the (uniform) mesh in km
    """

    def __init__(self, test_case, resolution, dt_minutes):
        """
        Create a new step

        Parameters
        ----------
        test_case : compass.ocean.tests.global_convergence.rotation2D.Rotation2D
            The test case this step belongs to

        resolution : int
            The resolution of the (uniform) mesh in km
        dt_minutes : int
            The time step size in minutes.  **must divide 1 day (24*60)**
        """
        super().__init__(test_case=test_case,
                         name='QU{}_forward'.format(resolution),
                         subdir='QU{}/forward'.format(resolution))

        self.resolution = resolution
        self.dt_minutes = dt_minutes

        self.add_namelist_file(
            'compass.ocean.tests.global_convergence.rotation2D',
            'namelist.forward')
        self.add_streams_file(
            'compass.ocean.tests.global_convergence.rotation2D',
            'streams.forward')

        self.add_input_file(filename='init.nc',
                            target='../init/initial_state.nc')
        self.add_input_file(filename='graph.info',
                            target='../mesh/graph.info')

        self.add_model_as_input()

        self.add_output_file(filename='output.nc')

    def setup(self):
        """
        Set namelist options base on config options
        """
        config = self.config
        dtstr = self.get_timestep_config_str()
        self.add_namelist_options({'config_dt': dtstr,
          'config_time_integrator':config.get('rotation2D', 'time_integrator')})

    def run(self):
        """
        Run this step of the testcase
        """
        config = self.config
        dt = self.get_timestep_str(self.dt_minutes)
        self.update_namelist_at_runtime(options={'config_dt': dt,
          'config_time_integrator':config.get('rotation2D', 'time_integrator')},
          out_name='namelist.ocean')

        run_model(self)

    def get_timestep_str(dtminutes):
        """
          These tests expect the time step to be input in units of minutes, but MPAS
          requires an "HH:MM:SS" string.  This function converts the time step input
          into the formatted string used by MPAS.
        """
        dt = timedelta(minutes=dtminutes)
        if  dtminutes < 1:
           dtstr = "00:00:" + str(dt.total_seconds())[:2]
        elif dtminutes >= 60:
           dthours = dt/timedelta(hours=1)
           dt = dt - timedelta(hours=int(dthours))
           dtminutes = dt/timedelta(minutes=1)
           dt = dt - timedelta(minutes=int(dtminutes))
           dtstr = str(int(dthours))[:2].zfill(2) + ":" + str(int(dtminutes))[:2].zfill(2) +\
           ":"+str(int(dt.total_seconds()))[:2].zfill(2)
        else:
           dtminutes = dt/timedelta(minutes=1)
           dtstr = "00:" + str(int(dtminutes))[:2].zfill(2) + ":00"
        return dtstr

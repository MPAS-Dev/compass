from compass.model import ModelStep
from datetime import timedelta


class Forward(ModelStep):
    """
    A step for performing forward MPAS-Ocean runs as part of the
      nondivergent_2d test case.

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
        test_case : compass.ocean.tests.global_convergence.nondivergent_2d.Nondivergent2D
            The test case this step belongs to

        resolution : int
            The resolution of the (uniform) mesh in km
        dt_minutes : int
            The time step size in minutes.  **must divide 1 day (24*60)**
        """
        super().__init__(test_case=test_case, openmp_threads=1,
                         name=f'QU{resolution}_forward',
                         subdir=f'QU{resolution}/forward')

        self.resolution = resolution
        self.dt_minutes = dt_minutes

        package = 'compass.ocean.tests.sphere_transport.nondivergent_2d'

        self.add_namelist_file(package, 'namelist.forward', mode='forward')
        self.add_streams_file(package, 'streams.forward', mode='forward')

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
        dtstr = self.get_timestep_str()
        self.add_namelist_options({'config_dt': dtstr,
                                   'config_time_integrator': config.get(
                                       'nondivergent_2d', 'time_integrator')})

    def runtime_setup(self):
        """
        Update the resources and time step in case the user has update config
        options
        """
        super().runtime_setup()
        config = self.config
        dt = self.get_timestep_str()
        self.update_namelist_at_runtime(
            options={
                'config_dt': dt,
                'config_time_integrator': config.get(
                    'nondivergent_2d',
                    'time_integrator')},
            out_name='namelist.ocean')

    def get_timestep_str(self):
        """
          These tests expect the time step to be input in units of minutes,
          but MPAS requires an "HH:MM:SS" string.  This function converts the
          time step input into the formatted string used by MPAS.
        """
        dtminutes = self.dt_minutes
        dt = timedelta(minutes=dtminutes)
        if dtminutes < 1:
            dtstr = "00:00:" + str(dt.total_seconds())[:2]
        elif dtminutes >= 60:
            dthours = dt / timedelta(hours=1)
            dt = dt - timedelta(hours=int(dthours))
            dtminutes = dt / timedelta(minutes=1)
            dt = dt - timedelta(minutes=int(dtminutes))
            dtstr = str(int(dthours))[:2].zfill(2) + ":" + str(int(dtminutes))[
                :2].zfill(2) + ":" + str(int(dt.total_seconds()))[:2].zfill(2)
        else:
            dtminutes = dt / timedelta(minutes=1)
            dtstr = "00:" + str(int(dtminutes))[:2].zfill(2) + ":00"
        return dtstr

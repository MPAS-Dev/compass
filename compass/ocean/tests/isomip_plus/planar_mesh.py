from mpas_tools.planar_hex import make_planar_hex_mesh
from mpas_tools.translate import translate
from mpas_tools.io import write_netcdf

from compass.step import Step


class PlanarMesh(Step):
    """
    A step for creating a planar ISOMIP+ mesh and culling out land and
    grounded-ice cells

    Attributes
    ----------
    thin_film_present: bool
        Whether the run includes a thin film below grounded ice
    """
    def __init__(self, test_case, thin_film_present):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        thin_film_present: bool
            Whether the run includes a thin film below grounded ice
        """
        super().__init__(test_case=test_case, name='planar_mesh')
        self.thin_film_present = thin_film_present
        self.add_output_file('base_mesh.nc')

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        section = config['isomip_plus']
        nx = section.getint('nx')
        nx_thin_film = section.getint('nx_thin_film')
        ny = section.getint('ny')
        dc = section.getfloat('dc')

        thin_film_present = self.thin_film_present
        # Add xOffset to reduce distance between x=0 and start of GL
        if thin_film_present:
            nx_offset = nx_thin_film
            # consider increasing nx
            ds_mesh = make_planar_hex_mesh(nx=nx+nx_offset, ny=ny, dc=dc,
                                           nonperiodic_x=True,
                                           nonperiodic_y=True)
        else:
            nx_offset = 0
            ds_mesh = make_planar_hex_mesh(nx=nx+2, ny=ny+2, dc=dc,
                                           nonperiodic_x=False,
                                           nonperiodic_y=False)

        translate(mesh=ds_mesh, xOffset=-1*nx_offset*dc, yOffset=-2*dc)

        write_netcdf(ds_mesh, 'base_mesh.nc')

import numpy
import cartopy

class topo:
    """
    A class to aid in playing with the design of idealized domains
    """
    
    def __init__(self, nj, ni, dlon=1, dlat=1, lon0=0, lat0=0, D=1):
        """
        Create a topo object with a mesh of nj*ni cells with depth -D
        on a mesg ranging from lon0..lon0+dlon and lat0..lat0+dlat.
        
        By default D=1, dlon=1, dlat=1, lon0=0 and lat0=0.
        """
        self.z = -D * numpy.ones((nj,ni))
        self.D0 = D # Nominal deepest depth
        # Coordinates of grid nodes (0..1)
        self.xg = numpy.arange(ni+1)/ni * dlon + lon0
        self.yg = numpy.arange(nj+1)/nj * dlat + lat0
        # Coordinates of cell centers (0..1)
        self.xc = (numpy.arange(ni)+0.5)/ni * dlon + lon0
        self.yc = (numpy.arange(nj)+0.5)/nj * dlat + lat0
        # Store 2D arrays of coordinates
        self.XG, self.YG = numpy.meshgrid(self.xg, self.yg)
        self.XC, self.YC = numpy.meshgrid(self.xc, self.yc)
    # Some 1D functions to generate simple shapes
    def heaviside(x, x0):
        """Returns 0 for x < x0, 1 or x >= x0"""
        b = 0*x
        b[x>=x0] = 1
        return b
    def box(x, x0, x1):
        """Returns 0 for x < x0, 1 or x0 <= x <= x1, 0 for x > x1"""
        return topo.heaviside(x, x0) * topo.heaviside(-x, -x1)
    def cone(x, x0, dx):
        """Returns 0 for |x-x0| > dx, straight lines peaking at x = x0"""
        return numpy.maximum(0, 1. - numpy.abs(x-x0)/dx)
    def clipped_cone(x, x0, dx, clip):
        """Returns a cone clipped at height 'clip'"""
        return numpy.minimum(clip, topo.cone(x, x0, dx))
    def scurve(x, x0, dx):
        """Returns 0 for x<x0 or x>x+dx, and a cubic in between."""
        s = numpy.minimum(1, numpy.maximum(0, (x-x0)/dx))
        return (3 - 2*s)*( s*s )
    # Actual coastal profile
    def coastal_sprofile(x, x0, dx, shelf, lf=.125, bf=.125, sf=.5):
        """A 'coastal profile' with coastal shelf and slope.
        Of profile width dx:
          - lf is the land fraction (value 0)
          - bf is the fraction that is the beach slope.
          - sf is the fraction that is the shelf slope.
        The remaining fraction is the shelf.
        """
        s = ( x - x0 )/dx
        sbs = s - lf
        ssd = s - (1-sf)
        return shelf * topo.scurve(sbs,0,bf) + ( 1 - shelf ) * topo.scurve(ssd,0,sf)
    # More complicate structures built from the above simple shapes
    def add_NS_ridge(self, lon, lat0, lat1, dlon, dH, clip=0, p=1):
        r_fn = topo.cone(topo.dist_from_line(self.XC, lon, self.YC, lat0, lat1), 0, dlon)**p
        self.z = numpy.maximum(self.z, numpy.minimum(clip, (self.D0 - dH) * ( r_fn - 1 ) - dH))
    def add_NS_coast(self, lon, lat0, lat1, dlon, shelf):
        r = topo.dist_from_line(self.XC, lon, self.YC, lat0, lat1)
        self.z = numpy.maximum(self.z, - self.D0 * topo.coastal_sprofile(r, 0, dlon, shelf/self.D0) )
    def add_EW_ridge(self, lon0, lon1, lat, dlat, dH, clip=0, p=1):
        r_fn = topo.cone(topo.dist_from_line(self.YC, lat, self.XC, lon0, lon1), 0, dlat)**p
        self.z = numpy.maximum(self.z, numpy.minimum(clip, dH * ( r_fn - 1) - self.D0))
    def add_EW_coast(self, lon0, lon1, lat, dlat, shelf):
        r = topo.dist_from_line(self.YC, lat, self.XC, lon0, lon1)
        self.z = numpy.maximum(self.z, -self.D0 * topo.coastal_sprofile(r, 0, dlat, shelf/self.D0) )
    def add_angled_coast(self, lon_eq, lat_mer, dr, shelf):
        A, B, C = lat_mer, lon_eq, -lon_eq * lat_mer
        r = 1. / numpy.sqrt( A*A + B*B )
        r = r * ( A * self.XC + B * self.YC + C )
        r_fn = topo.coastal_sprofile(r, 0, dr, shelf/self.D0)
        self.z = numpy.maximum(self.z, -self.D0 * r_fn )
    def add_circular_ridge(self, lon0, lat0, radius, dr, dH, clip=0):
        r = numpy.sqrt( (self.XC - lon0)**2 + (self.YC - lat0)**2 )
        r = numpy.abs( r - radius)
        r_fn = topo.clipped_cone(r, 0, dr, 1 - dH/self.D0)
        self.z = numpy.maximum(self.z, numpy.minimum(clip, self.D0 * ( r_fn - 1 ) ) )
    def dist_from_line(X,x0,Y,y0,y1):
        """Returns distance from line x=x0 between y=y0 and y=y1"""
        dx = X - x0
        yr = numpy.minimum( numpy.maximum(Y, y0), y1)
        dy = Y - yr
        return numpy.sqrt( dx*dx + dy*dy)
    def test1d(ax):
        """Displays the library of 1D simple profiles"""
        x = numpy.linspace(-.1,1.2,100)
        ax.plot(x, topo.box(x,.25,.5), label='box(x,0.25,.5)')
        ax.plot(x, topo.cone(x,0.5,.25), label='cone(x,0,.5)')
        ax.plot(x, topo.clipped_cone(x,0.5,.2,.8), label='clippedcone(x,0.5,.2,.8)')
        ax.plot(x, topo.scurve(x,0,1), label='scurve(x,0,1)')
        ax.plot(x, topo.coastal_sprofile(x,0,1,.2), label='coastal_sprofile(x,0,1,.2)')
        ax.legend()
    def plot(self, fig, Atlantic_lon_offset=None):
        ax = fig.add_subplot(2,2,1)
        im = ax.contour(self.xc, self.yc, self.z, levels=numpy.arange(-self.D0,1,500))
        fig.colorbar(im, ax=ax); ax.set_title('Depth (plan view)')

        # Draw coastlines in NeverWorld2 space (i.e. offset in longitude)
        if Atlantic_lon_offset is not None:
            for geo in cartopy.feature.COASTLINE.geometries():
                x,y=geo.xy
                ax.plot(numpy.array(x)-Atlantic_lon_offset,y, 'k:')
        ax.set_xlim(self.xg.min(), self.xg.max())
        ax.set_ylim(self.yg.min(), self.yg.max())
        ax.set_aspect('equal')

        ax = fig.add_subplot(2,2,2)
        ax.plot( self.yc, self.z.max(axis=1), 'k:')
        ax.plot( self.yc, self.z[:,::10]); ax.set_title('Profiles at various longitudes');
        ax = fig.add_subplot(2,2,3)
        ax.plot( self.xc, self.z[::10,:].T); ax.set_title('Profiles at various latitudes');
        ax = fig.add_subplot(2,2,4)
        im = ax.pcolormesh(self.xg, self.yg, self.z); fig.colorbar(im, ax=ax);
        ax.set_aspect('equal')


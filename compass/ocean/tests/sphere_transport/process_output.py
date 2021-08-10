from netCDF4 import Dataset
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
import numpy as np
from importlib import resources

def appx_mesh_size(dataset):
  ncells = len(dataset.dimensions["nCells"])
  return np.sqrt(4*np.pi/ncells)

def compute_error_from_output_ncfile(dataset, lev=1):
  """
    Given a netCDF4 Dataset associated with the output.nc file from a test case in the sphere_transport test group,
    this function computes the linf and l2 relative error values by comparing the final time step to the initial condition.


    Args:
      dataset: a Dataset instance from the netCDF4 module, initialized with an MPAS output.nc file.
      lev: vertical level to plot.

    Returns:
      dictionary containing the linf and l2 relative errors for each of the 3 debug tracers.
  """
  tracer1_exact = dataset.variables["tracer1"][0,:,lev]
  tracer2_exact = dataset.variables["tracer2"][0,:,lev]
  tracer3_exact = dataset.variables["tracer3"][0,:,lev]
  tracer1_error = np.abs(dataset.variables["tracer1"][12,:,lev] - tracer1_exact)
  tracer2_error = np.abs(dataset.variables["tracer2"][12,:,lev] - tracer2_exact)
  tracer3_error = np.abs(dataset.variables["tracer3"][12,:,lev] - tracer3_exact)
  cell_area = dataset.variables["areaCell"][:]
  tracer1_linf = np.amax(tracer1_error) / np.amax(np.abs(tracer1_exact))
  tracer2_linf = np.amax(tracer2_error) / np.amax(np.abs(tracer2_exact))
  tracer3_linf = np.amax(tracer3_error) / np.amax(np.abs(tracer3_exact))
  tracer1_l2 = np.sqrt(np.sum(np.square(tracer1_error)*cell_area) / np.sum(np.square(tracer1_exact)*cell_area))
  tracer2_l2 = np.sqrt(np.sum(np.square(tracer2_error)*cell_area) / np.sum(np.square(tracer2_exact)*cell_area))
  tracer3_l2 = np.sqrt(np.sum(np.square(tracer3_error)*cell_area) / np.sum(np.square(tracer3_exact)*cell_area))
  result = dict()
  result["tracer1"] = {"linf":tracer1_linf, "l2":tracer1_l2}
  result["tracer2"] = {"linf":tracer2_linf, "l2":tracer2_l2}
  result["tracer3"] = {"linf":tracer3_linf, "l2":tracer3_l2}
  return result

def compute_convergence_rates(dlambda, linf, l2):
  """
    Given a set of approximate mesh sizes (dlambda) and the corresponding linf and l2 relative error values,
    this function computes the approximate convergence rates for each error.  These values are computed by
    compute_error_from_output_ncfile for tracer1.

    Args:
      dlambda: approximate mesh size
      linf: linf relative error associated with each mesh size
      l2: l2 relative error associated with each mesh size

    Returns:
      linfrates, l2rates:  Approximate convergence rates for each error.
  """
  runs = np.log(dlambda[1:]) - np.log(dlambda[:-1])
  linfrises = np.log(linf[1:]) - np.log(linf[:-1])
  l2rises = np.log(l2[1:]) - np.log(l2[:-1])
  return linfrises/runs, l2rises/runs

def print_error_conv_table(tcname, resvals, dlambda, l2, l2rates, linf, linfrates):
  """
    Print error values and approximate convergence rates to the console as a table.

    Args:
      tcname: Name of test case
      resvals: resolution values such as 240, for QU240
      dlambda: approximate mesh size
      l2: l2 error, computed by compute_error_from_output_ncfile for tracer1
      l2rates: appx. convergence rates for l2, computed by compute_convergence_rates
      linf: linf error, computed by compute_error_from_output_ncfile for tracer1
      linfrates: appx. convergence rates for linf, computed by compute_convergence_rates
  """
  table_rows = []
  for i, r in enumerate(resvals):
    table_rows.append([r, dlambda[i], l2[i], l2rates[i-1] if i>0 else '-', linf[i], linfrates[i-1] if i>0 else '-'])
  print(tcname + ": error data for tracer1")
  row_headers = ["res", "dlambda", "l2", "l2 rate", "linf", "linf rate"]
  row_format = "{:>24}"*len(row_headers)
  print(row_format.format(*row_headers))
  for row in table_rows:
    print(row_format.format(*row))

def read_ncl_rgb_file(cmap_filename):
  """
    Read a .rgb file from the NCAR Command Language, and return a matplotlib colormap.

    Prerequisite: Download an RGB file using the links provided by the NCL web pages,
    https://www.ncl.ucar.edu/Document/Graphics/color_table_gallery.shtml

    Args:
      filename: downloaded .rgb file name

    Returns:
      colormap usable by matplotlib that matches the ncl colormap
  """
  map_file_found = False
  try :
    with resources.open_text("compass.ocean.tests.sphere_transport.resources", cmap_filename) as f:
      flines = f.readlines()
    map_file_found = True
  except:
    pass
  if map_file_found:
    ncolors = int(flines[0].split()[-1])
    rgb = np.zeros((ncolors,3))
    for i, l in enumerate(flines[3:]):
      ls = l.split()
      for j in range(3):
        rgb[i,j] = ls[j]
    rgb /= 255
    result = ListedColormap(rgb, name=cmap_filename)
  else:
    print("error reading ncl colormap. using matplotlib default instead.")
    result = matplotlib.cm.get_cmap()
  return result

def plot_sol(fig, tcname, dataset):
  """
    Plot the solution at time 0, t = T/2, and T=T for test cases in the sphere_transport test group.
    Each tracer is plotted in its own row.  Columns correspond to t=0, t=T/2, and t=T.

    Args:
      fig: A matplotlib figure instance
      tcname: name of the test case whose solutions will be plotted
      dataset: instance of a netCDF4 dataset initialized initialized with an MPAS output.nc file.
  """
  xc = dataset.variables["lonCell"][:]
  yc = dataset.variables["latCell"][:]
  gspc = GridSpec(nrows=3, ncols=3, figure=fig)
  yticks = np.pi * np.array([-0.5, -0.25, 0, 0.25, 0.5])
  yticklabels = [-90, -45, 0, 45, 90]
  xticks = np.pi * np.array([0, 0.5, 1, 1.5, 2])
  xticklabels = [0, 90, 180, 270, 360]

  clev = np.linspace(0,1.1,20)
  nclcmap = read_ncl_rgb_file("wh-bl-gr-ye-re.rgb")
  axes = []
  for i in range(3):
    for j in range(3):
      axes.append(fig.add_subplot(gspc[i,j]))
  cm=axes[0].tricontourf(xc, yc, dataset.variables["tracer1"][0,:,1],levels=clev)
  axes[1].tricontourf(xc, yc, dataset.variables["tracer1"][6,:,1],levels=clev)
  axes[2].tricontourf(xc, yc, dataset.variables["tracer1"][12,:,1],levels=clev)
  axes[3].tricontourf(xc, yc, dataset.variables["tracer2"][0,:,1],levels=clev)
  axes[4].tricontourf(xc, yc, dataset.variables["tracer2"][6,:,1],levels=clev)
  axes[5].tricontourf(xc, yc, dataset.variables["tracer2"][12,:,1],levels=clev)
  axes[6].tricontourf(xc, yc, dataset.variables["tracer3"][0,:,1],levels=clev)
  axes[7].tricontourf(xc, yc, dataset.variables["tracer3"][6,:,1],levels=clev)
  axes[8].tricontourf(xc, yc, dataset.variables["tracer3"][12,:,1],levels=clev)
  for i in range(9):
    axes[i].set_xticks(xticks)
    axes[i].set_yticks(yticks)
    if i%3 != 0:
      axes[i].set_yticklabels([])
  for i in range(3):
    axes[3*i].set_yticklabels(yticklabels)
    axes[6+i].set_xticklabels(xticklabels)
  for i in range(6):
    axes[i].set_xticklabels([])
  fig.colorbar(ScalarMappable(cmap=nclcmap))
  fig.colorbar(cm)

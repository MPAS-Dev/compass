import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.interpolate import NearestNDInterpolator, RegularGridInterpolator

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

# resolutions = [256, 128, 64, 32, 16]
resolutions = [256, 128, 64, 32]

ds_dem = xr.open_dataset('subgrid/viz/buttermilk_bathy.nc')
values = -1.0 * ds_dem.Band1.values
interp_dem = RegularGridInterpolator(
    (ds_dem.x.values, ds_dem.y.values), values.T, method='nearest')

ds_reference = xr.open_dataset('subgrid/viz/output_8m.nc')
xy = np.vstack((ds_reference.xCell.values, ds_reference.yCell.values)).T
interp_ref = NearestNDInterpolator(xy, ds_reference.ssh.values.T)

# points = np.array(
#     [[2.8, 0.53], [1.9, 1.66], [2.4, 3.029], [2.51, 3.027], [1.26, 1.56]])
points = np.array(
    [[2.8, 0.53], [1.9, 1.66], [3.0, 3.25], [2.6, 3.25], [1.26, 1.56]])
# No #3
# points = np.array([[2.8, 0.53], [1.9, 1.66], [2.51, 3.027], [1.26, 1.56]])
# #5 Only
# points = np.array([[1.26, 1.56]])
points = points * 1000

# ds_eval = xr.open_dataset('subgrid/viz/output_32m.nc')
# points = np.vstack((ds_eval.xCell.values, ds_eval.yCell.values)).T
# max_x = 3500
# min_x = 0.5
# max_y = 3500
# min_y = 0.5
# mask_x = (points[:,0] >= min_x) & (points[:,0] <= max_x)
# mask_y = (points[:,1] >= min_y) & (points[:,1] <= max_y)
# combined_mask = mask_x & mask_y
# points = points[combined_mask]

h0 = 0.05
# h0 = 0

rmse_subgrid = []
rmse_standard = []
Linf_subgrid = []
Linf_standard = []
for j, res in enumerate(resolutions):
    print(f'resolution: {res}')

    ds_subgrid = xr.open_dataset(f'subgrid/viz/output_{res}m.nc')
    xy = np.vstack((ds_subgrid.xCell.values, ds_subgrid.yCell.values)).T
    interp_subgrid = NearestNDInterpolator(xy, ds_subgrid.ssh.values.T)

    ds_standard = xr.open_dataset(f'standard/viz/output_{res}m.nc')
    xy = np.vstack((ds_standard.xCell.values, ds_standard.yCell.values)).T
    interp_standard = NearestNDInterpolator(
        xy, np.squeeze(ds_standard.layerThickness.values).T)

    bathy = interp_dem(points)

    subgrid_ssh = interp_subgrid(points).T
    subgrid_h = subgrid_ssh + bathy
    # subgrid_h = np.clip(subgrid_h, h0, None)
    subgrid_h[subgrid_h <= h0] = np.nan

    standard_h = interp_standard(points).T

    ref_ssh = interp_ref(points).T
    ref_h = ref_ssh + bathy
    # ref_h = np.clip(ref_h, h0, None)
    ref_h[ref_h <= h0] = np.nan

    diff_subgrid = subgrid_h - ref_h
    diff_standard = standard_h - ref_h

    sum_sqdiff_subgrid = np.nansum(np.square(diff_subgrid))
    sum_sqdiff_standard = np.nansum(np.square(diff_standard))

    N_subgrid = np.sum(~np.isnan(diff_subgrid))
    N_standard = np.sum(~np.isnan(diff_standard))

    rmse_subgrid.append(np.sqrt(sum_sqdiff_subgrid / N_subgrid))
    rmse_standard.append(np.sqrt(sum_sqdiff_standard / N_standard))
    print(f'rmse_subgrid: {rmse_subgrid[j]}')
    print(f'rmse_standard: {rmse_standard[j]}')

    Linf_subgrid.append(np.nanmax(np.abs(subgrid_h - ref_h)))
    Linf_standard.append(np.nanmax(np.abs(standard_h - ref_h)))
    print(f'Linf_subgrid: {Linf_subgrid[j]}')
    print(f'Linf_standard: {Linf_standard[j]}')

# wct_subgrid = np.array([4.12, 9.82, 18.98, 47.39, 113.49]) * 12.0
# wct_standard = np.array([3.99, 9.49, 17.61, 43.26, 104.97]) * 12.0
# nh_subgrid = np.array([4.12, 9.82, 18.98, 47.39*4, 113.49*8]) * 12.0 / 3600.0
# nh_standard = np.array([3.99, 9.49, 17.61, 43.26*4, 104.97*8]) * 12.0/3600.0
wct_subgrid = np.array([4.12, 9.82, 18.98, 47.39]) * 12.0
wct_standard = np.array([3.99, 9.49, 17.61, 43.26]) * 12.0
nh_subgrid = np.array([4.12, 9.82, 18.98, 47.39 * 4]) * 12.0 / 3600.0
nh_standard = np.array([3.99, 9.49, 17.61, 43.26 * 4]) * 12.0 / 3600.0
rmse_subgrid = np.array(rmse_subgrid)
rmse_standard = np.array(rmse_standard)

log_wct_standard_interp = np.interp(
    np.log10(rmse_subgrid), np.log10(np.flip(rmse_standard)),
    np.log10(np.flip(wct_standard)))
wct_standard_interp = 10**log_wct_standard_interp
print(wct_standard_interp)
print(wct_subgrid)
speedup = np.divide(wct_standard_interp, wct_subgrid)
print(speedup)

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(1, 1, 1)
for i in range(3):
    # ax.plot(
    #     [wct_subgrid[i], wct_standard_interp[i]],
    #     [rmse_subgrid[i], rmse_subgrid[i]], 'k-')
    ax.annotate(
        "", xy=(wct_subgrid[i], rmse_subgrid[i]),
        xytext=(wct_standard_interp[i], rmse_subgrid[i]),
        arrowprops=dict(arrowstyle="->"))
    x = 0.5 * (wct_subgrid[i] + wct_standard_interp[i])
    ax.text(
        0.95 * x, 1.03 * rmse_subgrid[i],
        f'{round(speedup[i], 2)}' + r'$\times$',
        horizontalalignment='center')
ax.plot(wct_subgrid, rmse_subgrid, 'o-', label='subgrid')
ax.plot(wct_standard, rmse_standard, 'o-', label='standard')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('wall clock time per day (s)')
ax.set_ylabel('RMSE (m)')
fig.subplots_adjust(bottom=0.2)
fig.legend(loc='lower center', ncol=2)
fig.savefig('buttermilk_bay_error_wct.png', dpi=400, bbox_inches='tight')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(nh_subgrid, rmse_subgrid, 'o-', label='subgrid')
ax.plot(nh_standard, rmse_standard, 'o-', label='standard')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel('node hours per day')
ax.set_ylabel('RMSE (m)')
fig.subplots_adjust(bottom=0.2)
fig.legend(loc='lower center', ncol=2)
fig.savefig('buttermilk_bay_error_ndhr.png', dpi=400)

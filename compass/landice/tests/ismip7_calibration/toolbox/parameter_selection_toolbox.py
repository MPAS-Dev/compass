import os

import numpy as np
import xarray as xr


def calculate_objective_function(
    sample_size,
    reso,
    t1_model,
    t1_obs_mean,
    t1_obs_sigma,
    t1_weights,
    t2_model,
    t2_obs_mean,
    t2_obs_sigma,
    t2_weights,
    t3_model,
    t3_obs_mean,
    t3_obs_sigma,
    t3_weights,
    t4_model,
    t4_obs_mean,
    t4_obs_sigma,
    t4_weights,
):
    """
    Input:
    To create input datasets, please see calc_term1, etc
    - sample_size: number of samples to take
    - reso: native model resolution
    - t1_model: xarray dataset containing modelled melt rates, aggregated
                and indexed by basins
    - t1_obs_mean, sigma: target melt rates in basins, and uncertainties
    - t2_model: xarray dataset containing modelled melt rates, aggregated
                and indexed by buttressing bins
    - t2_obs_mean, sigma: target melt rates in buttressing bins,
                and uncertainties
    - t2_weights: based on buttressing values
    - t3_model: xarray dataset containing modelled melt rates,
                averaged and indexed by basins, and ocean models
    - t3_obs_mean, sigma: target melt rates in basins and ocean models,
                and uncertainties
    - t3_weights: weighting of different terms
    - t4_model: xarray dataset containing modelled melt rates,
                aggregated and indexed by region = pine island, dotson,
                and observations year
    - t4_obs_mean, sigma: target melt rates in regions, years,
                and uncertainties
    - t4_weights:  weighting of different terms
    Output:
    -min_p1: list of p1 values that minimise randomly sampled objective
             function, length of sample size
    -min_p2: list of cooresponding p2 values

    This function finds the pair of p1,p2 for which the parameterised melt
    optimises three terms:
    -J1: basin-integrated melt for present-day
    -J2: buttressing-bin integrated melt for present-day,
         weighted by buttressing
    -J3: basin-averaged melt for cold and warm cases of the
         ocean modelling datasets
    -J4: basin-integrated melt from observations of PIG and Dotson
    """

    nBasins = int(t1_model.basins.values.max())

    ################################################
    # randomly sample the weights of the terms
    a1 = xr.DataArray(
        np.random.uniform(0, 1, size=sample_size), dims=['sample']
    )
    a2 = xr.DataArray(
        np.random.uniform(0, 1, size=sample_size), dims=['sample']
    )
    a3 = xr.DataArray(
        np.random.uniform(0, 1, size=sample_size), dims=['sample']
    )
    a4 = xr.DataArray(
        np.random.uniform(0, 1, size=sample_size), dims=['sample']
    )

    ###############################################
    # Sample uncertainties in terms

    # Sample uncertainty in term 1
    t1_target_s = []
    for b in range(nBasins + 1):
        t1_target_s = t1_target_s + [
            np.random.normal(
                loc=t1_obs_mean.values[b],
                scale=t1_obs_sigma.values[b],
                size=sample_size,
            )
        ]
    t1_target_s = xr.DataArray(
        t1_target_s,
        dims=['basins', 'sample'],
        coords={
            'basins': t1_model.basins.values,
            'sample': np.arange(sample_size),
        },
    )

    # Sample uncertainty in term 2
    t2_target_s = []
    nBins = 10
    for b in range(nBins):
        t2_target_s = t2_target_s + [
            np.random.normal(
                loc=t2_obs_mean[b],
                scale=t2_obs_sigma[b],
                size=sample_size,
            )
        ]
    t2_target_s = xr.DataArray(
        t2_target_s,
        dims=['BFRN_bins', 'sample'],
        coords={
            'BFRN_bins': t2_model.BFRN_bins.values,
            'sample': np.arange(sample_size),
        },
    )

    # Term 3 targets
    samples = np.random.normal(
        loc=t3_obs_mean.values[..., np.newaxis],
        scale=t3_obs_sigma.values[..., np.newaxis],
        size=(*t3_obs_mean.values.shape, sample_size),
    )
    t3_target_s = xr.DataArray(
        samples,
        dims=[*t3_obs_mean.dims, 'sample'],
        coords={**t3_obs_mean.coords, 'sample': np.arange(sample_size)},
    )

    t3_weights_samples = xr.DataArray(
        np.random.uniform(
            0,
            1,
            size=(
                len(t3_obs_mean.model.values),
                len(t3_obs_mean.basins.values),
                sample_size,
            ),
        ),
        dims=['model', 'basins', 'sample'],
        coords={
            'model': t3_obs_mean.model.values,
            'basins': t3_obs_mean.basins.values,
            'sample': np.arange(sample_size),
        },
    )

    # Term 4
    samples = np.random.normal(
        loc=t4_obs_mean.values[..., np.newaxis],
        scale=t4_obs_sigma.values[..., np.newaxis],
        size=(*t4_obs_mean.values.shape, sample_size),
    )
    t4_target_s = xr.DataArray(
        samples,
        dims=[*t4_obs_mean.dims, 'sample'],
        coords={**t4_obs_mean.coords, 'sample': np.arange(sample_size)},
    )
    t4_weights_samples = xr.DataArray(
        np.random.uniform(
            0,
            1,
            size=(
                len(t4_obs_mean.region.values),
                len(t4_obs_mean.year.values),
                sample_size,
            ),
        ),
        dims=['region', 'year', 'sample'],
        coords={
            'region': t4_obs_mean.region.values,
            'year': t4_obs_mean.year.values,
            'sample': np.arange(sample_size),
        },
    )

    ###################
    # Calculate objective function

    term1 = mae(t1_model, t1_target_s, t1_weights, 'basins')
    term2 = mae(t2_model, t2_target_s, t2_weights, ['BFRN_bins'])
    # important to use skipna here
    term3 = mae(
        t3_model,
        t3_target_s,
        t3_weights * t3_weights_samples,
        ['basins', 'model'],
        True,
    )
    # important to use skipna here
    term4 = mae(
        t4_model,
        t4_target_s,
        t4_weights * t4_weights_samples,
        ['region', 'year'],
        True,
    )

    # Sample the objective function
    eps = 0.000001  # to avoid divison by 0
    objective_function = (
        a1 * term1 / (term1.median(dim=['p1', 'p2']) + eps)
        + a2 * term2 / (term2.median(dim=['p1', 'p2']) + eps)
        + a3 * term3 / (term3.median(dim=['p1', 'p2']) + eps)
        + a4 * term4 / (term4.median(dim=['p1', 'p2']) + eps)
    )

    # Identify minimum values
    objective_function_stacked = objective_function.stack(params=('p1', 'p2'))
    min_params = objective_function_stacked.idxmin(dim='params').values

    min_p1 = np.array([v[0] for v in min_params])
    min_p2 = np.array([v[1] for v in min_params])

    return min_p1, min_p2


def mae(predicted=None, observed=None, weights=1, dims='basins', skipna=False):
    """
    Calculates mean absolute error
    """
    return (
        abs(weights * (predicted - observed))
        .mean(dims, skipna=skipna)
        .rename('result')
    )


def calculate_term1(pd_ensemble, mask_m, basins_m, nBasins, cvt_m, MeltData):
    ########
    # TERM 1
    # parameterisaition melt, aggregate to Gt/a per basin
    t1_model = (
        pd_ensemble['melt_rate']
        .where(mask_m, np.nan)
        .groupby(basins_m)
        .sum(skipna=True)
        * cvt_m
    )  # convert to Gt/a
    # make sure to remove regions that do not have optimal dT for any basin
    t1_model = t1_model.where(t1_model != 0, np.nan)

    # Observed melt in Gt/a per basin, observed melt is "sample_size"-times
    # randomly sampled assuming normal distribution
    t1_obs_mean = MeltData['BMR (Gt/yr)'].values
    t1_obs_mean = xr.DataArray(
        data=t1_obs_mean,
        name='melt_Gt_per_y',
        dims=['basin'],
        coords={'basin': range(len(MeltData))},
    )

    t1_obs_sigma = MeltData['BMR uncert (Gt/yr)'].values
    t1_obs_sigma = xr.DataArray(
        data=t1_obs_sigma,
        dims=['basin'],
        coords={'basin': range(len(MeltData))},
        name='melt_unc_Gt_per_y',
    )

    return t1_model, t1_obs_mean, t1_obs_sigma


def calculate_term2(pd_ensemble, mask_m, bfrn_m, cvt_m, buttressing_target):
    ########
    # TERM 2

    # parameterisation melt aggregated per buttressing bin, in Gt/a
    t2_model = (
        pd_ensemble['melt_rate']
        .where(mask_m, np.nan)
        .groupby(bfrn_m['BFRN_bins'])
        .sum(skipna=True)
        * cvt_m
    )
    t2_model = t2_model.where(t2_model != 0, np.nan)

    t2_obs_mean = buttressing_target['melt_mean']
    t2_obs_sigma = buttressing_target['melt_mean_err']

    return t2_model, t2_obs_mean, t2_obs_sigma


def calculate_term3(
    cold_ensemble,
    warm_ensemble,
    cold_target,
    warm_target,
    mask_m,
    basins_m,
):
    ########
    # TERM 3

    # parameterisation melt, average in kg/m2/a per basin for cold ocean
    t3_model_cold = (
        (cold_ensemble['melt_rate'].where(mask_m, np.nan))
        .groupby(basins_m)
        .mean()
    )
    t3_model_cold = t3_model_cold.where(t3_model_cold != 0, np.nan)

    # parameterisation melt, average in kg/m2/a per basin for warm ocean
    t3_model_warm = (
        (warm_ensemble['melt_rate'].where(mask_m, np.nan))
        .groupby(basins_m)
        .mean()
    )
    t3_model_warm = t3_model_warm.where(t3_model_warm != 0, np.nan)

    t3_model = t3_model_warm - t3_model_cold

    t3_obs_mean = warm_target.melt_rate - cold_target.melt_rate

    t3_obs_sigma = np.sqrt(
        warm_target.melt_rate_uncert**2 + cold_target.melt_rate_uncert**2
    )

    return t3_model, t3_obs_mean, t3_obs_sigma


def calculate_term4(obs_ensemble, region_label_m, t4_obs, mask_m, cvt_m):
    """
    Calculate Term 4 which is modelled melt in different years
    for PIG and Dotson
    """
    region_label_m = region_label_m.reindex_like(obs_ensemble)

    obs_ensemble_stacked = obs_ensemble.stack(grid=('x', 'y'))
    mask_stacked = region_label_m.stack(grid=('x', 'y'))
    t4_model = (
        obs_ensemble_stacked['melt_rate'].groupby(mask_stacked).sum(dim='grid')
        * cvt_m
    )  # Gt/a
    t4_model = t4_model.rename({'group': 'region'})
    t4_model = t4_model.where(t4_model != 0, np.nan)
    t4_model = t4_model.where(t4_obs.melt_rate.notnull())

    t4_obs_mean = t4_obs.melt_rate
    t4_obs_sigma = t4_obs.melt_rate_uncert
    # make to order region index to pig, dotson
    t4_model = t4_model.reindex_like(t4_obs_mean)

    return t4_model, t4_obs_mean, t4_obs_sigma


def optimise_deltaT(dT_ensemble, basins, reso, MeltDataImbie):
    """
    Calculate optimal deltaT for basin-wide present-day melt.
    """

    number_of_basins = int(basins.max().values)
    cvt = reso**2 / 1e12  # to convert to Gt/a

    optimal_deltaT_per_basin = []
    residual_per_basin = []
    sensitivity_per_basin = []

    param_melt_rate = dT_ensemble.sel(deltaT=0).copy(deep=True) * np.nan

    for basin_i in range(number_of_basins + 1):
        bmr = dT_ensemble.where(basins == basin_i, 0.0).sum(['x', 'y']) * cvt

        # only use deltaT between -2 and 2
        bmr = bmr.where(
            np.logical_and(bmr['deltaT'] <= 2.0, bmr['deltaT'] >= -2.0), np.nan
        )

        optimal_deltaT_per_basin.append(
            np.round(
                (abs(bmr - MeltDataImbie.loc[basin_i, 'BMR (Gt/yr)']))
                .idxmin()
                .item(),
                3,
            )
        )
        residual_per_basin.append(
            (abs(bmr - MeltDataImbie.loc[basin_i, 'BMR (Gt/yr)'])).min().item()
        )

        # if an optimal delatT exists, save melt and calc melt sensitivity
        if not np.isnan(optimal_deltaT_per_basin[-1]):
            param_melt_rate = param_melt_rate.where(
                basins != basin_i,
                dT_ensemble.sel(
                    deltaT=optimal_deltaT_per_basin[-1], method='nearest'
                ),
            )
            # Calc approx melt sensitivity.
            # otherwise approximate with higher values, ideally +1deg C
            # Note that this is only approximate
            sensitivity_per_basin.append(
                np.round(
                    (
                        (
                            dT_ensemble.sel(
                                deltaT=optimal_deltaT_per_basin[-1] + 1,
                                method='nearest',
                            )
                            .where(basins == basin_i, np.nan)
                            .mean()
                            - param_melt_rate.where(
                                basins == basin_i, np.nan
                            ).mean()
                        ).item()
                        / 1
                    ),
                    2,
                )
            )
        else:
            sensitivity_per_basin.append(np.nan)
    result_ds = xr.Dataset(
        data_vars=dict(
            melt_rate=(['y', 'x'], param_melt_rate.values),
            optimal_deltaT_per_basin=(
                ['basin'],
                np.array(optimal_deltaT_per_basin),
            ),
            sensitivity_per_basin=(['basin'], np.array(sensitivity_per_basin)),
            residual_per_basin=(['basin'], np.array(residual_per_basin)),
        ),
        coords=dict(
            x=(['x'], dT_ensemble['x'].values),
            y=(['y'], dT_ensemble['y'].values),
            basin=(['basin'], np.arange(0, number_of_basins + 1)),
        ),
    )

    return result_ds


def select_optimal_deltaT(
    ds, basins, boxes, obs_data, param_type, outname, reso, ice_density, dT
):
    """
    Only used for PICO
    Input:
    - ds: xarray dataset containing melt rates (m.i.e/a), with a dimension
      deltaT that will be optimised over, on a regular grid
    - basins: xarray dataset containing basin numbers, starting at 1
    - obs_data: data frame containing basin-aggregated melt rates in Gt/a
    - param_type: pico, quadratic, ...
    - outname: output file to save to
    - reso: resolution of model output
    - dT: allowed adjustment range
    Output:
    - saves a netcdf file to "outname" containing the melt rates for each
    basin based on optimal deltaT, and arrays of optimal_deltaT, residuals
    and melt sensitivities
    """

    # print('Identifying optimal delta T for each basin...')

    number_of_basins = int(basins.max().values)
    cvt = reso**2 * ice_density / 1e12  # to convert to Gt/a

    optimal_deltaT_per_basin = []
    residual_per_basin = []
    sensitivity_per_basin = []

    param_melt_rate = ds['melt_rate'].sel(deltaT=0).copy(deep=True) * np.nan

    for basin_i in range(1, number_of_basins + 1):
        bmr = (
            ds['melt_rate'].where(basins == basin_i, 0.0).sum(['x', 'y']) * cvt
        )
        if param_type == 'pico':
            # Add physical constraints from Reese et al., 2018
            bmrBox1 = (
                ds['melt_rate']
                .where(np.logical_and(basins == basin_i, boxes == 1), 0.0)
                .sum(['x', 'y'])
                * cvt
            )
            bmrBox2 = (
                ds['melt_rate']
                .where(np.logical_and(basins == basin_i, boxes == 2), 0.0)
                .sum(['x', 'y'])
                * cvt
            )
            bmr = bmr.where(
                np.logical_and(bmrBox1 > 0, bmrBox1 > bmrBox2), np.nan
            )

        # only use deltaT between +-dT
        bmr = bmr.where(
            np.logical_and(bmr['deltaT'] <= dT, bmr['deltaT'] >= -1 * dT),
            np.nan,
        )

        optimal_deltaT_per_basin.append(
            (abs(bmr - obs_data.loc[basin_i, 'BMR (Gt/yr)'])).idxmin()
        )
        residual_per_basin.append(
            (abs(bmr - obs_data.loc[basin_i, 'BMR (Gt/yr)'])).min()
        )

        # if an optimal delatT exists, save melt and calc melt sensitivity
        if not np.isnan(optimal_deltaT_per_basin[-1]):
            param_melt_rate = param_melt_rate.where(
                basins != basin_i,
                ds['melt_rate'].sel(deltaT=optimal_deltaT_per_basin[-1]),
            )
            # Calc approx melt sensitivity.
            # If this is the max deltaT, use half a degree colder,
            # otherwise approximate with higher values, ideally +1deg C
            # Note that this is only approximate
            if optimal_deltaT_per_basin[-1] == ds.deltaT.max():
                sensitivity_per_basin.append(
                    (
                        param_melt_rate.where(basins == basin_i, np.nan).mean()
                        - ds['melt_rate']
                        .sel(
                            deltaT=optimal_deltaT_per_basin[-1] - 0.5,
                            method='nearest',
                        )
                        .where(basins == basin_i, np.nan)
                        .mean()
                    )
                    / 0.5
                )
            else:
                sensitivity_per_basin.append(
                    (
                        ds['melt_rate']
                        .sel(
                            deltaT=optimal_deltaT_per_basin[-1] + 1,
                            method='nearest',
                        )
                        .where(basins == basin_i, np.nan)
                        .mean()
                        - param_melt_rate.where(
                            basins == basin_i, np.nan
                        ).mean()
                    )
                    / 1
                )
        else:
            sensitivity_per_basin.append(np.nan)

    result_ds = xr.Dataset(
        data_vars=dict(
            melt_rate=(['y', 'x'], param_melt_rate.values),
            optimal_deltaT_per_basin=(
                ['basin'],
                np.array(optimal_deltaT_per_basin),
            ),
            sensitivity_per_basin=(['basin'], np.array(sensitivity_per_basin)),
            residual_per_basin=(['basin'], np.array(residual_per_basin)),
        ),
        coords=dict(
            x=(['x'], ds['x'].values),
            y=(['y'], ds['y'].values),
            basin=(['basin'], np.arange(1, number_of_basins + 1)),
        ),
    )
    result_ds.to_netcdf(outname)
    ds.close()
    result_ds.close()
    return result_ds.drop_vars('melt_rate')


def select_subensemble_using_optimal_deltaT(
    ds, basins, opt_ensemble, outname, p1, p2
):
    """
    Input:
    - ds: xarray dataset containing melt rates (m.i.e/a),
          wit dimension deltaT that will be selected from, on a regular grid
    - basins: xarray dataset containing basin numbers, starting at 1
    - opt_ensemble: array containing optimised deltaT's for present-day
    - outname: output file to save to
    Output:
    - saves a netcdf file to "outname" containing the melt rates for each
      basin based on optimal deltaT
    """

    # print('Select sub-ensemble...')

    number_of_basins = int(basins.max().values)

    # Create melt rate dataset based on optimal deltaT
    melt_rate = ds['melt_rate'].sel(deltaT=0).copy(deep=True) * np.nan

    for basin in range(1, number_of_basins + 1):
        optimal_deltaT = opt_ensemble['optimal_deltaT_per_basin'].loc[
            dict(p1=p1, p2=p2, basin=basin)
        ]

        if np.isnan(optimal_deltaT.values):
            melt_rate = melt_rate.where(
                basins != basin, ds['melt_rate'].sel(deltaT=0) * np.nan
            )
        else:
            melt_rate = melt_rate.where(
                basins != basin, ds['melt_rate'].sel(deltaT=optimal_deltaT)
            )

    result_ds = xr.Dataset(
        data_vars=dict(
            melt_rate=(['y', 'x'], melt_rate.values),
        ),
        coords=dict(x=(['x'], ds['x'].values), y=(['y'], ds['y'].values)),
    )

    result_ds.to_netcdf(outname)


def load_melt_rates_into_dataset(
    ensemble_name, ensemble_table, ensemble_path, p1_name, p2_name, identifier
):
    print('Loading ' + ensemble_name + ' into one dataset...')
    members = []
    p1s = []
    p2s = []

    for _i, ehash in enumerate(ensemble_table.index):
        p1 = ensemble_table.loc[ehash, p1_name]
        p2 = ensemble_table.loc[ehash, p2_name]
        p1s.append(p1)
        p2s.append(p2)

        print(
            os.path.join(
                ensemble_path,
                ensemble_name
                + '_'
                + str(ehash)
                + '/optimised'
                + identifier
                + '.nc',
            )
        )

        if os.path.isfile(
            os.path.join(
                ensemble_path,
                ensemble_name
                + '_'
                + str(ehash)
                + '/optimised'
                + identifier
                + '.nc',
            )
        ):
            ds = xr.load_dataset(
                os.path.join(
                    ensemble_path,
                    ensemble_name
                    + '_'
                    + str(ehash)
                    + '/optimised'
                    + identifier
                    + '.nc',
                )
            )
        elif os.path.isfile(
            os.path.join(
                ensemble_path,
                ensemble_name
                + '_'
                + str(ehash)
                + '_optimised'
                + identifier
                + '.nc',
            )
        ):
            ds = xr.load_dataset(
                os.path.join(
                    ensemble_path,
                    ensemble_name
                    + '_'
                    + str(ehash)
                    + '_optimised'
                    + identifier
                    + '.nc',
                )
            )
        else:
            print('Error: Cannot find dataset')

        ds = ds.assign_coords(ehash=ehash)
        members.append(ds)

    print('Combining datasets')
    ensemble = xr.concat(members, dim='ehash', coords='minimal')

    ensemble = (
        ensemble.assign_coords({'p1': ('ehash', p1s), 'p2': ('ehash', p2s)})
        .set_index(ehash=['p1', 'p2'])
        .unstack('ehash')
    )
    return ensemble

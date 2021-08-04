#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 15:41:04 2021

@author: shlomi
"""
from MA_paths import work_david
ml_path = work_david / 'ML'
features = ['Floor_number', 'SEI', 'New', 'Periph_value', 'Sale_year', 'Rooms_345',
            'distance_to_nearest_kindergarten', 'distance_to_nearest_school', 'Total_ends']

features1 = ['FLOORNO', 'DEALNATURE', 'NEWPROJECTTEXT',
             'BUILDINGYEAR',  'SEI_value', 'Ground', 'P2015_value', 'year', 'Building_Growth_Rate']

features2 = ['FLOORNO', 'DEALNATURE', 'NEWPROJECTTEXT',
             'SEI_value', 'Ground', 'year', 'Building_Growth_Rate']

features3 = ['Floor_number', 'SEI', 'New', 'Periph_value', 'Sale_year', 'Rooms_345',
             'Total_ends', 'mean_distance_to_4_mokdim']

features4 = ['Floor_number', 'SEI', 'New', 'Sale_year', 'Rooms_345',
             'Total_ends', 'mean_distance_to_28_mokdim']

apts = ['דירה', 'דירה בבית קומות']
apts_more = apts + ["קוטג' דו משפחתי", "קוטג' חד משפחתי",
                    "דירת גן", "בית בודד", "דירת גג", "דירת גג (פנטהאוז)"]
plot_names = {'Floor_number': 'Floor',
              'New': 'New Apartment',
              'Periph_value': 'Peripheriality',
              'distance_to_nearest_kindergarten': 'Nearest kindergarten',
              'distance_to_nearest_school': 'Nearest school',
              'Total_ends': 'Finished apartments',
              'Rooms_3': '3 rooms', 'Rooms_5': '5 rooms'
              }


def calculate_distance_from_gdf_to_employment_centers(gdf, path=work_david, n=4,
                                                      weights='Pop2020'):
    from cbs_procedures import read_emploment_centers_2008
    import numpy as np
    gdf = gdf[~gdf['ITM-E'].isnull()]
    gdf = gdf[~gdf['ITM-N'].isnull()]

    def mean_distance_to_n_mokdim(x, weights=None):
        # x = gdf['geometry']
        dists = points.distance(x).to_frame('distance') / 1000
        dists['Pop2020'] = points['Pop2020'] / 1000
        dists = dists.sort_values('distance')
        if weights is None:
            mean_dist = dists.iloc[0:n].mean()
        else:
            mean_dist = np.average(dists.iloc[0:n]['distance'], weights=dists.iloc[0:n][weights])
        return mean_dist.item()

    points = read_emploment_centers_2008(path, shape=True)
    if n is not None:
        gdf['mean_distance_to_{}_mokdim'.format(n)] = gdf['geometry'].apply(mean_distance_to_n_mokdim, weights=weights)
    else:
        for i, row in points.iterrows():
            print('calculating distance to {}.'.format(row['NameHE']))
            name = 'kms_to_{}'.format(i)
            gdf[name] = gdf.distance(row['geometry']) / 1000.0
    return gdf


def create_total_inout_timeseries_from_migration_network_and_cbs():
    from cbs_procedures import read_yearly_inner_migration
    from Migration_main import read_all_multi_year_gpickles
    from Migration_main import produce_nodes_time_series
    Gs = read_all_multi_year_gpickles()
    da = produce_nodes_time_series(Gs)
    df_in = da.sel(parameter='total_in').reset_coords(
        drop=True).to_dataset('node').to_dataframe()
    df_out = da.sel(parameter='total_out').reset_coords(
        drop=True).to_dataset('node').to_dataframe()
    df = read_yearly_inner_migration()
    inflow = df[df['year'] == 2018][[
        'city_code', 'inflow']].set_index('city_code').T
    inflow = inflow.append(
        df[df['year'] == 2019][['city_code', 'inflow']].set_index('city_code').T)
    inflow.index = [2018, 2019]
    inflow.index.name = 'time'
    inflow.columns.name = ''
    inflow.columns = [str(x) for x in inflow.columns]
    outflow = df[df['year'] == 2018][[
        'city_code', 'outflow']].set_index('city_code').T
    outflow = outflow.append(
        df[df['year'] == 2019][['city_code', 'outflow']].set_index('city_code').T)
    outflow.index = [2018, 2019]
    outflow.index.name = 'time'
    outflow.columns.name = ''
    outflow.columns = [str(x) for x in outflow.columns]
    df_in = df_in.append(inflow)
    df_out = df_out.append(outflow)
    return df_in, df_out


def prepare_features_and_save(path=work_david, savepath=None):
    from nadlan_EDA import load_nadlan_combined_deal
    from cbs_procedures import read_school_coords
    from cbs_procedures import read_kindergarten_coords
    from cbs_procedures import read_historic_SEI
    from cbs_procedures import read_building_starts_ends
    from cbs_procedures import calculate_building_rates
    from Migration_main import path_glob
    from cbs_procedures import calculate_minimum_distance_between_two_gdfs
    import numpy as np
    import pandas as pd

    def add_bgr_func(grp, bgr, name='3Rooms_starts'):
        # import numpy as np
        year = grp['Sale_year'].unique()[0]
        cc = grp['city_code'].unique()[0]
        try:
            if bgr.columns.dtype == 'object':
                gr = bgr.loc[year, str(cc)]
            elif bgr.columns.dtype == 'int':
                gr = bgr.loc[year, cc]
        except KeyError:
            gr = np.nan
        grp[name] = gr
        return grp

    df = load_nadlan_combined_deal(
        add_bgr=None, add_geo_layers=False, return_XY=True)
    # add distances to kindergarden, schools, building rates for each room type etc.
    print('Adding Building Growth rate.')
    bdf = read_building_starts_ends()
    for room in ['3rooms', '4rooms', '5rooms', 'Total']:
        room_begins = calculate_building_rates(
            bdf, phase='Begin', rooms=room, fillna=False)
        room_ends = calculate_building_rates(
            bdf, phase='End', rooms=room, fillna=False)
        df = df.groupby(['Sale_year', 'city_code']).apply(
            add_bgr_func, room_begins, name='{}_starts'.format(room))
        df = df.groupby(['Sale_year', 'city_code']).apply(
            add_bgr_func, room_ends, name='{}_ends'.format(room))
        # df.loc[df['{}_starts'.format(room)] == 0] = np.nan
        # df.loc[df['{}_ends'.format(room)] == 0] = np.nan
    print('Adding minimum distance to kindergartens.')
    kinder = read_kindergarten_coords()
    df = df.groupby('Sale_year').apply(
        calculate_minimum_distance_between_two_gdfs, kinder, 'kindergarten')
    df.index = df.index.droplevel(0)
    df = df.reset_index(drop=True)
    print('Adding minimum distance to schools.')
    school = read_school_coords()
    df = df.groupby('Sale_year').apply(
        calculate_minimum_distance_between_two_gdfs, school, 'school')
    df.index = df.index.droplevel(0)
    df = df.reset_index(drop=True)
    print('Adding historic city-level SEI.')
    sei = read_historic_SEI()
    sei.loc[2018] = sei.loc[2017]
    sei.loc[2019] = sei.loc[2017]
    df = df.groupby(['Sale_year', 'city_code']).apply(
        add_bgr_func, sei, name='SEI')
    # add inflow and outflow:
    print('Adding Inflow and Outflow')
    dfi, dfo = create_total_inout_timeseries_from_migration_network_and_cbs()
    df = df.groupby(['Sale_year', 'city_code']).apply(
        add_bgr_func, dfi, name='Inflow')
    df = df.groupby(['Sale_year', 'city_code']).apply(
        add_bgr_func, dfo, name='Outflow')
    # finally drop some cols so saving will not take a lot of space:
    df = df.drop(['P2015_cluster2', 'Parcel_Lot', 'Sale_Y_Q', 'Sale_quarter', 'Sale_month', 'District_HE', 'm2_per_room',
                  'StatArea_ID', 'Building', 'street_code', 'Street', 'ObjectID', 'TREND_FORMAT', 'TREND_IS_NEGATIVE', 'POLYGON_ID'], axis=1)
    if savepath is not None:
        filename = 'Nadaln_with_features.csv'
        df.to_csv(savepath/filename, na_rep='None', index=False)
        print('{} was saved to {}.'.format(filename, savepath))
    return df


def calc_vif(X, dropna=True, asfloat=True):
    import pandas as pd
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    if dropna:
        print('dropping na.')
        X = X.dropna()
    if asfloat:
        print('considering as float.')
        X = X.astype(float)
    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(
        X.values, i) for i in range(X.shape[1])]
    return(vif)


def scale_df(df, scaler, cols=None):
    import pandas as pd
    print('using {} scaler.'.format(scaler.__repr__()))
    if cols is None:
        scaled_vals = scaler.fit_transform(df)
        df_scaled = pd.DataFrame(scaled_vals)
        df_scaled.columns = df.columns
    else:
        print('scaling only {} cols.'.format(cols))
        df_sliced = df[cols]
        scaled_vals = scaler.fit_transform(df_sliced)
        df_scaled = pd.DataFrame(scaled_vals)
        df_scaled.columns = cols
        df_rest = df[[x for x in df.columns if x not in cols]]
        df_scaled = pd.concat([df_scaled, df_rest], axis=1)
        df_scaled = df_scaled[[x for x in df.columns]]
    return df_scaled, scaler


def load_nadlan_with_features(path=work_david, years=[2000, 2019], asset_type=apts, mokdim_version=False):
    import pandas as pd
    if mokdim_version:
        df = pd.read_csv(path/'Nadaln_with_features_and_distance_to_employment_centers.csv', na_values='None')
    else:
        df = pd.read_csv(path/'Nadaln_with_features.csv', na_values='None')
        print('sclicing to {} - {}.'.format(years[0], years[1]))
        df = df.loc[(df['Sale_year'] >= years[0]) & (df['Sale_year'] <= years[1])]
        print('choosing {} only.'.format(asset_type))
        df = df[df['Type_of_asset'].isin(asset_type)]
        print('adding to floor number.')
        floor1 = df.loc[(~df['Another_floor_1'].isnull())]['Another_floor_1']
        df.loc[floor1.index, 'Floor_number'] = floor1.values
    return df


def run_MLR_on_all_years(df, feats=features, dummy='Rooms_345'):
    import numpy as np
    import xarray as xr
    import statsmodels.api as sm

    # from sklearn.feature_selection import f_regression
    years = np.arange(2000, 2020, 1)
    das = []
    for year in years:
        X, y, scaler = produce_X_y(df, year=year, y_name='Price', plot_Xcorr=False,
                                   feats=feats, dummy=dummy, scale_X=True)
        # ml = ML_Classifier_Switcher()
        # mlr = ml.pick_model('MLR')
        # mlr.fit(X, y)
        # score = mlr.score(X, y)
        X2 = sm.add_constant(X)
        est = sm.OLS(y, X2)
        est = est.fit()
        # _, pval = f_regression(X, y)
        # beta = mlr.coef_
        pval = est.summary2().tables[1]['P>|t|'][1:]
        beta = est.summary2().tables[1]['Coef.'][1:]
        score = est.rsquared
        beta_da = xr.DataArray(beta, dims=['regressor'])
        beta_da.name = 'beta'
        pval_da = xr.DataArray(pval, dims=['regressor'])
        pval_da.name = 'pvalues'
        r2_da = xr.DataArray(score)
        r2_da.name = 'r2_score'
        ds = xr.merge([beta_da, pval_da, r2_da])
        ds['regressor'] = X.columns
        das.append(ds)
    ds = xr.concat(das, 'year')
    ds['year'] = years
    return ds


def plot_MLR_results(ds):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_theme(style='ticks', font_scale=1.5)
    fig, ax = plt.subplots(figsize=(17, 10))
    df = ds['beta'].to_dataset('regressor').to_dataframe()
    df = df.rename(plot_names, axis=1)
    df.index = pd.to_datetime(df.index, format='%Y')
    df.plot(ax=ax, legend=False)
    ax.legend(ncol=2, handleheight=0.1, labelspacing=0.01)
    ax.set_ylabel(r'$\beta$ coefficient')
    ax.grid(True)
    fig.tight_layout()
    return fig


def read_and_run_FI_on_all_years(df, path=ml_path, pgrid='light'):
    import numpy as np
    import xarray as xr
    years = np.arange(2000, 2020, 1)
    fis = []
    cvrs = []
    for year in years:
        print('running test on year {}.'.format(year))
        X, y, sc = produce_X_y(df, year=year, y_name='Price', plot_Xcorr=False,
                               feats=features, dummy=None, scale_X=False)
        df_r, gr = load_HP_params_from_optimized_model(path, pgrid=pgrid, year=year,
                                                       model_name='RF', return_df=False,
                                                       return_object=True)
        est = gr.best_estimator_
        fi = manual_cross_validation_for_RF_feature_importances(X, y, est,
                                                                n_splits=5,
                                                                n_repeats=None,
                                                                scorers=['r2'])
        cvr = cross_validate_using_optimized_HP(X, y, est, model='RF', n_splits=5,
                                                n_repeats=None, scorers=['r2'])
        fis.append(fi)
        cvrs.append(cvr)
    fi = xr.concat(fis, 'year')
    cvr = xr.concat(cvrs, 'year')
    ds = xr.merge([fi, cvr])
    ds['year'] = years
    return ds


def plot_RF_FI_results(ds):
    import pandas as pd
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    sns.set_theme(style='ticks', font_scale=1.5)
    fig, ax = plt.subplots(figsize=(17, 10))
    df = ds['feature_importances'].mean('repeats').to_dataset('regressor').to_dataframe()
    df = df * 100
    df = df.rename(plot_names, axis=1)
    df.index = pd.to_datetime(df.index, format='%Y')
    x = df.index
    ys=[df[x] for x in df.columns]
    ax.stackplot(x,*ys, labels=[x for x in df.columns])
    # df.plot(ax=ax, legend=False)
    ax.legend(loc='center', ncol=2, handleheight=0.1, labelspacing=0.01)
    # df_total = df.sum(axis=1)
    # df_rel = df[df.columns[1:]].div(df_total, 0)*100
    # for n in df_rel:
    #     for i, (cs, ab, pc) in enumerate(zip(df.iloc[:, 1:].cumsum(1)[n],
    #                                          df[n], df_rel[n])):
    #         ax.text(cs - ab / 2, i, str(np.round(pc, 1)) + '%',
    #                  va = 'center', ha = 'center')
    ax.set_ylabel('Feature importances [%]')
    ax.grid(True)
    fig.tight_layout()
    return fig

    return

def run_CV_on_all_years(df, model_name='RF', savepath=ml_path, pgrid='normal',
                        year=None, year_start=2010):
    import numpy as np
    if year is None:
        years = np.arange(2000, 2020, 1)
        if year_start is not None:
            years = np.arange(year_start, 2020, 1)
        for year in years:
            X, y, scaler = produce_X_y(df, year=year, y_name='Price', plot_Xcorr=False,
                                       feats=features, dummy=None, scale_X=False)
            # ml = ML_Classifier_Switcher()
            # model = ml.pick_model(model_name)
            cross_validation(X, y, model_name=model_name, n_splits=5, pgrid=pgrid,
                             savepath=savepath, verbose=0, n_jobs=-1, year=year)
    else:
        X, y, scaler = produce_X_y(df, year=year, y_name='Price', plot_Xcorr=False,
                                   feats=features, dummy=None, scale_X=False)
        # ml = ML_Classifier_Switcher()
        # model = ml.pick_model(model_name)
        cross_validation(X, y, model_name=model_name, n_splits=5, pgrid=pgrid,
                         savepath=savepath, verbose=0, n_jobs=-1, year=year)

    return


def produce_X_y(df, year=2015, y_name='Price', plot_Xcorr=True,
                feats=features, dummy='Rooms_345', scale_X=True):
    import seaborn as sns
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.preprocessing import PowerTransformer
    # from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    import pandas as pd
    # first, subset for year:
    # df = df[df['Type_of_asset'].isin(apts_more)]
    df = df[df['Sale_year'] == year]
    # now slice for the features:
    if feats is not None:
        print('picking {} as features.'.format(feats))
        X = df[feats].dropna()
        X.drop('Sale_year', axis=1, inplace=True)
    y = df.loc[X.index, y_name]
    X = X.reset_index(drop=True)
    if 'New' in X.columns:
        X['New'] = X['New'].astype(int)
    y = y.reset_index(drop=True)
    # df['Rooms'] = df['Rooms'].astype(int)
    # df['Floor_number'] = df['Floor_number'].astype(int)
    # df['New_apartment'] = df['New_apartment'].astype(int)
    if plot_Xcorr:
        sns.heatmap(X.corr(), annot=True)
    if 'Rooms_345' in X.columns:
        X['Rooms_345'] = X['Rooms_345'].astype(int)
    # do onhotencoding on rooms:
    # jobs_encoder = LabelBinarizer()
    # jobs_encoder.fit(X['Rooms_345'])
    # transformed = jobs_encoder.transform(X['Rooms_345'])
    # rooms = pd.DataFrame(transformed)
    # rooms.columns = ['3_Rooms', '4_Rooms', '5_Rooms']
    if dummy is not None:
        prefix = dummy.split('_')[0]
        rooms = pd.get_dummies(data=X[dummy], prefix=prefix)
        # drop one col from one-hot encoding not to fall into dummy trap!:
        X = pd.concat([X, rooms.drop('Rooms_4', axis=1)], axis=1)
        X = X.drop([dummy], axis=1)
    # scale Floor numbers:
    if 'Floor_number' in X.columns:
        X['Floor_number'] = np.log(X['Floor_number']+1)
    if any(X.columns.str.contains('mean_distance')):
        col = X.loc[:, X.columns.str.contains('mean_distance')].columns[0]
        X[col] = np.log(X[col])
    if 'distance_to_nearest_kindergarten' in X.columns:
        X['distance_to_nearest_kindergarten'] = np.log(
            X['distance_to_nearest_kindergarten'])
    if 'distance_to_nearest_school' in X.columns:
        X['distance_to_nearest_school'] = np.log(X['distance_to_nearest_school'])
    # X['Year_Built'] = np.log(X['Year_Built'])
    # finally, scale y to log10 and X to minmax 0-1:
    Xscaler = MinMaxScaler()
    #yscaler = MinMaxScaler()
    # yscaler = PowerTransformer(method='yeo-johnson',standardize=True)
    y_scaled = y.apply(np.log10)
    # y_scaled = yscaler.fit_transform(y_scaled.values.reshape(-1,1))
    y = pd.DataFrame(y_scaled, columns=[y_name])
    if scale_X:
        X, scaler = scale_df(X, scaler=Xscaler)
    y = y[y_name]
    return X, y, Xscaler  # , yscaler


def nadlan_simple_ML(df, year=2000, model_name='RF', feats=features1):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    X = df[[x for x in feats]]
    X = X.dropna()
    y = df['DEALAMOUNT']
    y = y.loc[X.index]
    if year is not None:
        X = X[X['year'] == year].drop('year', axis=1)
        y = y.loc[X.index]
    ml = ML_Classifier_Switcher()
    model = ml.pick_model(model_name)
    model.fit(X, y)
    if hasattr(model, 'feature_importances_'):
        di = dict(zip(X.columns, model.feature_importances_))
    elif hasattr(model, 'coef_'):
        di = dict(zip(X.columns, model.coef_))
    dff = pd.DataFrame(di, index=[0]).T
    dff.columns = ['feature_importances']
    dff = dff.sort_values('feature_importances', ascending=False)
    dff.plot(kind='barh')
    plt.figure()
    sns.heatmap(X.corr(), annot=True)
    return X, y, model


def cross_validation(X, y, model_name='RF', n_splits=5, pgrid='light',
                     savepath=None, verbose=0, n_jobs=-1, year=None):
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import KFold
    cv = KFold(n_splits, shuffle=True, random_state=1)
    ml = ML_Classifier_Switcher()
    model = ml.pick_model(model_name, pgrid=pgrid)
    param_grid = ml.param_grid
    gr = GridSearchCV(model, scoring='r2', param_grid=param_grid,
                      cv=cv, verbose=verbose, n_jobs=n_jobs)
    gr.fit(X, y)
    if savepath is not None:
        if year is not None:
            filename = 'GRSRCHCV_{}_{}_{}_{}.pkl'.format(model_name, n_splits,
                                                         pgrid, year)
        else:
            filename = 'GRSRCHCV_{}_{}_{}.pkl'.format(model_name, n_splits,
                                                      pgrid)
        save_gridsearchcv_object(gr, savepath, filename)
    return gr


def cross_validate_using_optimized_HP(X, y, estimator, model='RF', n_splits=5,
                                      n_repeats=20, scorers=['r2']):
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.model_selection import KFold
    from sklearn.model_selection import RepeatedKFold
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.model_selection import GroupShuffleSplit
    import xarray as xr
    import pandas as pd
    # logo = LeaveOneGroupOut()
    # gss = GroupShuffleSplit(n_splits=20, test_size=0.1, random_state=1)
    # from sklearn.metrics import make_scorer
    # X = produce_X()
    # if add_MLR2:
    #     X = add_enso2_and_enso_qbo_to_X(X)
    #     print('adding ENSO^2 and ENSO*QBO')
    # y = produce_y()
    # X = X.sel(time=slice('1994', '2019'))
    # y = y.sel(time=slice('1994', '2019'))
    # groups = X['time'].dt.year
    # scores_dict = {s: s for s in scorers}
    # if 'r2_adj' in scorers:
    #     scores_dict['r2_adj'] = make_scorer(r2_adj_score)
    # if 'MLR' not in model:
    #     hp_params = get_HP_params_from_optimized_model(path, model)
    # ml = ML_Classifier_Switcher()
    # ml_model = ml.pick_model(model_name=model)
    # if 'MLR' not in model:
    #     ml_model.set_params(**hp_params)
    print(estimator)
    # cv = TimeSeriesSplit(5)
    cv = KFold(n_splits, shuffle=True, random_state=1)
    # cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
    #                    random_state=1)
    # if strategy == 'LOGO-year':
    #     print('using LeaveOneGroupOut strategy.')
    cvr = cross_validate(estimator, X, y, scoring=scorers, cv=cv)
    df = pd.DataFrame(cvr)
    df.index.name = 'repeats'
    das = []
    for scorer in scorers:
        da = df['test_{}'.format(scorer)].to_xarray()
        das.append(da)
    ds = xr.merge(das)
    # elif strategy == 'GSS-year':
    #     print('using GroupShuffleSplit strategy.')
    #     cvr = cross_validate(ml_model, X, y, scoring=scores_dict, cv=gss,
    #                          groups=groups)
    # else:
    #     cvr = cross_validate(ml_model, X, y, scoring=scores_dict, cv=cv)
    return ds


def save_gridsearchcv_object(GridSearchCV, savepath, filename):
    import joblib
    print('{} was saved to {}'.format(filename, savepath))
    joblib.dump(GridSearchCV, savepath / filename)
    return
    # cvr = gr.cv_results_
    return GridSearchCV


def manual_cross_validation_for_RF_feature_importances(X, y, rf_model,
                                                       n_splits=5,
                                                       n_repeats=20,
                                                       scorers=['r2']):
    from sklearn.model_selection import KFold
    import xarray as xr
    import numpy as np
    from sklearn.model_selection import RepeatedKFold
    # from sklearn.metrics import make_scorer
    # scores_dict = {s: s for s in scorers}
    print(rf_model)
    # cv = TimeSeriesSplit(5)
    # cv = KFold(5, shuffle=True, random_state=2)  # , shuffle=False, random_state=42)
    if n_repeats is not None:
        print('chose {} repeats'.format(n_repeats))
        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                           random_state=42)
    else:
        # , shuffle=False, random_state=42)
        cv = KFold(n_splits, shuffle=True, random_state=42)
    fis = []
    for train_index, test_index in cv.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        rf_model.fit(X_train, y_train)
        if hasattr(rf_model, 'feature_importances_'):
            fis.append(rf_model.feature_importances_)
            name = 'feature_importances'
        elif hasattr(rf_model, 'coef_'):
            fis.append(rf_model.coef_)
            name = 'beta'
    fi = xr.DataArray(fis, dims=['repeats', 'regressor'])
    fi['repeats'] = np.arange(1, len(fis)+1)
    fi['regressor'] = X.columns
    fi.name = name
    # fi = sort_fi(fi)
    return fi


def compare_ml_models_years(df, ml_path=ml_path, pgrid='light',
                            model_name='RF', years=[2015, 2017]):
    cvrs = []
    fis = []
    for year in years:
        _, gr = load_HP_params_from_optimized_model(ml_path,
                                                    pgrid=pgrid,
                                                    model_name=model_name,
                                                    year=year)
        best = gr.best_estimator_
        X, y, Xscaler = produce_X_y(df, year=year, y_name='Price')
        cvr = cross_validate_using_optimized_HP(
            X, y, best)
        fi = manual_cross_validation_for_RF_feature_importances(X, y, best,
                                                                n_repeats=20,
                                                                n_splits=5)
        cvrs.append(cvr)
        fis.append(fi)

    cvr = concat_categories(cvrs[0], cvrs[1])
    beta = concat_categories(fis[0], fis[1])
    return cvr, beta


# def sort_fi(fi):
#     fi_mean = fi.mean('repeats')
#     dff = fi_mean.to_dataframe().sort_values('feature_importances')
#     da = dff.to_xarray()['feature_importances']
#     fi = fi.sortby(da)
#     return fi


def concat_categories(da1, da2, cat1_val=2015, cat2_val=2017, cat_name='Sale_year'):
    import xarray as xr
    from nadlan_EDA import convert_da_to_long_form_df
    import pandas as pd
    if isinstance(da1, xr.DataArray):
        df1 = convert_da_to_long_form_df(da1)

        df2 = convert_da_to_long_form_df(da2)
    else:
        df1 = da1
        df2 = da2
    df1[cat_name] = cat1_val
    df2[cat_name] = cat2_val
    df = pd.concat([df1, df2], axis=0)
    return df


def plot_feature_importances(fi, year=2017, mode='beta'):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    from nadlan_EDA import convert_da_to_long_form_df
    sns.set_theme(style='ticks', font_scale=1.5)
    fig, ax = plt.subplots(figsize=(17, 5))
    if not isinstance(fi, pd.DataFrame):
        df = convert_da_to_long_form_df(fi, value_name='feature_importances')
        if mode != 'beta':
            df['feature_importances'] = df['feature_importances'] * 100
        sns.barplot(data=df, x='feature_importances',
                    y='regressor', ci='sd', ax=ax)
    else:
        if mode != 'beta':
            fi['value'] = fi['value'] * 100
        sns.barplot(data=fi, x='value', y='regressor',
                    hue='Sale_year', ci='sd')
    ax.grid(True)
    if mode == 'beta':
        ax.set_xlabel('Coefficiants')
    else:
        ax.set_xlabel('Feature Importances [%]')
    ax.set_ylabel('Regressors')
    if year is not None:
        fig.suptitle('Year {}'.format(year))
    fig.tight_layout()
    return fig


def plot_cvr(cvr, y='test_r2', x='Model'):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    sns.set_theme(style='ticks', font_scale=1.5)
    fig, ax = plt.subplots(figsize=(6, 6))
    g = sns.barplot(data=cvr, x=x, y=y, hue='Sale_year', ci='sd')
    # g.despine(left=True)
    plt.legend(loc='upper left')
    ax.grid(True)
    ax.set_xlabel('Model')
    ax.set_ylabel(r'R$^2$')
    fig.tight_layout()
    return fig


def load_HP_params_from_optimized_model(path, pgrid='light', year=2000,
                                        model_name='RF', return_df=False, return_object=True):
    import joblib
    from Migration_main import path_glob
    files = path_glob(
        path, 'GRSRCHCV_{}_*_{}_{}.pkl'.format(model_name, pgrid, year))
    file = [x for x in files if model_name in x.as_posix()][0]
    gr = joblib.load(file)
    if return_object:
        df, gr = read_one_gridsearchcv_object(gr)
        return df, gr
    else:
        df = read_one_gridsearchcv_object(gr)
    if return_df:
        return df
    else:
        return df.iloc[0][:-2].to_dict()


def read_one_gridsearchcv_object(gr, return_object=True):
    """read one gridsearchcv multimetric object and
    get the best params, best mean/std scores"""
    import pandas as pd
    # first get all the scorers used:
    if isinstance(gr.scoring, list):
        scorers = [x for x in gr.scoring]
    elif isinstance(gr.scoring, str):
        scorers = ['score']
    # now loop over the scorers:
    best_params = []
    best_mean_scores = []
    best_std_scores = []
    for scorer in scorers:
        df_mean = pd.concat([pd.DataFrame(gr.cv_results_["params"]), pd.DataFrame(
            gr.cv_results_["mean_test_{}".format(scorer)], columns=[scorer])], axis=1)
        df_std = pd.concat([pd.DataFrame(gr.cv_results_["params"]), pd.DataFrame(
            gr.cv_results_["std_test_{}".format(scorer)], columns=[scorer])], axis=1)
        # best index = highest score:
        best_ind = df_mean[scorer].idxmax()
        best_mean_scores.append(df_mean.iloc[best_ind][scorer])
        best_std_scores.append(df_std.iloc[best_ind][scorer])
        best_params.append(df_mean.iloc[best_ind].to_frame().T.iloc[:, :-1])
    best_df = pd.concat(best_params)
    best_df['mean_score'] = best_mean_scores
    best_df['std_score'] = best_std_scores
    best_df.index = scorers
    if return_object:
        return best_df, gr
    else:
        return best_df


# def process_gridsearch_results(GridSearchCV, model_name,
#                                split_dim='inner_kfold', features=None,
#                                pwv_id=None, hs_id=None, test_size=None):
#     import xarray as xr
#     import pandas as pd
#     import numpy as np
#     # finish getting best results from all scorers togather
#     """takes GridSreachCV object with cv_results and xarray it into dataarray"""
#     params = GridSearchCV.param_grid
#     scoring = GridSearchCV.scoring
#     results = GridSearchCV.cv_results_
#     names = [x for x in params.keys()]
#     # unpack param_grid vals to list of lists:
#     pro = [[y for y in x] for x in params.values()]
#     ind = pd.MultiIndex.from_product((pro), names=names)
#     result_names = [
#         x for x in results.keys() if 'param' not in x]
#     ds = xr.Dataset()
#     for da_name in result_names:
#         da = xr.DataArray(results[da_name])
#         ds[da_name] = da
#     ds = ds.assign(dim_0=ind).unstack('dim_0')
#     for dim in ds.dims:
#         if ds[dim].dtype == 'O':
#             try:
#                 ds[dim] = ds[dim].astype(str)
#             except ValueError:
#                 ds = ds.assign_coords({dim: [str(x) for x in ds[dim].values]})
#         if ('True' in ds[dim]) and ('False' in ds[dim]):
#             ds[dim] = ds[dim] == 'True'
#     # get all splits data and concat them along number of splits:
#     all_splits = [x for x in ds.data_vars if 'split' in x]
#     train_splits = [x for x in all_splits if 'train' in x]
#     test_splits = [x for x in all_splits if 'test' in x]
#     return ds
#     # loop over scorers:
#     # trains = []
#     tests = []
#     for scorer in scoring:
#         train_splits_scorer = [x for x in train_splits if scorer in x]
#         # trains.append(xr.concat([ds[x]
#                                  # for x in train_splits_scorer], split_dim))
#         test_splits_scorer = [x for x in test_splits if scorer in x]
#         tests.append(xr.concat([ds[x] for x in test_splits_scorer], split_dim))
#         splits_scorer = np.arange(1, len(train_splits_scorer) + 1)
#     return tests
#     # train_splits = xr.concat(trains, 'scoring')
#     test_splits = xr.concat(tests, 'scoring')
#     return ds
# #    splits = [x for x in range(len(train_splits))]
# #    train_splits = xr.concat([ds[x] for x in train_splits], 'split')
# #    test_splits = xr.concat([ds[x] for x in test_splits], 'split')
#     # replace splits data vars with newly dataarrays:
#     ds = ds[[x for x in ds.data_vars if x not in all_splits]]
#     # ds['split_train_score'] = train_splits
#     ds['split_test_score'] = test_splits
#     ds[split_dim] = splits_scorer
#     if isinstance(scoring, list):
#         ds['scoring'] = scoring
#     elif isinstance(scoring, dict):
#         ds['scoring'] = [x for x in scoring.keys()]
#     ds.attrs['name'] = 'CV_results'
#     ds.attrs['param_names'] = names
#     ds.attrs['model_name'] = model_name
#     ds.attrs['{}_splits'.format(split_dim)] = ds[split_dim].size
#     if GridSearchCV.refit:
#         if hasattr(GridSearchCV.best_estimator_, 'feature_importances_'):
#             f_import = xr.DataArray(
#                 GridSearchCV.best_estimator_.feature_importances_,
#                 dims=['feature'])
#             f_import['feature'] = features
#             ds['feature_importances'] = f_import
#         ds['best_score'] = GridSearchCV.best_score_
# #        ds['best_model'] = GridSearchCV.best_estimator_
#         ds.attrs['refitted_scorer'] = GridSearchCV.refit
#         for name in names:
#             if isinstance(GridSearchCV.best_params_[name], tuple):
#                 GridSearchCV.best_params_[name] = ','.join(
#                     map(str, GridSearchCV.best_params_[name]))
#             ds['best_{}'.format(name)] = GridSearchCV.best_params_[name]
#         return ds, GridSearchCV.best_estimator_
#     else:
#         return ds, None


class ML_Classifier_Switcher(object):

    def pick_model(self, model_name, pgrid='normal'):
        """Dispatch method"""
        # from sklearn.model_selection import GridSearchCV
        self.param_grid = None
        method_name = str(model_name)
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, method_name, lambda: "Invalid ML Model")
#        if gridsearch:
#            return(GridSearchCV(method(), self.param_grid, n_jobs=-1,
#                                return_train_score=True))
#        else:
        # Call the method as we return it
        # whether to select lighter param grid, e.g., for testing purposes.
        self.pgrid = pgrid
        return method()

    def SVM(self):
        from sklearn.svm import SVR
        import numpy as np
        if self.pgrid == 'light':
            self.param_grid = {'kernel': ['poly'],
                               'C': [0.1],
                               'gamma': [0.0001],
                               'degree': [1, 2],
                               'coef0': [1, 4]}
        # elif self.pgrid == 'normal':
        #     self.param_grid = {'kernel': ['rbf', 'sigmoid', 'linear', 'poly'],
        #                        'C': order_of_mag(-1, 2),
        #                        'gamma': order_of_mag(-5, 0),
        #                        'degree': [1, 2, 3, 4, 5],
        #                        'coef0': [0, 1, 2, 3, 4]}
        elif self.pgrid == 'dense':
            # self.param_grid = {'kernel': ['rbf', 'sigmoid', 'linear', 'poly'],
            #                    'C': np.logspace(-2, 2, 10), # order_of_mag(-2, 2),
            #                    'gamma': np.logspace(-5, 1, 14), # order_of_mag(-5, 0),
            #                    'degree': [1, 2, 3, 4, 5],
            #                    'coef0': [0, 1, 2, 3, 4]}
            self.param_grid = {'kernel': ['rbf', 'sigmoid', 'linear'],
                               # order_of_mag(-2, 2),
                               'C': np.logspace(-2, 2, 10),
                               'gamma': np.logspace(-5, 1, 14)}  # , # order_of_mag(-5, 0),
            # 'degree': [1, 2, 3, 4, 5],
            # 'coef0': [0, 1, 2, 3, 4]}
        return SVR()

    def MLP(self):
        import numpy as np
        from sklearn.neural_network import MLPRegressor
        if self.pgrid == 'light':
            self.param_grid = {
                'activation': [
                    'identity',
                    'relu'],
                'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50)]}
        # elif self.pgrid == 'normal':
        #     self.param_grid = {'alpha': order_of_mag(-5, 1),
        #                        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        #                        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
        #                        'learning_rate': ['constant', 'adaptive'],
        #                        'solver': ['adam', 'lbfgs', 'sgd']}
        elif self.pgrid == 'dense':
            self.param_grid = {'alpha': np.logspace(-5, 1, 7),
                               'activation': ['identity', 'logistic', 'tanh', 'relu'],
                               'hidden_layer_sizes': [(10, 10, 10), (10, 20, 10), (10,), (5,), (1,)],
                               'learning_rate': ['constant', 'adaptive'],
                               'solver': ['adam', 'lbfgs', 'sgd']}
            #(1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),(11,), (12,),(13,),(14,),(15,),(16,),(17,),(18,),(19,),(20,),(21,)
        return MLPRegressor(random_state=42, max_iter=500, learning_rate_init=0.1)

    def RF(self):
        from sklearn.ensemble import RandomForestRegressor
        # import numpy as np
        if self.pgrid == 'light':
            self.param_grid = {'max_features': ['auto', 'sqrt'],
                               'max_depth': [5, 10],
                               'min_samples_leaf': [1, 2],
                               'min_samples_split': [2, 5],
                               'n_estimators': [100, 300]}
        elif self.pgrid == 'normal':
            self.param_grid = {'max_depth': [10, 15, 20],
                               'max_features': ['auto'],
                               'min_samples_leaf': [2, 5, 10],
                               'min_samples_split': [5, 10, 15],
                               'n_estimators': [300, 500, 700]
                               }
        elif self.pgrid == 'dense':
            self.param_grid = {'max_depth': [5, 10, 25, 50, 100, 150, 250],
                               'max_features': ['auto', 'sqrt'],
                               'min_samples_leaf': [1, 2, 5, 10, 15, 25],
                               'min_samples_split': [2, 5, 15, 30, 50, 70, 100],
                               'n_estimators': [100, 200, 300, 500, 700, 1000, 1300, 1500]
                               }
        return RandomForestRegressor(random_state=42, n_jobs=-1)

    def MLR(self):
        from sklearn.linear_model import LinearRegression
        return LinearRegression(n_jobs=-1)

    def Ridge(self):
        from sklearn.linear_model import Ridge
        import numpy as np
        if self.pgrid == 'light':
            self.param_grid = {'alpha': np.logspace(-5, 5, 11),
                               'solver': ['auto', 'svd', 'sparse_cg', 'saga']}
        elif self.pgrid == 'dense':
            self.param_grid = {'alpha': np.logspace(-10, 10, 21),
                               'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}

        return Ridge(normalize=False, random_state=1)

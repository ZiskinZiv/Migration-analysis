#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 15:41:04 2021
Run MLR hedonic with run_MLR_on_all_years(features=best1)
use plot_price_rooms_new_from_new_ds for time_series new rooms MLR
for standertized betas use plot_regular_feats_comparison_from_new_ds
For RF, HP tuning :
run_CV_on_all_years(df,savepath=ml_path,model_name='RF', feats=best_rf2+['SEI'])
Multifunction for RF results:
loop_over_RF_models_years(df, path=work_david/'ML', mode='score',
                          pgrid='normal'
use mode = 'score' to calculate the R^2 for training and test
use mode = 'time-series' to get the predictions.
use mode = 'shap' to calculate the SHAP values for the test sets.(warning this takes longest)
use mode = 'X_test' to get the test sets.
use mode = 'FI' to get feature importances.
then there are plot functions for RF and MLR:
1) plot_RF_time_series(time-series)
2) plot_RF_FI_results(fi)
3) First, produce MLR SHAPS: svs=produce_shap_MLR_all_years(df)
   then, produce_RF_abs_SHAP_all_years(path=ml_path/'RF_rooms_345',mlr_shap=svs)
4)

how to produce weighted mean distance to ECs for all Israeli settelments:
first load israeli settelment mid-points:
gdf=geo_location_settelments_israel() (from cbs_procedures)
then run calculate_distance_from_gdf_to_employment_centers:
dis = calculate_distance_from_gdf_to_employment_centers(gdf,n=18, x_coord_name='X', y_coord_name='Y')
finally save to csv:
dis.to_csv(work_david/'Israel_settlments_with_mean_weighted_distance_to_ECs.csv', na_rep='NA',sep=',', index=False)

@author: shlomi
"""
from MA_paths import work_david
from MA_paths import savefig_path
import numpy as np
ml_path = work_david / 'ML'
features = ['Floor_number', 'SEI', 'New', 'Periph_value', 'Sale_year', 'Rooms_345',
            'distance_to_nearest_kindergarten', 'distance_to_nearest_school', 'Total_ends']

features1 = ['FLOORNO', 'DEALNATURE', 'NEWPROJECTTEXT',
             'BUILDINGYEAR',  'SEI_value', 'Ground', 'P2015_value', 'year', 'Building_Growth_Rate']

features2 = ['FLOORNO', 'DEALNATURE', 'NEWPROJECTTEXT',
             'SEI_value', 'Ground', 'year', 'Building_Growth_Rate']

features3 = ['Floor_number', 'SEI', 'New', 'Periph_value', 'Sale_year', 'Rooms_345',
             'Total_ends', 'mean_distance_to_4_mokdim']

best = ['SEI', 'New', 'Sale_year', 'Rooms_345',
        'Total_ends', 'mean_distance_to_28_mokdim', 'Netflow']

best1 = ['SEI', 'New', 'Sale_year', 'Rooms_345',
        'Total_ends', 'mean_distance_to_28_mokdim']

best_years = best + ['year_{}'.format(x) for x in np.arange(2001, 2020)]
best_for_bs = best + ['city_code', 'Price']

next_best = ['Floor_number', 'New', 'Sale_year', 'Rooms',
             'Total_ends']

best_rf = ['SEI_value_2015', 'SEI_value_2017',
           'New', 'Sale_year', 'Rooms','Netflow',
           'Total_ends', 'mean_distance_to_28_mokdim']
best_rf1 = ['SEI_value_2015', 'SEI_value_2017',
           'New', 'Sale_year', 'Rooms',
           'Total_ends', 'mean_distance_to_28_mokdim']

best_rf2 = ['SEI_value_2015', 'SEI_value_2017',
           'New', 'Sale_year', 'Rooms_345',
           'Total_ends', 'mean_distance_to_28_mokdim']

dummies = ['New', 'Rooms_4', 'Rooms_5']

year_dummies = ['year_{}'.format(x) for x in np.arange(2001,2020)]

room_dummies = ['Rooms_4', 'Rooms_5']

best_regular = ['SEI', 'Total_ends', 'mean_distance_to_28_mokdim', 'Netflow']
best_regular1 = ['SEI', 'Total_ends', 'mean_distance_to_28_mokdim']

general_features = ['Price', 'Rooms', 'Area_m2', 'New', 'Floor_number', 'Floors_In_Building',
                    'Age', 'Total_ends', 'SEI', 'mean_distance_to_28_mokdim']

apts = ['דירה', 'דירה בבית קומות']
apts_more = apts + ["קוטג' דו משפחתי", "קוטג' חד משפחתי",
                    "דירת גן", "בית בודד", "דירת גג", "דירת גג (פנטהאוז)"]
plot_names = {'Floor_number': 'Floor',
              # 'New': 'New Apartment',
              'Periph_value': 'Peripheriality',
              'distance_to_nearest_kindergarten': 'Nearest kindergarten',
              'distance_to_nearest_school': 'Nearest school',
              'Total_ends': 'Building rate',
              'mean_distance_to_28_mokdim': 'Distance to ECs',
              'SEI': 'Socio-Economic Index',
              'SEI_value_2015': 'Social-Economic Index',
              'SEI_value_2017': 'Social-Economic Index',
              'Rooms': 'Rooms', 'Rooms_3': '3 Rooms', 'Rooms_5': '5 Rooms',
              'Netflow': 'Net migration',
              'MISH': 'AHP',
              'New': 'Used/New'
              }

short_plot_names = {'Total_ends': 'BR',
                    'mean_distance_to_28_mokdim': 'Distance',
                    'SEI': 'SEI', 'New': 'Used/New'}

vars_plot_names = {'Total_ends': 'BR',
                    'mean_distance_to_28_mokdim': 'DI',
                    'SEI': 'SE', 'New': 'NE', 'Rooms': 'RM'}

vars_explained_plot_names = {'Total_ends': 'BR (Building Rate)',
                    'mean_distance_to_28_mokdim': 'DI (Distance to ECs)',
                    'SEI': 'SE (Socio-Economic Index)', 'New': 'NE (Used/New)', 'Rooms': 'RM (# of Rooms)'}

add_units_dict = {'Distance': 'Distance [km]', 'BR': r'BR [Apts$\cdot$yr$^{-1}$]',
                  'Netflow': r'Netflow [people$\cdot$yr$^{-1}$]'}
add_units_dict_short = {'DI': 'DI [km]', 'BR': r'BR [Apts$\cdot$yr$^{-1}$]'}
# AHP : Afforable Housing Program


def pct_change(x):
    import numpy as np
    return (np.exp(x)-1)*100


def plot_single_tree(rf_model, X_train, y_train, est_index=100, samples=25, max_depth=2):
    from sklearn import tree
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    # rf = RandomForestRegressor(max_depth=15,n_estimators=250)
    # feats = ['Status', 'Rooms', 'BR', 'Distance', 'SEI']
    X_train = X_train.rename(vars_plot_names, axis=1)
    feats = ['NE', 'RM', 'BR', 'DI', 'SE']
    # sns.set_theme(font_scale=1.8)
    fig, ax = plt.subplots(figsize=(17, 10))
    inds = X_train.sample(n=samples).index
    y_train = np.log(np.exp(y_train)/4)
    rf_model.fit(X_train.loc[inds], y_train.loc[inds])
    _ = tree.plot_tree(rf_model[est_index],precision=2, fontsize=18, rounded=True,
                       feature_names=feats, filled=True, ax=ax, max_depth=max_depth, proportion=False)
    filename = 'Nadlan_tree_example.png'
    plt.savefig(savefig_path / filename, bbox_inches='tight', pad_inches=0.1)
    return fig



def compare_r2_RF_MLR(sc, ds, mode='diagram'):
    """compare R2 score from dataset (sc=loop_over with mode=score)
    and ds=run_MLR_on_all_years"""
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_theme(style='ticks', font_scale=1.6)
    fig, ax = plt.subplots(figsize=(17, 10))
    df = ds['R-squared'].to_dataframe()
    df = pd.concat([df, sc], axis=1)
    df.columns = ['Hedonic', 'RF train', 'RF test']
    df['year'] = df.index
    df = df.melt(id_vars=['year'], var_name='Model',
                 value_name=r'R$^2$')
    # df['year'] = pd.to_datetime(df['year'], format='%Y')
    if mode == 'diagram':
        ax = sns.barplot(data=df, x='year', ax=ax, hue='Model', y=r'R$^2$')
        # ax.set_ylabel('Apartment area [{}]'.format(unit_label))
        h, l =ax.get_legend_handles_labels()
        ax.legend_.remove()
        ax.legend(h, l, ncol=3, title='Model')
        ax.set_xlabel('')
        ax.grid(True)
        sns.despine(fig)
        fig.tight_layout()
    # for wide dataframe:
    # df = df.pivot_table(columns=['Model'],values='R$^2$',index='year')
    return df


def remove_outlier_area_per_room(df, col='Area_m2', k=1.5):
    from Migration_main import remove_outlier
    import pandas as pd
    dfs = []
    for room in df['Rooms'].dropna().unique():
        df1 = remove_outlier(df[df['Rooms'] == room], col_name=col, k=k)
        dfs.append(df1)
    df = pd.concat(dfs, axis=0)
    return df


def plot_rooms_area_distribution(df, units='m2'):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_theme(style='ticks', font_scale=1.8)
    fig, ax = plt.subplots(figsize=(17, 10))
    if units == 'ft2':
        df['Area_ft2'] = df['Area_m2'] * 10.764
        col = 'Area_ft2'
        unit_label = 'ft$^2$'
    elif units == 'm2':
        col = 'Area_m2'
        unit_label = 'm$^2$'
    sns.violinplot(data=df, x='Rooms', y=col, ax=ax, palette='inferno')
    ax.set_ylabel('Apartment area [{}]'.format(unit_label))
    ax.set_xlabel('Number of rooms')
    ax.grid(True)
    sns.despine(fig)
    fig.tight_layout()
    return fig


def plot_general_features_corr_heatmap(df, feats=general_features, year=None):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_theme(style='ticks', font_scale=1.5)
    fig, ax = plt.subplots(figsize=(17, 10))
    if year is not None:
        df = df[df['Sale_year']==year]
        title = 'year = {}'.format(year)
    else:
        title = '2000 to 2019'
    dff = df[feats]
    dff = dff.rename(short_plot_names, axis=1)
    g = sns.heatmap(dff.corr(),annot=True,cmap='coolwarm', ax=ax, center=0)
    g.set_xticklabels(g.get_xticklabels(), rotation=45, ha='right')
    fig.tight_layout()
    fig.suptitle(title)
    fig.subplots_adjust(top=0.945)
    return fig


def plot_RF_time_series(X_ts, units='nis'):
    """plot rooms new time series from RF model"""
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    from cbs_procedures import read_mean_salary
    sns.set_theme(style='ticks', font_scale=1.8)
    fig, ax = plt.subplots(figsize=(17, 10))
    X_ts = X_ts[X_ts['Rooms'].isin([3, 4, 5])]
    X_ts['Rooms'] = X_ts['Rooms'].astype(int)
    X_ts = X_ts.rename({'New': 'Used/New'}, axis=1)
    X_ts['Used/New'][X_ts['Used/New']==0] = 'Used'
    X_ts['Used/New'][X_ts['Used/New']==1] = 'New'
    if units == 'dollar':
        X_ts['Price'] /= 4 * 1000
        ylabel = 'Apartment Price [Thousands $]'
    elif units == 'nis':
        X_ts['Price'] /= 1e6
        ylabel = 'Apartment Price [millions NIS]'
    elif units == 'salary':
        sal = read_mean_salary().rename({'year': 'Year'}, axis=1)
        X_ts = pd.merge(X_ts, sal, on='Year', how='inner')
        X_ts['Price'] /= X_ts['mean_salary']
        ylabel = 'Mean salary'
    X_ts['Year'] = pd.to_datetime(X_ts['Year'], format='%Y')
    X_ts = X_ts.reset_index(drop=True)
    sns.lineplot(data=X_ts, x='Year', y='Price', hue='Rooms', style='Used/New',
                 ax=ax, palette='tab10', markers=True, markersize=10)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('')
    ax.grid(True)
    sns.despine(fig)
    fig.tight_layout()
    return fig


def produce_shap_MLR_all_years(df, feats=best1, abs_val=True):
    from sklearn.linear_model import LinearRegression
    import shap
    import numpy as np
    years = np.arange(2000, 2020, 1)
    svs = []
    for year in years:
        print(year)
        X, y = prepare_new_X_y_with_year(df, features=feats, year=year,
                                         y_name='Price')
        lr = LinearRegression()
        lr.fit(X, y)
        ex = shap.LinearExplainer(lr, X)
        shap_values = ex.shap_values(X)
        SV = convert_shap_values_to_pandas(shap_values, X)
        if abs_val:
            print('producing ABS SHAP.')
            SV = produce_abs_SHAP_from_df(SV, X, plot=False)
        svs.append(SV)
    return svs


def loop_over_RF_models_years(df, path=work_david/'ML', mode='score',
                              pgrid='normal', feats=best_rf2+['SEI']):
    import numpy as np
    import pandas as pd
    import shap
    import xarray as xr
    years = np.arange(2000, 2020, 1)
    train_scores = []
    test_scores = []
    x_tests = []
    fis = []
    # shaps = []
    for year in years:
        print(year)
        _, gr = load_HP_params_from_optimized_model(path, pgrid=pgrid,
                                                    year=year)
        rf = gr.best_estimator_
        X_train, X_test, y_train, y_test = produce_X_y_RF_per_year(df,
                                                                   year=year,
                                                                   verbose=0, feats=feats)
        rf.fit(X_train, y_train)
        if mode == 'score':
            train_scores.append(rf.score(X_train, y_train))
            test_scores.append(rf.score(X_test, y_test))
        elif mode == 'time-series':
            y_pred = rf.predict(X_test)
            y_pred = np.exp(y_pred)
            X_test['Price'] = y_pred
            X_test['Year'] = year
            X_test = X_test.reset_index(drop=True)
            x_tests.append(X_test)
        elif mode == 'shap':
            # rf.fit(X_train, y_train)
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_test.values)
            SV = convert_shap_values_to_pandas(shap_values, X_test)
            filename = 'Nadlan_SHAP_RF_{}.csv'.format(year)
            SV.to_csv(path/filename, index=False)
            # SV = SV.to_xarray().to_array('feature')
            # return SV, X_test
            # shaps.append(SV)
        elif mode == 'X_test':
            X_test.index.name = 'sample'
            filename = 'Nadlan_X_test_RF_{}.csv'.format(year)
            X_test.to_csv(path/filename, index=False)
            # x_tests.append(X_test.to_xarray().to_array('feature'))
        elif mode == 'FI':
            fi = pd.DataFrame(rf.feature_importances_).T
            fi.columns = X_train.columns
            fi['year'] = year
            fis.append(fi)
    if mode == 'score':
        sc = pd.DataFrame(train_scores)
        sc.columns = ['train_r2']
        sc['test_r2'] = test_scores
        sc.index = years
        return sc
    elif mode == 'time-series':
        X_ts = pd.concat(x_tests, axis=0)
        return X_ts
    elif mode == 'FI':
        FI = pd.concat(fis, axis=0)
        return FI
    # elif mode == 'shap':
    #     sv_da = xr.concat(shaps, 'year')
    #     sv_da['year'] = years
    #     sv_da.attrs['long_name'] = 'Shapley values via SHAP Python package.'
    #     sv_da.to_netcdf(path/'Nadlan_SHAP_RF_{}-{}.nc'.format(years[0], years[-1]))
    #     return sv_da
    # elif mode == 'X_test':
    #     X_ts = xr.concat(x_tests, 'year')
    #     X_ts['year'] = years
    #     X_ts.attrs['long_name'] = 'X_tests per year to use with the SHAP'
    #     X_ts.to_netcdf(path/'Nadlan_X_test_RF_{}-{}.nc'.format(years[0], years[-1]))
    #     return X_ts


def load_all_yearly_shap_values(path=work_david/'ML'):
    import numpy as np
    years = np.arange(2000, 2020, 1)
    svs = []
    X_tests = []
    for year in years:
        sv, X_test = load_yearly_shap_values(path, year)
        svs.append(sv)
        X_tests.append(X_test)
    return svs, X_tests


def load_yearly_shap_values(path=work_david/'ML', year=2000):
    import pandas as pd
    X_test = pd.read_csv(path/'Nadlan_X_test_RF_{}.csv'.format(year))
    shap_values = pd.read_csv(path/'Nadlan_SHAP_RF_{}.csv'.format(year))
    assert len(X_test)==len(shap_values)
    return shap_values, X_test


def load_shap_values(path=work_david/'ML', samples=10000,
                     interaction_too=True, rename=True):
    import pandas as pd
    import xarray as xr
    print('loading {} samples.'.format(samples))
    X_test = pd.read_csv(path/'X_test_RF_{}.csv'.format(samples))
    shap_values = pd.read_csv(path/'SHAP_values_RF_{}.csv'.format(samples))
    if rename:
        X_test = X_test.rename(short_plot_names, axis=1)
        shap_values = shap_values.rename(short_plot_names, axis=1)
    if interaction_too:
        print('loading interaction values too.')
        shap_interaction_values = xr.load_dataarray(path/'SHAP_interaction_values_RF_{}.nc'.format(samples))
        shap_interaction_values['feature1'] = X_test.columns
        shap_interaction_values['feature2'] = X_test.columns
        return X_test, shap_values, shap_interaction_values
    else:
        return X_test, shap_values


def plot_dependence(shap_values, X_test, x_feature='RM',
                    y_features=['DI', 'SE', 'BR'],
                    alpha=0.2, cmap=None, units='pct_change',
                    plot_size=1.5, fontsize=16, x_jitter=0.75):
    import shap
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.ticker as tck
    sns.set_theme(style='ticks', font_scale=1.2)
    fig, axes = plt.subplots(len(y_features), 1, sharex=True, figsize=(8, 10))
    X = X_test.copy()
    X = X.rename(vars_plot_names, axis=1)
    shap_values = shap_values.rename(vars_plot_names, axis=1)
    X = X.rename(add_units_dict_short, axis=1)
    # X['Old/New'] = X['Old/New'].astype(int)
    # new_dict = {0: 'Old', 1: 'New'}
    # X['Old/New'] = X['Old/New'].map(new_dict)
    if units == 'pct_change':
        shap_values = shap_values.apply(pct_change)
    for i, y in enumerate(y_features):
        y_new = add_units_dict_short.get(y, y)
        shap.dependence_plot(x_feature, shap_values.values, X, x_jitter=x_jitter,
                             dot_size=4, alpha=alpha, interaction_index=y_new,
                             ax=axes[i])
        if 'DI' in x_feature:
            axes[i].set_xlim(25, 150)
        if 'RM' in x_feature:
            axes[i].set_xlabel('RM [# of rooms]')
        cb = fig.axes[-1]
        mapp = cb.collections[1]
        fig.canvas.draw()
        cbar = fig.colorbar(mapp, ax=axes[i],aspect=50, pad=0.05,
                            label=y_new)
        cbar.set_alpha(0.85)
        cbar.draw_all()
        cb.remove()
        # cbar.ax.set_yticklabels(['Low', 'High'], fontsize=fontsize)
        # cbar.set_label('Predictor value')
        cbar.outline.set_visible(False)

        # axes[i].set_ylabel(axes[i].get_ylabel(), fontsize=fontsize)
        # axes[i].set_xlabel(axes[i].get_xlabel(), fontsize=fontsize)
        # axes[i].tick_params(labelsize=fontsize)
        axes[i].grid(True)
        if units == 'pct_change':
            la = 'Price change\nfor {} [%]'.format(x_feature)
            axes[i].set_ylabel(la)
    [ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize) for ax in fig.axes]
    [ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize) for ax in fig.axes]
    [ax.tick_params(labelsize=fontsize) for ax in fig.axes]
    [ax.yaxis.set_major_locator(tck.MaxNLocator(5)) for ax in fig.axes]
    fig.tight_layout()
    return fig


def plot_summary_shap_values(shap_values, X_test, alpha=0.7, cmap=None,
                             plot_size=1.5, fontsize=16, units='pct_change'):
    import shap
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_theme(style='ticks', font_scale=1.8)
    X_test = X_test.rename(vars_plot_names, axis=1)
    shap_values = shap_values.rename(vars_plot_names, axis=1)
    if units == 'pct_change':
        shap_values = shap_values.apply(pct_change)
    if cmap is None:
        shap.summary_plot(shap_values.values, X_test, alpha=alpha, plot_size=plot_size)
    else:
        if not isinstance(cmap, str):
            cm = cmap.get_mpl_colormap()
        else:
            cm = sns.color_palette(cmap, as_cmap=True)
        shap.summary_plot(shap_values.values, X_test, alpha=alpha, cmap=cm, plot_size=plot_size)
    if len(shap_values.shape) > 2:
        fig = plt.gcf()
        [ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize) for ax in fig.axes]
        [ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize) for ax in fig.axes]
        [ax.set_title(ax.get_title(), fontsize=fontsize) for ax in fig.axes]
        [ax.tick_params(labelsize=fontsize) for ax in fig.axes]
    else:
        fig, ax = plt.gcf(), plt.gca()
        if units == 'pct_change':
            ax.set_xlabel('Price change [%]', fontsize=fontsize)
        else:
            ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
        ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        cb = fig.axes[-1]
        cbar = fig.colorbar(cb.collections[1], ticks=[0, 1],
                                aspect=50, pad=0.05)
        cb.remove()
        cbar.ax.set_yticklabels(['Low', 'High'], fontsize=fontsize)
        cbar.set_label('Predictor value')
        cbar.ax.tick_params(size=0)
        cbar.outline.set_visible(False)
        # cb.set_ylabel(cb.get_ylabel(), fontsize=fontsize)
        # cb.tick_params(labelsize=fontsize)
    fig.tight_layout()
    return fig


def select_years_interaction_term(ds, regressor='SEI'):
    regs = ['{}_{}'.format(x, regressor) for x in year_dummies]
    ds = ds.sel(regressor=regs)
    return ds


def produce_RF_abs_SHAP_all_years(path=ml_path, plot=True, mlr_shap=None,
                                  units=None):
    import xarray as xr
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    SVs, X_tests = load_all_yearly_shap_values(path)
    k2s = []
    for i, year in enumerate(np.arange(2000, 2020, 1)):
        shap_df = SVs[i]
        # shap_df.drop('year', axis=1, inplace=True)
        X_test = X_tests[i]
        # X_test.drop('year', axis=1, inplace=True)
        k2 = produce_abs_SHAP_from_df(shap_df, X_test, plot=False)
        k2['year'] = year
        if mlr_shap is not None:
            k2['Model'] = 'RF'
            k2_mlr = mlr_shap[i]
            k2_mlr['year'] = year
            k2_mlr['Model'] = 'Hedonic'
            k2_mlr = k2_mlr[k2_mlr['Predictor'].isin(best_regular1)]
            k2 = pd.concat([k2, k2_mlr], axis=0)
        k2s.append(k2)
    abs_shap = pd.concat(k2s, axis=0)
    abs_shap = abs_shap.reset_index(drop=True)
    if plot:
        sns.set_theme(style='ticks', font_scale=1.6)
        fig, ax = plt.subplots(figsize=(17, 10))
        abs_shap['year'] = pd.to_datetime(abs_shap['year'], format='%Y')
        abs_shap = abs_shap[abs_shap['Predictor']!='New']
        abs_shap = abs_shap[abs_shap['Predictor']!='Rooms']
        # order:
        order = ['SE (Socio-Economic Index)', 'BR (Building Rate)', 'DI (Distance to ECs)']
        abs_shap['Predictor'] = abs_shap['Predictor'].map(vars_explained_plot_names)
        abs_shap['SHAP_abs'] *= np.sign(abs_shap['Corr'])
        if units == 'pct_change':
            abs_shap['SHAP_abs'] = abs_shap['SHAP_abs'].apply(pct_change)
        # order = ['Socio-Economic Index', 'Building rate', 'Distance to ECs']
        if mlr_shap is not None:
            sns.lineplot(data=abs_shap, x='year', y='SHAP_abs', hue='Predictor',
                         ax=ax, palette='Dark2', ci='sd', markers=True, linewidth=2,
                         hue_order=order, style='Model', markersize=10)
        else:
            sns.lineplot(data=abs_shap, x='year', y='SHAP_abs', hue='Predictor',
                         ax=ax, palette='Dark2', ci='sd', markers=True, linewidth=2,
                         hue_order=order, markersize=10)
        if units == 'pct_change':
            ax.set_ylabel('Price change [%]')
        else:
            ax.set_ylabel("mean |SHAP values|")
        ax.set_xlabel('')
        ax.grid(True)
        h, la = ax.get_legend_handles_labels()
        ax.legend_.remove()
        ax.legend(h, la, ncol=2, loc='center')
        sns.despine(fig)
        fig.tight_layout()
    return abs_shap


def produce_abs_SHAP_from_df(shap_df, X_test, plot=False):
    import pandas as pd
    shap_v = pd.DataFrame(shap_df)
    feature_list = X_test.columns
    shap_v.columns = feature_list
    df_v = X_test.copy().reset_index()#.drop('time', axis=1)
    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i], df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat(
        [pd.Series(feature_list), pd.Series(corr_list)], axis=1).fillna(0)
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns = ['Predictor', 'Corr']
    corr_df['Sign'] = np.where(corr_df['Corr'] > 0, 'red', 'blue')

    # Plot it
    shap_abs = np.abs(shap_v)
    k = pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Predictor', 'SHAP_abs']
    k2 = k.merge(corr_df, left_on='Predictor', right_on='Predictor', how='inner')
    k2 = k2.sort_values(by='SHAP_abs', ascending=True)
    if plot:
        colorlist = k2['Sign']
        ax = k2.plot.barh(x='Predictor', y='SHAP_abs',
                          color=colorlist, figsize=(5, 6), legend=False)
        ax.set_xlabel("SHAP Value (Red = Positive Impact)")
    return k2


def ABS_SHAP(df_shap, df):
    import numpy as np
    import pandas as pd
    import seaborn as sns
    sns.set_theme(style='ticks', font_scale=1.2)
    #import matplotlib as plt
    # Make a copy of the input data
    shap_v = pd.DataFrame(df_shap)
    feature_list = df.columns
    shap_v.columns = feature_list
    df_v = df.copy().reset_index()#.drop('time', axis=1)

    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i], df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat(
        [pd.Series(feature_list), pd.Series(corr_list)], axis=1).fillna(0)
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns = ['Predictor', 'Corr']
    corr_df['Sign'] = np.where(corr_df['Corr'] > 0, 'red', 'blue')

    # Plot it
    shap_abs = np.abs(shap_v)
    k = pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Predictor', 'SHAP_abs']
    k2 = k.merge(corr_df, left_on='Predictor', right_on='Predictor', how='inner')
    k2 = k2.sort_values(by='SHAP_abs', ascending=True)
    colorlist = k2['Sign']
    ax = k2.plot.barh(x='Predictor', y='SHAP_abs',
                      color=colorlist, figsize=(5, 6), legend=False)
    ax.set_xlabel("SHAP Value (Red = Positive Impact)")
    return


def plot_simplified_shap_tree_explainer(rf_model):
    import shap
    rf_model.fit(X, y)
    dfX = X.to_dataset('regressor').to_dataframe()
    dfX = dfX.rename(
        {'qbo_cdas': 'QBO', 'anom_nino3p4': 'ENSO', 'co2': r'CO$_2$'}, axis=1)
    ex_rf = shap.Explainer(rf_model)
    shap_values_rf = ex_rf.shap_values(dfX)
    ABS_SHAP(shap_values_rf, dfX)
    return


def convert_shap_values_to_pandas(shap_values, X_test):
    import pandas as pd
    SV = pd.DataFrame(shap_values)
    SV.columns = X_test.columns
    SV.index.name = 'sample'
    return SV


def plot_Tree_explainer_shap(rf_model, X_train, y_train, X_test, samples=1000):
    import shap
    from shap.utils import sample
    print('fitting...')
    rf_model.fit(X_train, y_train)
    # explain all the predictions in the test set
    print('explaining...')
    explainer = shap.TreeExplainer(rf_model)
    # rename features:
    X_test = X_test.rename(plot_names, axis=1)
    if samples is not None:
        print('using just {} samples out of {}.'.format(samples, len(X_test)))
        shap_values = explainer.shap_values(sample(X_test, samples).values)
        shap.summary_plot(shap_values, sample(X_test, samples))
        SV = convert_shap_values_to_pandas(shap_values, sample(X_test, samples))
    else:
        shap_values = explainer.shap_values(X_test.values)
        shap.summary_plot(shap_values, X_test)
        SV = convert_shap_values_to_pandas(shap_values, X_test)
    # shap.summary_plot(shap_values_rf, dfX, plot_size=1.1)
    return SV
# def get_mean_std_from_df_feats(df, feats=best, ignore=['New', 'Rooms_345', 'Sale_year'],
#                                log=['Total_ends']):
#     import numpy as np
#     f = [x for x in best if x not in ignore]
#     df1 = df.copy()
#     if log is not None:
#         df1[log] = (df1[log]+1).apply(np.log)
#     mean = df1[f].mean()
#     std = df1[f].std()
#     return mean, std


def produce_rooms_new_years_from_ds_var(ds, dsvar='beta_coef', new_cat='Used/New',
                                        new='New', old='Used'):
    import numpy as np
    import pandas as pd
    df = ds[dsvar].to_dataset('year').to_dataframe().T
    dfs = []
    # 3 rooms old:
    dff = df['const'].apply(np.exp).to_frame('Price')
    dff['Rooms'] = 3
    dff[new_cat] = old
    dfs.append(dff)
    # 3 rooms new:
    dff = (df['const']+df['New']).apply(np.exp).to_frame('Price')
    dff['Rooms'] = 3
    dff[new_cat] = new
    dfs.append(dff)
    # 4 rooms old:
    dff = (df['const']+df['Rooms_4']).apply(np.exp).to_frame('Price')
    dff['Rooms'] = 4
    dff[new_cat] = old
    dfs.append(dff)
    # 4 rooms new:
    dff = (df['const']+df['New']+df['Rooms_4']+df['Rooms_4_New']).apply(np.exp).to_frame('Price')
    dff['Rooms'] = 4
    dff[new_cat] = new
    dfs.append(dff)
    # 5 rooms old:
    dff = (df['const']+df['Rooms_5']).apply(np.exp).to_frame('Price')
    dff['Rooms'] = 5
    dff[new_cat] = old
    dfs.append(dff)
    # 5 rooms new:
    dff = (df['const']+df['New']+df['Rooms_5']+df['Rooms_5_New']).apply(np.exp).to_frame('Price')
    dff['Rooms'] = 5
    dff[new_cat] = new
    dfs.append(dff)
    dff = pd.concat(dfs, axis=0)
    dff['year'] = dff.index
    return dff


def calculate_pct_change_for_long_ds_var(ds_var_long, year=2000):
    d = ds_var_long.pivot(index='year', columns=[
                          'Rooms', 'Old/New'], values='Price')
    d_ref = d.loc[year]
    d /= d_ref
    d -= 1
    d *= 100
    d['year']=d.index
    df = d.melt(id_vars=['year'],value_name='Price')
    return df


def calculate_period_pct_change_from_ds(ds, syear=2008, eyear=2019):
    beta=produce_rooms_new_years_from_ds_var(ds,'beta_coef')
    beta = beta.pivot(index='year', columns=['Rooms', 'Used/New'],
                      values='Price')
    beta.columns = ['{}-{}'.format(rooms, new) for rooms, new in beta.columns]
    pct = 100 * (beta.loc[eyear] - beta.loc[syear]) / beta.loc[syear]
    return pct


def plot_price_rooms_new_from_new_ds(ds, add_cbs_index=False,
                                     units='nis'):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    from cbs_procedures import read_apt_price_index
    from cbs_procedures import read_mean_salary
    sns.set_theme(style='ticks', font_scale=1.8)
    fig, ax = plt.subplots(figsize=(17, 10))
    beta = produce_rooms_new_years_from_ds_var(ds, 'beta_coef')
    # calculate pct change between 2008 and 2019:
    pct = (beta.loc[2019,'Price'].values-beta.loc[2008,'Price'].values)/beta.loc[2008,'Price'].values
    pct *= 100
    beta1 = beta.copy()
    beta1.loc[2019, 'pct_change_2019_2008'] = pct
    print(beta1.loc[2019])
    # calculate pct change Old/New in 2008:
    pct=(beta[beta['Used/New']=='New'].loc[2008,'Price']-beta[beta['Used/New']=='Used'].loc[2008,'Price'])/beta[beta['Used/New']=='Used'].loc[2008,'Price']
    pct *= 100
    print(pct)
    # calculate pct change Old/New in 2019:
    pct=(beta[beta['Used/New']=='New'].loc[2019,'Price']-beta[beta['Used/New']=='Used'].loc[2019,'Price'])/beta[beta['Used/New']=='Used'].loc[2019,'Price']
    pct *= 100
    print(pct)
    upper = produce_rooms_new_years_from_ds_var(ds, 'CI_95_upper')
    lower = produce_rooms_new_years_from_ds_var(ds, 'CI_95_lower')
    if units == 'pct_change':
        beta = calculate_pct_change_for_long_ds_var(beta, 2000)
        upper = calculate_pct_change_for_long_ds_var(upper, 2000)
        lower = calculate_pct_change_for_long_ds_var(lower, 2000)
    df = pd.concat([lower, beta, upper], axis=0)

    if units == 'dollar':
        # approx 4 NIS to 1 $ in whole 2000-2019
        df['Price'] /= 4 * 1000  # price in thousands of $
        ylabel = 'Apartment Price [Thousands $]'
    elif units == 'nis':
        ylabel = 'Apartment Price [millions NIS]'
        df['Price'] /= 1e6
    elif units == 'salary':
        sal = read_mean_salary()
        df = pd.merge(df, sal, on='year', how='inner')
        df['Price'] /= df['mean_salary']
        ylabel = 'Mean salary'
    elif units == 'pct_change':
        ylabel = 'Apartment price change from 2000 [%]'
    df['year'] = pd.to_datetime(df['year'], format='%Y')
    df = df.reset_index(drop=True)
    sns.lineplot(data=df, x='year', y='Price', hue='Rooms', style='Used/New',
                 ax=ax, palette='tab10', ci='sd', markers=True, markersize=10)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('')
    if add_cbs_index:
        cbs = read_apt_price_index(path=work_david, resample='AS',
                                   normalize_year=2000)
        cbs = cbs.loc['2000':'2019']
        if units == 'pct_change':
            cbs /= cbs.iloc[0]
            cbs -= 1
            cbs *= 100
            cbs_label = 'Dwellings price index change from 2000 [%]'
        cbs.columns = ['Apartment Price Index']
        cbs['year'] = pd.to_datetime(cbs.index, format='%Y')
        if units != 'pct_change':
            twin = ax.twinx()
        else:
            twin = ax
        sns.lineplot(data=cbs, x='year', y='Apartment Price Index', ax=twin,
                     color='k', linewidth=2)
        twin.set_ylabel('Dwellings Price Index')
        twin.set_xlabel('')
        twin.set_ylim(50, 300)
    ax.grid(True)
    sns.despine(fig)
    fig.tight_layout()
    return fig


def plot_regular_feats_comparison_from_new_ds(ds,reg_name='Predictor',
                                              feats=best_regular1, units='pct_change'):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    sns.set_theme(style='ticks', font_scale=1.8)
    fig, ax = plt.subplots(figsize=(17, 10))
    dfs = []
    df = ds['beta_coef'].to_dataset('year').to_dataframe().T
    dff = df[feats].melt(ignore_index=False)
    dff['year'] = dff.index
    dfs.append(dff)
    df = ds['CI_95_upper'].to_dataset('year').to_dataframe().T
    dff = df[feats].melt(ignore_index=False)
    dff['year'] = dff.index
    dfs.append(dff)
    df = ds['CI_95_lower'].to_dataset('year').to_dataframe().T
    dff = df[feats].melt(ignore_index=False)
    dff['year'] = dff.index
    dfs.append(dff)
    dff = pd.concat(dfs, axis=0)
    dff['regressor'] = dff['regressor'].map(vars_explained_plot_names)
    dff = dff.rename({'regressor': reg_name}, axis=1)
    dff['year'] = pd.to_datetime(dff['year'], format='%Y')
    dff = dff.reset_index(drop=True)
    if units == 'pct_change':
        dff['value'] = dff['value'].apply(pct_change)
    sns.lineplot(data=dff, x='year', y='value', hue=reg_name,
                 ax=ax, ci='sd', markers=True,
                 palette='Dark2')
    if units == 'pct_change':
        ylabel = 'Price change [%]'
    else:
        ylabel = r'Standardized $\beta$s'
    ax.set_ylabel(ylabel)
    ax.set_xlabel('')
    h, l = ax.get_legend_handles_labels()
    ax.legend_.remove()
    ax.legend(h, l, ncol=1, title='Predictor', loc='center')
    ax.grid(True)
    sns.despine(fig)
    fig.tight_layout()
    return dff


def prepare_new_X_y_with_year(df, year=2000, y_name='Price', features=best1):
    import pandas as pd

    def return_X_with_interaction(X, dummy_list, var_list):
        Xs = []
        for num_var in var_list:
            X1 = get_design_with_pair_interaction(
                X, dummy_list+[num_var])
            Xs.append(X1)
        X1 = pd.concat(Xs, axis=1)
        X1 = X1.loc[:, ~X1.columns.duplicated()]
        return X1

    # m, s = get_mean_std_from_df_feats(df)
    X, y, scaler = produce_X_y(
        df, y_name=y_name, year=year, feats=features, dummy='Rooms_345',
        plot_Xcorr=True, scale_X=True)
    # X[best_regular] -= m
    # X[best_regular] /= s
    # regular vars vs. time (years):
    # X1 = return_X_with_interaction(X, ['trend'], best_regular)
    # rooms dummies and new:
    X2 = return_X_with_interaction(X, room_dummies, ['New'])
    # rooms dummies and years:
    # X3 = return_X_with_interaction(X, ['trend'], room_dummies)
    # New and years:
    # X4 = return_X_with_interaction(X, year_dummies, ['New'])
    X = pd.concat([X, X2],axis=1) #, X3, X4], axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    return X, y


def prepare_new_X_y(df, y_name='Price'):
    import pandas as pd

    def return_X_with_interaction(X, dummy_list, var_list):
        Xs = []
        for num_var in var_list:
            X1 = get_design_with_pair_interaction(
                X, dummy_list+[num_var])
            Xs.append(X1)
        X1 = pd.concat(Xs, axis=1)
        X1 = X1.loc[:, ~X1.columns.duplicated()]
        return X1

    X, y, scaler = produce_X_y(
        df, y_name=y_name, year=None, feats=best_years, dummy='Rooms_345',
        plot_Xcorr=True, scale_X=True)
    # regular vars vs. time (years):
    X1 = return_X_with_interaction(X, year_dummies, best_regular)
    # rooms dummies and new:
    X2 = return_X_with_interaction(X, room_dummies, ['New'])
    # rooms dummies and years:
    # X3 = return_X_with_interaction(X, year_dummies, room_dummies)
    # New and years:
    # X4 = return_X_with_interaction(X, year_dummies, ['New'])
    X = pd.concat([X1, X2],axis=1) #, X3, X4], axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    return X, y


def prepare_new_X_y_with_trend(df, y_name='Price'):
    import pandas as pd

    def return_X_with_interaction(X, dummy_list, var_list):
        Xs = []
        for num_var in var_list:
            X1 = get_design_with_pair_interaction(
                X, dummy_list+[num_var])
            Xs.append(X1)
        X1 = pd.concat(Xs, axis=1)
        X1 = X1.loc[:, ~X1.columns.duplicated()]
        return X1

    X, y, scaler = produce_X_y(
        df, y_name=y_name, year='trend', feats=best, dummy='Rooms_345',
        plot_Xcorr=True, scale_X=True)
    # regular vars vs. time (years):
    X1 = return_X_with_interaction(X, ['trend'], best_regular)
    # rooms dummies and new:
    X2 = return_X_with_interaction(X, room_dummies, ['New'])
    # rooms dummies and years:
    X3 = return_X_with_interaction(X, ['trend'], room_dummies)
    # New and years:
    # X4 = return_X_with_interaction(X, year_dummies, ['New'])
    X = pd.concat([X1, X2, X3],axis=1) #, X3, X4], axis=1)
    X = X.loc[:, ~X.columns.duplicated()]
    return X, y


def get_design_with_pair_interaction(data, group_pair):
    """ Get the design matrix with the pairwise interactions

    Parameters
    ----------
    data (pandas.DataFrame):
       Pandas data frame with the two variables to build the design matrix of their two main effects and their interaction
    group_pair (iterator):
       List with the name of the two variables (name of the columns) to build the design matrix of their two main effects and their interaction

    Returns
    -------
    x_new (pandas.DataFrame):
       Pandas data frame with the design matrix of their two main effects and their interaction

    """
    import pandas as pd
    import itertools
    x = pd.get_dummies(data[group_pair])
    interactions_lst = list(
        itertools.combinations(
            x.columns.tolist(),
            2,
        ),
    )
    x_new = x.copy()
    for level_1, level_2 in interactions_lst:
        if level_1.split('_')[0] == level_2.split('_')[0]:
            continue
        x_new = pd.concat(
            [
                x_new,
                x[level_1] * x[level_2]
            ],
            axis=1,
        )
        x_new = x_new.rename(
            columns = {
                0: (level_1 + '_' + level_2)
            }
        )
    return x_new


def calculate_distance_from_gdf_to_employment_centers(gdf, path=work_david, n=4,
                                                      weights='Pop2020', inverse=None,
                                                      x_coord_name='ITM-E', y_coord_name='ITM-N'):
    from cbs_procedures import read_emploment_centers_2008
    import numpy as np
    gdf = gdf[~gdf[x_coord_name].isnull()]
    gdf = gdf[~gdf[y_coord_name].isnull()]

    def mean_distance_to_n_mokdim(x, weights=None):
        # x = gdf['geometry']
        dists = points.distance(x).to_frame('distance')
        dists['Pop2020'] = points['Pop2020'] / 1000
        dists = dists.sort_values('distance')
        if inverse is not None:
            dists['distance'] = dists['distance']**inverse
            # return dists['distance'].mean()
        if weights is None:
            mean_dist = dists.iloc[0:n].mean()
        else:
            mean_dist = np.average(
                dists.iloc[0:n]['distance'], weights=dists.iloc[0:n][weights])
        return mean_dist.item()

    points = read_emploment_centers_2008(path, shape=True)
    if n is not None:
        gdf['mean_distance_to_{}_mokdim'.format(n)] = gdf['geometry'].apply(
            mean_distance_to_n_mokdim, weights=weights)
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


def calc_vif(X, dropna=True, asfloat=True, remove_mean=True):
    import pandas as pd
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    if dropna:
        print('dropping na.')
        X = X.dropna()
    if asfloat:
        print('considering as float.')
        X = X.astype(float)
    if remove_mean:
        X = X - X.mean()
    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(
        X.values, i) for i in range(X.shape[1])]
    return(vif)


def interpert_beta_coefs(ds, name='beta_coef', dummies=dummies):
    import numpy as np
    import xarray as xr
    ds1 = ds[name].to_dataset('regressor')
    if len(ds1.dims) == 0:
        df = ds1.expand_dims('dumm').to_dataframe()
    else:
        df = ds1.to_dataframe()
    betas = []
    # interpet dummy variables:
    for dummy in dummies:
        print('interperting {} variable.'.format(dummy))
        ser = 100*(np.exp(df[dummy])-1)
        da = ser.to_xarray()
        betas.append(da)
    # interpet regular log variables:
    # for every 10% change in var, the predicted log var is changed...:
    regulars = [x for x in ds['regressor'].values if x not in dummies]
    if 'const' in regulars:
        regulars.remove('const')
    if 'dumm' in regulars:
        regulars.remove('dumm')
    for regular in regulars:
        print('interperting {} variable.'.format(regular))
        ser = 100*(1.1**df[regular]-1)
        da = ser.to_xarray()
        betas.append(da)
    # now, the constant is the geometric mean of the Price:
    da = np.exp(df['const']).to_xarray()
    betas.append(da)
    beta = xr.merge(betas)
    try:
        beta = beta.to_array('regressor').drop('dumm')
    except ValueError:
        beta = beta.to_array('regressor')
    # beta = beta.sortby(ds['regressor'])
    ds['{}_inter'.format(name)] = beta.transpose().squeeze()
    return ds


def scale_log(df, cols=None, plus1_cols=None):
    import numpy as np
    import pandas as pd
    if cols is None:
        df_scaled = df.copy()
        for col in df.columns:
            if plus1_cols is None:
                df_scaled[col] = df[col].apply(np.log)
            else:
                print('{} is scaled using log(x+1)!'.format(col))
                df_scaled[col] = (df[col]+1).apply(np.log)
    else:
        print('scaling only {} cols.'.format(cols))
        df_sliced = df[cols]
        df_scaled = df_sliced.copy()
        for col in df_sliced.columns:
            if plus1_cols is None:
                df_scaled[col] = df_sliced[col].apply(np.log)
            else:
                print('{} is scaled using log(x+1)!'.format(col))
                df_scaled[col] = (df[col]+1).apply(np.log)
        df_rest = df[[x for x in df.columns if x not in cols]]
        df_scaled = pd.concat([df_scaled, df_rest], axis=1)
        df_scaled = df_scaled[[x for x in df.columns]]
    return df_scaled


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


def load_nadlan_with_features(path=work_david, years=[2000, 2019], asset_type=apts,
                              mokdim_version=False):
    import pandas as pd
    from cbs_procedures import read_price_index
    from nadlan_procedures import remove_outlier_by_value_counts

    def add_inflation_func(grp, pi,name='Price_inflation_fixed'):
        year = grp['Sale_year'].unique()[0]
        weight = pi.loc[str(year)].item()
        grp[name] = weight*grp['Price']
        return grp

    if mokdim_version:
        df = pd.read_csv(
            path/'Nadaln_with_features_and_distance_to_employment_centers.csv', na_values='None')
    else:
        df = pd.read_csv(path/'Nadaln_with_features.csv', na_values='None')
        print('sclicing to {} - {}.'.format(years[0], years[1]))
        df = df.loc[(df['Sale_year'] >= years[0]) &
                    (df['Sale_year'] <= years[1])]
        print('choosing {} only.'.format(asset_type))
        df = df[df['Type_of_asset'].isin(asset_type)]
        print('adding to floor number.')
        floor1 = df.loc[(~df['Another_floor_1'].isnull())]['Another_floor_1']
        df.loc[floor1.index, 'Floor_number'] = floor1.values
        print('adding Netflow')
        df['Netflow'] = df['Inflow']-df['Outflow']
        # shift the 0 of netflow to +10000 so that i could
        # use np.log afterwards in features preproccesing
        # also shift the SEI +3 for the log:
        # df['SEI'] += 3
        # now use price inflation fixing:
        # pi = read_price_index()['price_index_without_housing']
        # pi = 100 / pi
        # df = df.groupby('Sale_year').apply(add_inflation_func, pi)
        #create linear trend from dt:
        df = df.set_index(pd.to_datetime(df['Date'])).sort_index()
        df['trend'] = df.index.to_julian_date()
        df['trend'] -= df['trend'].iloc[0]
        df['trend'] /= df['trend'].iloc[-1]
        # remove number of rooms that are rare:
        df = remove_outlier_by_value_counts(df, 'Rooms', thresh=900)
        # remove outliers in rooms:
        df = remove_outlier_area_per_room(df, col='Area_m2', k=1.5)
        # remove 1 rooms with area > 50 m^2:
        d = df[df['Rooms']==1]['Area_m2'].dropna()
        inds = d[d>50].index
        df = df.drop(inds, axis=0)
        df = df.reset_index(drop=True)
    return df


def create_bootstrapped_samples_for_each_city_code_and_year(df, cols=best_for_bs,
                                                            min_deals=500,
                                                            min_years=20,
                                                            n_items=400,
                                                            n_samples=5):
    import pandas as pd
    df2 = df[cols].dropna()
    df2 = df2.reset_index(drop=True)
    df1 = filter_df_by_minimum_deals_per_year(
        df2, min_deals=min_deals, min_years=min_years, col='Price')
    years = [x for x in df1.columns]
    cities = [x for x in df1.index]
    dfs = []
    cnt = 0
    for year in years:
        for city in cities:
            for i in range(n_samples):
                df3 = df2[(df2['city_code'] == city) & (df2['Sale_year'] == year)].sample(n=n_items,
                                                                                          replace=False,
                                                                                          random_state=cnt)
                cnt += 1
                dfs.append(df3)
    dff = pd.concat(dfs, axis=0)
    dff = dff.reset_index(drop=True)
    return dff


def filter_df_by_minimum_deals_per_year(df, min_deals=200, min_years=20, col='Price'):
    df1 = df.groupby(['city_code', 'Sale_year'])[col].count().unstack()
    n_total_cities = len(df1)
    print('original number of cities: ', n_total_cities)
    df1 = df1[df1.count(axis=1) == min_years]
    n_years_cities = len(df1)
    print('number of cities with {} years total: '.format(
        min_years), n_years_cities)
    df1 = df1[df1 >= min_deals].dropna()
    n_deals_cities = len(df1)
    print('number of cities with minimum {} deals: '.format(
        min_deals), n_deals_cities)
    # sort:
    df1 = df1.sort_values(by=[x for x in df1.columns], axis=0, ascending=False)
    return df1


def convert_statsmodels_object_results_to_xarray(est):
    import pandas as pd
    import xarray as xr
    # get main regression results per predictor:
    t1 = est.summary().tables[1].as_html()
    t1 = pd.read_html(t1, header=0, index_col=0)[0]
    t1.columns = ['beta_coef', 'std_err', 't',
                  'P>|t|', 'CI_95_lower', 'CI_95_upper']
    t1.index.name = 'regressor'
    # get general results per all the data:
    t0 = est.summary().tables[0].as_html()
    t0 = pd.read_html(t0, header=None)[0]
    t0_ser1 = t0.loc[:, [0, 1]].set_index(0)[1]
    t0_ser1.index.name = ''
    t0_ser2 = t0.loc[:, [2, 3]].set_index(2)[3].dropna()
    t0_ser2.index.name = ''
    t0 = pd.concat([t0_ser1, t0_ser2])
    t0.index = t0.index.str.replace(':', '')
    t2 = est.summary().tables[2].as_html()
    t2 = pd.read_html(t2, header=None)[0]
    t2_ser1 = t2.loc[:, [0, 1]].set_index(0)[1]
    t2_ser1.index.name = ''
    t2_ser2 = t2.loc[:, [2, 3]].set_index(2)[3].dropna()
    t2_ser2.index.name = ''
    t2 = pd.concat([t2_ser1, t2_ser2])
    t2.index = t2.index.str.replace(':', '')
    t = pd.concat([t0, t2])
    t.index.name = 'variable'
    ds = t.to_xarray().to_dataset('variable')
    ds = xr.merge([ds, t1.to_xarray()])
    return ds


def run_MLR_on_all_years(df, feats=best1, dummy='Rooms_345', scale_X=False):
    import numpy as np
    import xarray as xr
    import statsmodels.api as sm

    # from sklearn.feature_selection import f_regression
    years = np.arange(2000, 2020, 1)
    das = []
    for year in years:
        X, y = prepare_new_X_y_with_year(df, features=best1, year=year, y_name='Price')
        # X, y, scaler = produce_X_y(df, year=year, y_name='Price', plot_Xcorr=False,
                                   # feats=feats, dummy=dummy, scale_X=scale_X)
        vif = calc_vif(X).set_index('variables')
        vif.index.name = 'regressor'
        vif = vif.to_xarray()['VIF']
        # ml = ML_Classifier_Switcher()
        # mlr = ml.pick_model('MLR')
        # mlr.fit(X, y)
        # score = mlr.score(X, y)
        X2 = sm.add_constant(X)
        est = sm.OLS(y, X2)
        est = est.fit()
        # _, pval = f_regression(X, y)
        # beta = mlr.coef_
        # pval = est.summary2().tables[1]['P>|t|'][1:]
        # beta = est.summary2().tables[1]['Coef.'][1:]
        # score = est.rsquared
        # beta_da = xr.DataArray(beta, dims=['regressor'])
        # beta_da.name = 'beta'
        # pval_da = xr.DataArray(pval, dims=['regressor'])
        # pval_da.name = 'pvalues'
        # r2_da = xr.DataArray(score)
        # r2_da.name = 'r2_score'
        # ds = xr.merge([beta_da, pval_da, r2_da])
        # ds['regressor'] = X.columns
        ds = convert_statsmodels_object_results_to_xarray(est)
        ds['VIF'] = vif
        das.append(ds)
    ds = xr.concat(das, 'year')
    ds['year'] = years
    return ds


# def plot_MLR_const(ds):
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     import pandas as pd
#     import numpy as np
#     df = ds['P>|t|'].to_dataset('regressor').to_dataframe()
#     sns.set_theme(style='ticks', font_scale=1.8)
#     const = 10**(ds['beta_coef'].sel(regressor='const')) / 1e6
#     CI95_u = ds['CI_95_upper'].to_dataset('regressor').to_dataframe()['const']
#     CI95_l = ds['CI_95_lower'].to_dataset('regressor').to_dataframe()['const']
#     CI95_u = 10**CI95_u / 1e6
#     CI95_l = 10**CI95_l / 1e6
#     years = pd.to_datetime(const.year.values, format='%Y')
#     fig, ax = plt.subplots(figsize=(17, 10))
#     errors = np.array(
#         list(zip((const.values-CI95_l.values), (CI95_u.values-const.values))))
#     errors = np.abs(errors)
#     g = ax.errorbar(years, const.values, errors.T, label='Mean Apartment')
#     ax.set_xlabel('')
#     ax.legend()
#     ax.grid(True)
#     ax.set_ylabel('Price [Millions NIS]')
#     fig.tight_layout()
#     return


def plot_MLR_field(ds, field='VIF', title='Variance Inflation Factor'):
    import seaborn as sns
    import matplotlib.pyplot as plt
    ds = ds.drop_sel(regressor='const')
    df = ds[field].to_dataset('regressor').to_dataframe()
    sns.set_theme(style='ticks', font_scale=1.2)
    df = df.rename(plot_names, axis=1)
    fig, ax = plt.subplots(figsize=(17, 10))
    g = sns.heatmap(df.T.round(2), annot=True, ax=ax, cmap='Reds')
    g.set_xticklabels(g.get_xticklabels(), rotation=30)
    ax.set_xlabel('')
    fig.suptitle(title)
    fig.tight_layout()
    return
    # matplotlib.rc_file_defaults()
    # ax = sns.set_style(style=None, rc=None )


def run_MLR_years_as_dummies(df):
    import statsmodels.api as sm
    X, y, scaler = produce_X_y(
        df, year=None, feats=best_years, dummy='Rooms_345', plot_Xcorr=True, scale_X=False)
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2)
    est = est.fit()
    ds = convert_statsmodels_object_results_to_xarray(est)
    return ds


def plot_MLR_years_as_dummies_interpeted_results(ds):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib.ticker as ticker

    def convert_da_to_df(ds, da='beta_coef_inter'):
        df = ds[da].to_dataset('regressor').expand_dims('dumm').to_dataframe()
        return df

    def create_errors_from_CI(value, CI_lower, CI_upper):
        err_l = (value-CI_lower).abs().squeeze()
        err_u = (CI_upper-value).abs().squeeze()
        err = np.array(list(zip(err_l.values, err_u.values))).T
        err = pd.DataFrame(err)
        err.columns = value.columns
        err.index = ['lower', 'upper']
        return err

    sns.set_theme(style='ticks', font_scale=1.5)
    ds = interpert_beta_coefs(ds, name='CI_95_upper', dummies=dummies+year_dummies)
    ds = interpert_beta_coefs(ds, name='CI_95_lower', dummies=dummies+year_dummies)
    ds = interpert_beta_coefs(ds, name='beta_coef', dummies=dummies+year_dummies)
    ds = ds.drop_sel(regressor='const')
    beta = convert_da_to_df(ds, da='beta_coef_inter')
    upper = convert_da_to_df(ds, da='CI_95_upper_inter')
    lower = convert_da_to_df(ds, da='CI_95_lower_inter')
    errors = create_errors_from_CI(beta, lower, upper)
    # split to years, dummies and regular vars:
    beta_years = beta[[x for x in beta.columns if 'year' in x]].T
    years = pd.to_datetime(np.arange(2001, 2020), format='%Y')
    beta_years.index = years
    beta_years.columns = ['beta']
    err_years = errors[[x for x in errors.columns if 'year' in x]]
    beta_dumm = beta[[x for x in beta.columns if x in dummies]].rename(plot_names, axis=1)
    err_dumm = errors[[x for x in errors.columns if x in dummies]]
    beta_reg = beta[[x for x in beta.columns if x not in dummies+year_dummies]].rename(plot_names, axis=1)
    err_reg = errors[[x for x in errors.columns if x not in dummies+year_dummies]]
    # plot, define axes:
    fig = plt.figure(figsize=(17,10),constrained_layout=True)
    gs1 = fig.add_gridspec(nrows=2, ncols=2, left=0.05, wspace=0.05)
    ax_years = fig.add_subplot(gs1[1, 0:])
    ax_reg = fig.add_subplot(gs1[0, 0])
    ax_dumm = fig.add_subplot(gs1[0, 1])
    beta_years.plot(kind='bar', yerr=err_years.values, ax=ax_years, legend=False, color='tab:blue')
    ax_years.set_xticklabels([x.strftime("%Y") for x in beta_years.index], rotation=45)
    ax_years.set_ylabel('Price change from year 2000 [%]')
    ax_years.grid(axis='y')
    beta_dumm.squeeze().plot(kind='barh', xerr=err_dumm.values, ax=ax_dumm,
                             color=['tab:orange', 'tab:green', 'tab:red', 'tab:purple'])
    ax_dumm.grid(True, axis='x', which='major')
    ax_dumm.set_xlabel('Price change from baseline category [%]')
    beta_reg.squeeze().plot(kind='barh', xerr=err_reg.values, ax=ax_reg,
                            color = ['tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'])
    ax_reg.grid(True, axis='x', which='major')
    ax_reg.set_xlabel('Price change due to 10% change in predictor [%]')
    # df = df.rename(plot_names, axis=1)
    # ax[0].bar(beta_years.index, beta_years.values, yerr=err_years.values, align='center', alpha=0.5, ecolor='black', capsize=10)
    # ax[0].errorbar(beta_years.index, beta_years.values,
    #                err_years.values, lw=2)

    return errors, beta


def plot_MLR_interpeted_results(ds):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    sns.set_theme(style='ticks', font_scale=1.5)
    # matplotlib.rc_file_defaults()
    # ax = sns.set_style(style=None, rc=None )
    fig, ax = plt.subplots(2, 1, figsize=(17, 10))
    ds = interpert_beta_coefs(ds, name='CI_95_upper')
    ds = interpert_beta_coefs(ds, name='CI_95_lower')
    ds = interpert_beta_coefs(ds, name='beta_coef')
    ds = ds.drop_sel(regressor='const')
    beta = ds['beta_coef_inter'].to_dataset('regressor').to_dataframe()
    beta_dumm = beta[[x for x in beta.columns if x in dummies]]
    beta_reg = beta[[x for x in beta.columns if x not in dummies]]
    beta_dumm = beta_dumm.rename(plot_names, axis=1)
    beta_reg = beta_reg.rename(plot_names, axis=1)
    beta = beta.rename(plot_names, axis=1)
    CI95_u = ds['CI_95_upper_inter'].to_dataset('regressor').to_dataframe()
    CI95_u_dumm = CI95_u[[x for x in CI95_u.columns if x in dummies]]
    CI95_u_reg = CI95_u[[x for x in CI95_u.columns if x not in dummies]]
    CI95_u_dumm = CI95_u_dumm.rename(plot_names, axis=1)
    CI95_u_reg = CI95_u_reg.rename(plot_names, axis=1)
    CI95_l = ds['CI_95_lower_inter'].to_dataset('regressor').to_dataframe()
    CI95_l_dumm = CI95_l[[x for x in CI95_l.columns if x in dummies]]
    CI95_l_reg = CI95_l[[x for x in CI95_l.columns if x not in dummies]]
    CI95_l_dumm = CI95_l_dumm.rename(plot_names, axis=1)
    CI95_l_reg = CI95_l_reg.rename(plot_names, axis=1)
    beta.index = pd.to_datetime(beta.index, format='%Y')
    # plot dummies:
    colors = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    for i, reg in enumerate(beta_dumm.columns):
        errors = np.array(list(zip(
            (beta_dumm[reg].values-CI95_l_dumm[reg].values), (CI95_u_dumm[reg].values-beta_dumm[reg].values))))
        errors = np.abs(errors)
        ax[0].errorbar(beta[reg].index, beta_dumm[reg].values,
                       errors.T, label=reg, lw=2, color=colors[i])
    ax[0].legend(ncol=2, handleheight=0.1, labelspacing=0.01,
                 loc='center left', fontsize=18)
    ax[0].set_ylabel('Price change from\nbaseline category [%]')
    ax[0].set_ylim(-40, 40)
    ax[0].grid(True)

    # plot regulars:
    colors = ['tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    for i, reg in enumerate(beta_reg.columns):
        errors = np.array(list(zip(
            (beta_reg[reg].values-CI95_l_reg[reg].values), (CI95_u_reg[reg].values-beta_reg[reg].values))))
        errors = np.abs(errors)
        ax[1].errorbar(beta[reg].index, beta_reg[reg].values,
                       errors.T, label=reg, lw=2, color=colors[i])
    ax[1].legend(ncol=2, handleheight=0.1, labelspacing=0.01,
                 loc='upper right', fontsize=18)
    ax[1].set_ylabel(
        'Price change due to\n10% change in predictor [%]')
    ax[1].grid(True)
    ax[1].set_ylim(-10, 30)
    fig.tight_layout()
    return fig


def plot_MLR_results(ds):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    sns.set_theme(style='ticks', font_scale=1.8)
    # matplotlib.rc_file_defaults()
    # ax = sns.set_style(style=None, rc=None )
    fig, ax = plt.subplots(figsize=(17, 10))
    # drop const:
    ds = ds.drop_sel(regressor='const')
    beta = ds['beta_coef'].to_dataset('regressor').to_dataframe()
    beta = beta.rename(plot_names, axis=1)
    CI95_u = ds['CI_95_upper'].to_dataset('regressor').to_dataframe()
    CI95_u = CI95_u.rename(plot_names, axis=1)
    CI95_l = ds['CI_95_lower'].to_dataset('regressor').to_dataframe()
    CI95_l = CI95_l.rename(plot_names, axis=1)
    beta.index = pd.to_datetime(beta.index, format='%Y')
    twinx = ax.twinx()
    for reg in beta.columns:
        errors = np.array(list(zip(
            (beta[reg].values-CI95_l[reg].values), (CI95_u[reg].values-beta[reg].values))))
        errors = np.abs(errors)
        ax.errorbar(beta[reg].index, beta[reg].values,
                    errors.T, label=reg, lw=2)
    # sns.lineplot(data = df, marker='o', sort = False, ax=ax)
    # df.plot(legend=False, ax=ax, zorder=0)
    ax.legend(ncol=3, handleheight=0.1, labelspacing=0.01,
              loc='upper left', fontsize=18)
    ax.set_ylabel(r'$\beta$ coefficient')
    ax.grid(True)
    dfr2 = ds['R-squared'].to_dataframe()
    dfr2.index = pd.to_datetime(dfr2.index, format='%Y')
    twinx.bar(beta.index, dfr2['R-squared'].values,
              width=350, color='k', alpha=0.2)
    ax.set_ylim(-0.65, 0.65)
    twinx.set_ylim(0, 1)
    twinx.set_ylabel(r'R$^2$')
    # ax.axhline(0, color='k', lw=0.5)
    # sns.barplot(data = dfr2, x=df.index, y='r2_score', alpha=0.3, ax=twinx)
    # dfr2.plot(ax=twinx, kind='bar', alpha=0.3, zorder=-1, legend=False)
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


def plot_RF_FI_results(fi):
    """run loop over .. with mode='FI' first"""
    import pandas as pd
    import seaborn as sns
    import numpy as np
    from matplotlib.ticker import MultipleLocator
    import matplotlib.pyplot as plt
    sns.set_theme(style='ticks', font_scale=1.8)
    cmap = sns.color_palette('Dark2', as_cmap=False, n_colors=len(fi.columns))
    fig, ax = plt.subplots(figsize=(17, 10))
    # df = ds['feature_importances'].mean(
        # 'repeats').to_dataset('regressor').to_dataframe()
    # order the features:
    df = fi[['SEI', 'Total_ends', 'mean_distance_to_28_mokdim', 'Rooms', 'New', 'year']]
    # df = df.reset_index(drop=True)
    # df['year'] = df['year'].astype(str)
    df.index = pd.to_datetime(df['year'], format='%Y')
    df = df.rename(vars_explained_plot_names, axis=1)
    df = df.drop('year', axis=1)
    # order the columns:
    # df = df[['Socio-Economic Index', 'Building rate', 'Distance to ECs', 'Rooms', 'Used/New']]
    df *= 100
    x = df.index
    ys = [df[x] for x in df.columns]
    ax.stackplot(x, *ys, labels=[x for x in df.columns], colors=cmap, alpha=0.6)
    # df.plot(ax=ax, legend=False)
    ax.legend(loc='center', ncol=2, handleheight=0.1, labelspacing=0.01)
    # df_total = df.sum(axis=1)
    # df_rel = df[df.columns[1:]].div(df_total, 0)*100
    # for n in df_rel:
    #     for i, (cs, ab, pc) in enumerate(zip(df.iloc[:, 1:].cumsum(1)[n],
    #                                          df[n], df_rel[n])):
    #         ax.text(cs - ab / 2, i, str(np.round(pc, 1)) + '%',
    #                  va = 'center', ha = 'center')
    ax.set_ylabel('MDI predictor importances [%]')
    ax.grid(True)
    ml = MultipleLocator(5)
    ax.yaxis.set_minor_locator(ml)
    # ax.tick_params(axis='y', which='minor')
    fig.tight_layout()
    return fig


def run_CV_on_all_years(df, model_name='RF', savepath=ml_path, pgrid='normal',
                        year=None, year_start=None, feats=best_rf2+['SEI']):
    import numpy as np
    if year is None:
        years = np.arange(2000, 2020, 1)
        if year_start is not None:
            years = np.arange(year_start, 2020, 1)
        for year in years:
            if model_name == 'RF':
                X_train, X_test, y_train, y_test = produce_X_y_RF_per_year(df,
                                                                           year=year, feats=feats)
            else:
                X, y, scaler = produce_X_y(df, year=year, y_name='Price', plot_Xcorr=False,
                                           feats=features, dummy=None, scale_X=False)
            # ml = ML_Classifier_Switcher()
            # model = ml.pick_model(model_name)
            cross_validation(X_train, y_train, model_name=model_name, n_splits=5, pgrid=pgrid,
                             savepath=savepath, verbose=0, n_jobs=-1, year=year)
    else:
        X, y, scaler = produce_X_y(df, year=year, y_name='Price', plot_Xcorr=False,
                                   feats=features, dummy=None, scale_X=False)
        # ml = ML_Classifier_Switcher()
        # model = ml.pick_model(model_name)
        cross_validation(X, y, model_name=model_name, n_splits=5, pgrid=pgrid,
                         savepath=savepath, verbose=0, n_jobs=-1, year=year)

    return


def produce_employment_centers_table():
    from cbs_procedures import read_emploment_centers_2008
    df = read_emploment_centers_2008()
    df['#'] = np.arange(1, 29)
    df = df[['#', 'NameEN', 'Pop2020', 'latitude', 'longitude']]
    df.columns = ['Name', 'Population in 2020', 'Latitude', 'Longitude']
    print(df.to_latex(index=False))
    return df


def produce_rooms_new_price_table_for_all_years(df):
    import numpy as np
    import pandas as pd
    pd.options.display.float_format = '{:,.2f}'.format
    years = np.arange(2000, 2020, 1)
    tabs = []
    for year in years:
        X, y=produce_X_y_RF_per_year(df, year=year, split=False)
        X = X.join(y)
        X['Price_nis'] = X['Price'].apply(np.exp)
        tab = X['Rooms'].value_counts().to_frame().T
        tab.index = [year]
        tab.columns = ['3-rooms', '4-rooms', '5-rooms']
        tab['New'] = X['New'].sum()
        tab['Used'] = len(X) - tab['New']
        tab['Total'] = len(X)
        tab['Mean Price [10$^6$ NIS]'] = X['Price_nis'].mean().round(0)
        tabs.append(tab)
    dff = pd.concat(tabs, axis=0)
    dff = dff.append(dff.sum(), ignore_index=True)
    dff.index = [x for x in years] + ['Total']
    for col in [x for x in dff.columns]:
        dff[col] = dff.apply(lambda x: "{:,.0f}".format(x[col]), axis=1)
    dff.iat[-1, -1] = '-'
    return dff


def produce_X_y_RF_per_year(df, y_name='Price', feats=best_rf2+['SEI'],
                            year=2000, split=True,
                            test_size=0.1, verbose=1):
    from sklearn.model_selection import train_test_split
    import numpy as np
    if verbose > 0:
        print('picking year {}.'.format(year))
    df = df[df['Sale_year'] == year]
    if feats is not None:
        X = df[feats].dropna()
        if (year >= 2013) and (year <= 2015):
            X['SEI'] = X['SEI_value_2015']
            X = X.drop(['SEI_value_2015', 'SEI_value_2017'], axis=1)
        elif (year >= 2016) and (year <= 2019):
            X['SEI'] = X['SEI_value_2017']
            X = X.drop(['SEI_value_2015', 'SEI_value_2017'], axis=1)
        else:
            X['SEI'] -= 3
            X = X.drop(['SEI_value_2015', 'SEI_value_2017'], axis=1)
    if 'Rooms' in X.columns:
        X = X[(X['Rooms'] >= 1) & (X['Rooms'] <= 6)]
    if 'Rooms_345' in X.columns:
        X = X.rename({'Rooms_345': 'Rooms'}, axis=1)
    y = df.loc[X.index, [y_name, 'Sale_year']]
    X = X.reset_index(drop=True)
    if 'New' in X.columns:
        X['New'] = X['New'].astype(int)
    if verbose > 0:
        print('picking {} as features.'.format([x for x in X.columns]))
    y = y.reset_index(drop=True)
    y = y.apply(np.log)
    X = X.drop('Sale_year', axis=1)
    y = y.drop('Sale_year', axis=1)
    if not split:
        return X, y
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=42)
    y_train = y_train[y_name]
    y_test = y_test[y_name]
    if verbose > 0:
        print('picking test size of {} which is {} records.'.format(test_size, len(y_test)))
    return X_train, X_test, y_train, y_test


# def produce_X_y_RF(df, y_name='Price', feats=best_rf, train_years=[2015, 2016],
#                    test_years=[2017, 2018]):
#     from sklearn.preprocessing import StandardScaler
#     # from sklearn.preprocessing import MinMaxScaler
#     # import numpy as np
#     import pandas as pd
#     print('picking {} as train years.'.format(','.join([str(x) for x in train_years])))
#     print('picking {} as test years.'.format(','.join([str(x) for x in test_years])))
#     df = df[df['Sale_year'].isin(train_years+test_years)]
#     if feats is not None:
#         print('picking {} as features.'.format(feats))
#         X = df[feats].dropna()
#     if 'Rooms' in X.columns:
#         X = X[(X['Rooms']>=1) & (X['Rooms']<=6)]
#     y = df.loc[X.index, [y_name, 'Sale_year']]
#     X = X.reset_index(drop=True)
#     if 'New' in X.columns:
#         X['New'] = X['New'].astype(int)
#     if 'MISH' in X.columns:
#         X['MISH'] = X['MISH'].astype(int)
#     # if 'Total_ends' in X.columns:
#     #     X['Total_ends'] /= 1000
#     # if 'Netflow' in X.columns:
#     #     X['Netflow'] /= 10000
#     y = y.reset_index(drop=True)
#     # if dummy is not None:
#     #     prefix = dummy.split('_')[0]
#     #     rooms = pd.get_dummies(data=X[dummy], prefix=prefix)
#     #     # drop one col from one-hot encoding not to fall into dummy trap!:
#     #     X = pd.concat([X, rooms.drop('Rooms_4', axis=1)], axis=1)
#     #     X = X.drop([dummy], axis=1)

#     # X = scale_log(X, cols=[x for x in X.columns if x not in dummies], plus1_cols=[
#     #               'Total_ends', 'Floor_number'])

#     yscaler = StandardScaler()
#     #yscaler = MinMaxScaler()
#     # yscaler = PowerTransformer(method='yeo-johnson',standardize=True)
#     y_train = y[y['Sale_year'].isin(train_years)].apply(np.log)
#     y_test = y[y['Sale_year'].isin(test_years)].apply(np.log)
#     # y_train = yscaler.fit_transform(y_train[y_name].values.reshape(-1,1))
#     # y_test = yscaler.fit_transform(y_test[y_name].values.reshape(-1,1))
#     # y_train = pd.DataFrame(y_train, columns=[y_name])
#     # y_test = pd.DataFrame(y_test, columns=[y_name])
#     # y_test = y_test.apply(np.log)
#     # # y_scaled = yscaler.fit_transform(y_scaled.values.reshape(-1,1))
#     # y_train = pd.DataFrame(y_train, columns=[y_name])
#     # y_test = pd.DataFrame(y_test, columns=[y_name])
#     # scale X:
#     # cols_to_scale = [x for x in X.columns if x != 'Sale_year']
#     # X, scaler = scale_df(X, cols=cols_to_scale, scaler=Xscaler)
#     X_train = X[X['Sale_year'].isin(train_years)].drop('Sale_year', axis=1).astype(float)
#     X_train.drop('SEI_value_2017', axis=1, inplace=True)
#     X_test = X[X['Sale_year'].isin(test_years)].drop('Sale_year', axis=1).astype(float)
#     X_test.drop('SEI_value_2015', axis=1, inplace=True)
#     X_train = X_train.rename({'SEI_value_2015': 'SEI'}, axis=1)
#     X_test = X_test.rename({'SEI_value_2017': 'SEI'}, axis=1)
#     y_train = y_train[y_name]
#     y_test = y_test[y_name]
#     return X_train, y_train, X_test, y_test # , Xscaler


def produce_X_y(df, year=2015, y_name='Price', plot_Xcorr=True,
                feats=features, dummy='Rooms_345', scale_X=True):
    import seaborn as sns
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.preprocessing import PowerTransformer
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    import pandas as pd
    print('Choosing {} as target var (y)'.format(y_name))
    # first, subset for year:
    # df = df[df['Type_of_asset'].isin(apts_more)]
    if year is not None:
        if year != 'trend':
            df = df[df['Sale_year'] == year]
        else:
            feats.append('trend')
    else:
        years = pd.get_dummies(data=df['Sale_year'], prefix='year')
        # drop one col from one-hot encoding not to fall into dummy trap!:
        df = pd.concat([df, years.drop('year_2000', axis=1)], axis=1)
    # now slice for the features:
    if feats is not None:
        print('picking {} as features.'.format(feats))
        X = df[feats].dropna()
        X.drop('Sale_year', axis=1, inplace=True)
    y = df.loc[X.index, y_name]
    X = X.reset_index(drop=True)
    if 'New' in X.columns:
        X['New'] = X['New'].astype(int)
    if 'Has_ground_floor' in X.columns:
        X['Has_ground_floor'] = X['Has_ground_floor'].astype(int)
    # scale Netflow to 10,000 poeple:
    if 'Netflow' in X.columns:
        X['Netflow'] /= 10000
    # scale mean distance to 100 kms:
    if 'mean_distance_to_28_mokdim' in X.columns:
        X['mean_distance_to_28_mokdim'] /= 100
    y = y.reset_index(drop=True)
    # df['Rooms'] = df['Rooms'].astype(int)
    # df['Floor_number'] = df['Floor_number'].astype(int)
    # df['New_apartment'] = df['New_apartment'].astype(int)
    if plot_Xcorr:
        if year is None:
            sns.heatmap(X.drop(year_dummies, axis=1).corr('pearson'), annot=True)
        else:
            sns.heatmap(X.corr('pearson'), annot=True)
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
        X = pd.concat([X, rooms.drop('Rooms_3', axis=1)], axis=1)
        X = X.drop([dummy], axis=1)
    # scale to log all non-dummy variables:
    # dummies = ['New', 'Rooms_3', 'Rooms_5']
    if year is None:
        if year != 'trend':
            dumm = dummies + ['year_{}'.format(x) for x in np.arange(2001, 2020)]
        else:
            dumm = dummies
    else:
        dumm = dummies
    # log scale building rates:
    # X = scale_log(X, cols=['Total_ends'], plus1_cols=[
    #               'Total_ends', 'Floor_number'])
    # if city_code col exist, drop it:
    if 'city_code' in X.columns:
        X = X.drop('city_code', axis=1)
    # finally, scale y to log10 and X to minmax 0-1:
    # Xscaler = MinMaxScaler()
    Xscaler = StandardScaler()
    #yscaler = MinMaxScaler()
    # yscaler = PowerTransformer(method='yeo-johnson',standardize=True)
    y_scaled = y.apply(np.log)
    # y_scaled = yscaler.fit_transform(y_scaled.values.reshape(-1,1))
    y = pd.DataFrame(y_scaled, columns=[y_name])
    if scale_X:
        X, scaler = scale_df(X, scaler=Xscaler, cols=[
                             x for x in X.columns if x in best_regular])
        # y, scaler = scale_df(y, scaler=Xscaler)
    else:
        scaler = Xscaler
    y = y[y_name]
    return X, y, scaler  # , yscaler


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
    if verbose >=0:
        print('validating {} model with {} CV splits.'.format(model_name, n_splits))
        print(param_grid)
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
        fi = fi.drop_sel(regressor='Sale_year')
        df = convert_da_to_long_form_df(fi, value_name='feature_importances')
        df = df.sort_values('feature_importances')
        df['regressor'] = df['regressor'].map(plot_names)
        # df = df.rename(plot_names, axis=1)
        if mode != 'beta':
            df['feature_importances'] = df['feature_importances'] * 100
        sns.barplot(data=df, x='feature_importances',
                    y='regressor', ci='sd', ax=ax)
    else:
        if mode != 'beta':
            fi['value'] = fi['value'] * 100
        fi = fi.rename(plot_names, axis=1)
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

    def XGR(self):
        import numpy as np
        from xgboost import XGBRegressor
        if self.pgrid == 'light':
            self.param_grid = {'gamma': np.logspace(-5, 1, 7),
                               'subsample': [0.5, 0.75, 1.0],
                               'colsample_bytree': [0.5, 0.75, 1.0],
                               'learning_rate': ['constant', 'adaptive'],
                               'max_depth': [3, 5, 10]}
        return XGBRegressor(random_state=42, n_jobs=-1)

    def RF(self):
        from sklearn.ensemble import RandomForestRegressor
        # import numpy as np
        if self.pgrid == 'light':
            self.param_grid = {'max_features': ['auto', 'sqrt'],
                               'max_depth': [5, 10],
                               'min_samples_leaf': [1, 2],
                               'min_samples_split': [2, 5],
                               'n_estimators': [500]}
        elif self.pgrid == 'normal':
            self.param_grid = {'max_depth': [2, 5, 7],
                               'max_features': ['auto'],
                               'min_samples_leaf': [2, 5],
                               'min_samples_split': [2, 5],
                               'n_estimators': [100, 200, 400]
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

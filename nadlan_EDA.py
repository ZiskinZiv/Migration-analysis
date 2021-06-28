#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 08:32:19 2021

@author: shlomi
"""
from MA_paths import work_david
nadlan_path = work_david / 'Nadlan_deals'
apts = ['דירה', 'דירה בבית קומות']


def create_higher_group_category(df, existing_col='SEI_cluster', n_groups=2,
                                 new_col='SEI2_cluster', names=None):
    import pandas as pd
    lower_group = sorted(df[existing_col].dropna().unique())
    new_group = [lower_group[i:i+n_groups+1] for i in range(0, len(lower_group), n_groups+1)]
    new_dict = {}
    if names is not None:
        assert len(names) == len(new_group)
    for i, item in enumerate(new_group):
        if names is not None:
            new_dict[names[i]] = new_group[i]
        else:
            new_dict[i+1] = new_group[i]
    m = pd.Series(new_dict).explode().sort_values()
    d = {x: y for (x, y) in zip(m.values, m.index)}
    df[new_col] = df[existing_col].map(d)
    return df


def load_nadlan_deals(path=work_david, csv=True,
                      times=['1998Q1', '2021Q1'], filter_dealamount=True,
                      fix_new_status=True, add_SEI2_cluster=True):
    import pandas as pd
    import numpy as np
    from Migration_main import path_glob
    if csv:
        file = path_glob(path, 'Nadlan_deals_processed_*.csv')
        dtypes = {'FULLADRESS': 'object', 'Street': 'object', 'FLOORNO': 'object',
                  'NEWPROJECTTEXT': 'object', 'PROJECTNAME': 'object', 'DEALAMOUNT': float}
        df = pd.read_csv(file[0], na_values='None', parse_dates=['DEALDATETIME'],
                         dtype=dtypes)
    else:
        file = path_glob(path, 'Nadlan_deals_processed_*.hdf')
        df = pd.read_hdf(file)
    df['year'] = df['DEALDATETIME'].dt.year
    df['month'] = df['DEALDATETIME'].dt.month
    df['quarter'] = df['DEALDATETIME'].dt.quarter
    df['YQ'] = df['year'].astype(str) + 'Q' + df['quarter'].astype(str)
    if times is not None:
        print('Slicing to times {} to {}.'.format(*times))
        # df = df[df['year'].isin(np.arange(years[0], years[1] + 1))]
        df = df.set_index('DEALDATETIME')
        df = df.loc[times[0]:times[1]]
        df = df.reset_index()
    if filter_dealamount:
        print('Filtering DEALAMOUNT with IQR of  {}.'.format(1.5))
        df = df[~df.groupby('year')['DEALAMOUNT'].apply(
            is_outlier, method='iqr', k=2)]
    if fix_new_status:
        inds = df.loc[(df['Age'] < 0) & (df['Age'] > -5)].index
        df.loc[inds, 'New'] = True
    if add_SEI2_cluster:
        SEI_cluster = [x+1 for x in range(10)]
        new = [SEI_cluster[i:i+2] for i in range(0, len(SEI_cluster), 2)]
        SEI2 = {}
        for i, item in enumerate(new):
            SEI2[i+1] = new[i]
        m = pd.Series(SEI2).explode().sort_values()
        d = {x: y for (x, y) in zip(m.values, m.index)}
        df['SEI2_cluster'] = df['SEI_cluster'].map(d)
    return df


def is_outlier(s, k=3, method='std'):
    # add IQR
    if method == 'std':
        lower_limit = s.mean() - (s.std() * k)
        upper_limit = s.mean() + (s.std() * k)
    elif method == 'iqr':
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3-q1  # Interquartile range
        lower_limit = q1 - k * iqr
        upper_limit = q3 + k * iqr
    return ~s.between(lower_limit, upper_limit)

# df1 = df[~df.groupby('year')['DEALAMOUNT'].apply(is_outlier)]


def keep_only_historic_changed_assets(df):
    df = df.reset_index(drop=True)
    grps = df.groupby('GUSH').groups
    inds = []
    for gush, ind in grps.items():
        if len(ind) > 1:
            inds.append([x for x in ind])
    flat_list = [item for sublist in inds for item in sublist]
    df_slice = df.loc[flat_list]
    return df_slice


def create_neighborhood_polygons(gdf):
    import numpy as np
    gdf = gdf.reset_index()
    neis = gdf['Neighborhood'].unique()
    gdf['neighborhood_shape'] = gdf.geometry
    # Must be a geodataframe:
    for nei in neis:
        gdf1 = gdf[gdf['Neighborhood']==nei]
        inds = gdf1.index
        polygon = gdf1.geometry.unary_union.convex_hull
        # gdf.loc[inds, 'neighborhood_shape'] = [polygon for x in range(len(inds))]
        gdf.loc[inds, 'neighborhood_shape'] = polygon
    return gdf


def convert_da_to_long_form_df(da, var_name=None, value_name=None):
    """ convert xarray dataarray to long form pandas df
    to use with seaborn"""
    import xarray as xr
    if var_name is None:
        var_name = 'var'
    if value_name is None:
        value_name = 'value'
    dims = [x for x in da.dims]
    if isinstance(da, xr.Dataset):
        value_vars = [x for x in da]
    elif isinstance(da, xr.DataArray):
        value_vars = [da.name]
    df = da.to_dataframe()
    for i, dim in enumerate(da.dims):
        df[dim] = df.index.get_level_values(i)
    df = df.melt(value_vars=value_vars, value_name=value_name,
                 id_vars=dims, var_name=var_name)
    return df


def calculate_recurrent_times_and_pct_change(df, plot=True):
    import seaborn as sns
    sns.set_theme(style='ticks', font_scale=1.5)
    df = df.sort_values('DEALDATETIME')
    dff = df.groupby('GUSH')['DEALAMOUNT'].count()
    dff2 = df[df['GUSH'].isin(dff[dff > 1].index)]
    seconds_between_deals = dff2.groupby('GUSH')['DEALDATETIME'].diff().dt.total_seconds()
    deals_pct_change = dff2.groupby('GUSH')['DEALAMOUNT'].pct_change()
    df['years_between_deals'] = seconds_between_deals / 60 / 60 / 24 / 365.25
    df['mean_years_between_deals'] = df.groupby('GUSH')['years_between_deals'].transform('mean')
    df['deals_pct_change'] = deals_pct_change * 100
    df['mean_deals_pct_change'] = df.groupby('GUSH')['deals_pct_change'].transform('mean')
    # drop duplicated dt's:
    deals_inds_to_drop = deals_pct_change[deals_pct_change == 0].index
    seconds_inds_to_drop = seconds_between_deals[seconds_between_deals == 0].index
    inds_to_drop = deals_inds_to_drop.union(seconds_inds_to_drop)
    df = df.drop(inds_to_drop, axis=0)
    print('Dropped {} deals'.format(len(inds_to_drop)))
    if plot:
        g = sns.JointGrid(data=df, x='years_between_deals',
                          y='deals_pct_change', height=7.5)
        g.plot_joint(sns.kdeplot, fill=True, cut=1, gridsize=100)
        g.plot_marginals(sns.histplot)
        g.ax_joint.grid(True)
        g.ax_joint.set_xlim(-1, 21)
        g.ax_joint.set_ylim(-100, 260)
        g.ax_joint.set_ylabel('Change in recurrent deals [%]')
        g.ax_joint.set_xlabel('Years between recurrent deals')
        g.fig.tight_layout()
        return g
    else:
        return df

def plot_recurrent_deals(df, max_number_of_sells=6, rooms=[2, 3, 4, 5]):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    sns.set_theme(style='ticks', font_scale=1.5)
    if rooms is not None:
        df = df[df['ASSETROOMNUM'].isin(rooms)]
        df = df.rename({'ASSETROOMNUM': 'Number of rooms'}, axis=1)
        df['Number of rooms'] = df['Number of rooms'].astype(int)
        df = df[df['DEALNATUREDESCRIPTION'].isin(apts)]
        dff = df.groupby(['GUSH','Number of rooms'])['DEALAMOUNT'].count()
        dff = dff[dff <= 6]
        df1 = dff.groupby('Number of rooms').value_counts()
        f, ax = plt.subplots(figsize=(7, 7))
        sns.lineplot(x='DEALAMOUNT', y=df1, data=df1, hue='Number of rooms',
                     style='Number of rooms', palette='Set1', ax=ax, markers=True,
                     markersize=15)
        ax.set(xscale="linear", yscale="log")
        ax.grid(True)
        ax.set_xlabel('Number of times an apartment is sold')
        ax.set_ylabel('Total deals')
    else:
        df1 = df[df['DEALNATUREDESCRIPTION'].isin(apts)].groupby('GUSH')[
            'DEALAMOUNT'].count()
        v = np.arange(1, max_number_of_sells + 1)
        n = [len(df1[df1 == x]) for x in v]
        dfn = pd.DataFrame(n, index=v)
        dfn.columns = ['Number of Deals']
        f, ax = plt.subplots(figsize=(7, 7))
        ax = sns.scatterplot(x=dfn.index, y='Number of Deals',
                             data=dfn, ax=ax, s=50)
        p = np.polyfit(v, np.log(n), 1)
        fit = np.exp(np.polyval(p, v))
        print(fit)
        dfn['Fit'] = fit
        ax = sns.lineplot(x=dfn.index, y='Fit', data=dfn, ax=ax, color='r')
        ax.set(xscale="linear", yscale="log")
        ax.grid(True)
        ax.set_xlabel('Number of times an apartment is sold')
    return f


def plot_deal_amount_room_number(df, rooms=[2, 3, 4, 5],
                                 path=nadlan_path, yrmin='2000', yrmax='2020',
                                 just_with_historic_change=False):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from cbs_procedures import read_bycode_city_data
    import numpy as np
    sns.set_theme(style='ticks', font_scale=1.5)
    # df = df.loc[(df['ASSETROOMNUM'] >= room_min) &
    #             (df['ASSETROOMNUM'] <= room_max)]
    df = df[df['ASSETROOMNUM'].isin(rooms)]
    df.set_index('DEALDATETIME', inplace=True)
    df = df.loc[yrmin:yrmax]
    city_code = df.loc[:, 'city_code'].unique()[0]
    df = df.rename({'ASSETROOMNUM': 'Rooms', 'DEALAMOUNT': 'Price'}, axis=1)
    df['Price'] /= 1000000
    fig, ax = plt.subplots(figsize=(10, 8))
    ax = sns.lineplot(data=df, x=df.index.year, y='Price', hue='Rooms',
                      ci='sd', ax=ax, palette='Set1')
    ax.grid(True)
    ax.set_xlabel('')
    ax.set_xticks(np.arange(int(yrmin), int(yrmax) + 1, 2))
    ax.tick_params(axis='x', rotation=30)
    ax.set_ylabel('Price [millions of NIS]')
    bycode = read_bycode_city_data()
    city_name = bycode[bycode['city_code']==city_code]['NameEn'].values[0]
    fig.suptitle('Real-Estate prices in {}'.format(city_name))
    return ax


def plot_groupby_m2_price_time_series(df, grps=['City', 'Neighborhood'],
                                      col='NIS_per_M2'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style='ticks', font_scale=1.5)
    groups = grps.copy()
    groups.append('year')
    dfn = df.groupby(groups, as_index=False)[col].mean().groupby('year').mean()
    std = df.groupby(groups, as_index=False)[col].mean().groupby('year').std()[col].values
    dfn['plus_std'] = dfn[col] + std
    dfn['minus_std'] = dfn[col] - std
    fig, ax = plt.subplots(figsize=(14, 5))
    dfn[col].plot(ax=ax)
    ax.fill_between(dfn.index, dfn['minus_std'], dfn['plus_std'], alpha=0.3)
    ax.set_ylabel(r'NIS per M$^2$')
    fig.tight_layout()
    ax.grid(True)
    return fig


def plot_room_number_deals(df, rooms_range=[2, 6]):
    import seaborn as sns
    sns.set_theme(style='ticks', font_scale=1.5)
    if rooms_range is not None:
        df = df.loc[(df['ASSETROOMNUM'] >= rooms_range[0]) &
                    (df['ASSETROOMNUM'] <= rooms_range[1])]
    dff = df.groupby(['ASSETROOMNUM', 'year'])['DEALAMOUNT'].count()
    da = dff.to_xarray()
    dff = convert_da_to_long_form_df(da, value_name='Deals')
    ax = sns.barplot(data=dff,x='year', y='Deals', hue='ASSETROOMNUM', palette='Set1')
    ax.grid(True)
    return dff


def compare_kiryat_gat_israel_dealamount(df_kg, df_isr):
    # TODO: complete this
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style='ticks', font_scale=1.5)
    df_kg = df_kg.groupby(['rooms', 'YQ'])[
        'DEALAMOUNT'].mean().to_frame().unstack().T.droplevel(0)
    df_isr = df_isr.groupby(['rooms', 'YQ'])[
        'DEALAMOUNT'].mean().to_frame().unstack().T.droplevel(0)
    # df_kg['YQ'] = df_kg.index
    # df_isr['YQ'] = df_isr.index
    # df_kg = df_kg.melt(id_vars='YQ', value_name='price_in_kg')
    # df_isr = df_isr.melt(id_vars='YQ', value_name='price_in_israel')
    # df = pd.concat([df_kg, df_isr], axis=1)
    df = df_kg / df_isr
    # df['price_diff'] = df['price_in_kg'] - df['price_in_israel']
    fig, ax = plt.subplots(figsize=(15.5, 6))
    df1 = df #/ 1e6
    df1.index = pd.to_datetime(df1.index)
    df1.plot(ax=ax, cmap=sns.color_palette("tab10", as_cmap=True))
    ax.set_ylabel('Price difference [million NIS]')
    df2 = df1.rolling(4, center=True).mean()
    df2.columns = ['{} rolling mean'.format(x) for x in df2.columns]
    df2.plot(ax=ax, cmap=sns.color_palette("tab10", as_cmap=True), ls='--')
    ax.axvline(pd.to_datetime('2008-07-01'), color='g')
    ax.axvline(pd.to_datetime('2006-01-01'), color='r')
    ax.grid(True)
    ax.set_xlabel('')
    fig.suptitle('Kiryat Gat - Israel mean apartment prices')
    fig.tight_layout()
    return df


def plot_bootstrapped_dff_rooms(dff, time='year'):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    sns.set_theme(style='ticks', font_scale=1.5)
    fig, ax = plt.subplots(figsize=(15, 5))
    dff['price'] = dff['DEALAMOUNT'] / 1e6
    dff[time] = pd.to_datetime(dff[time])
    sns.lineplot(data=dff, x=time, y='price', ci='sd', hue='rooms',
                 ax=ax, style='rooms')
    ax.grid(True)
    ax.set_xlabel('')

    # ax.set_xticks(np.arange(1998, 2021, 2))
    ax.set_ylabel('Apartment price [millions NIS]')
    fig.suptitle('Mean apartment prices in Israel')
    fig.tight_layout()
    return fig


def prepare_bootstrapped_dfs_by_year_and_apts(df, nr=1000, frac=0.05,
                                              city_code=None, grp='year'):
    # for all Israel i did nr=5000, frac=0.35
    import pandas as pd
    df = df[df['DEALNATUREDESCRIPTION'].isin(apts)]
    if city_code is not None:
        df = df[df['city_code'] == city_code]
        print('{} city selected ({})'.format(
            df['City'].unique()[0], city_code))
    groups = [[2, 3], [3, 4], [4, 5]]
    grp_dfs = []
    for apt_group in groups:
        print('{} rooms selected.'.format(
            ','.join([str(x) for x in apt_group])))
        df1 = df[df['ASSETROOMNUM'].isin(apt_group)]
        dff = bootstrap_df_by_year(
            df1, col='DEALAMOUNT', n_replicas=nr, frac_deals=frac,
            grp=grp)
        dff['rooms'] = '-'.join([str(x) for x in apt_group])
        grp_dfs.append(dff)
    return pd.concat(grp_dfs, axis=0)


def bootstrap_df_by_year(df, grp='year', frac_deals=0.1, col='NIS_per_M2',
                         n_replicas=1000, plot=False):
    import pandas as pd
    import seaborn as sns
    count = df.groupby(grp)[col].count()
    dffs = []
    for year in count.index:
        samples = count.loc[year] * frac_deals
        print('bootstrapping year {} with {} samples.'.format(year, samples))
        df1 = df[df[grp] == year][col].dropna()
        # print(len(df1))
        # dff = bootstrap_df(df1, n_rep=n_replicas,
                           # frac=frac_deals, n_sam=None, col=col, grp=None)
        dff = pd.Series([df1.sample(n=None, frac=frac_deals, replace=True, random_state=None).mean() for i in range(n_replicas)])
        dff = dff.to_frame(year).reset_index(drop=True)
        dffs.append(dff)
    stats = pd.concat(dffs, axis=1).melt(var_name=grp, value_name=col)
    if plot:
        sns.lineplot(data=stats, x=grp, y=col, ci='sd')
    return stats


# def bootstrap_df(df, n_rep=1000, n_sam=1000, frac=None, grp='year', stat='mean',
#                  col='NIS_per_M2'):
#     import pandas as pd
#     print('bootstapping the {} from {} replicas of {} samples from {} in df.'.format(
#         stat, n_rep, n_sam, col))
#     if grp is not None:
#         print('{} groupby chosen.'.format(grp))
#         stats = pd.DataFrame([df.groupby(grp).sample(
#             n=n_sam, frac=frac, replace=True, random_state=None).groupby(grp)[col].agg(stat) for i in range(n_rep)])
#         stats = stats.melt(value_name=col)
#     else:
#         stats = pd.Series([df[col].sample(n=n_sam, frac=frac, replace=True, random_state=None).agg(stat) for i in range(n_rep)])

#     return stats


def run_lag_analysis_boi_interest_nadlan(ndf, idf, i_col='effective', months=48):
    ndf = ndf.set_index('DEALDATETIME')
    ndf_mean = ndf['DEALAMOUNT'].resample('M').mean()
    idf = idf[i_col].to_frame()
    idf = idf.rolling(6, center=True).mean()
    idf['apt_prices'] = ndf_mean.rolling(6, center=True).mean()
    for i in range(months):
        idf['{}_{}'.format(i_col, i+1)] = idf[i_col].shift(-i-1)
    return idf


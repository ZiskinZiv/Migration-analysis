#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 08:32:19 2021

@author: shlomi
"""
from MA_paths import work_david
nadlan_path = work_david / 'Nadlan_deals'
apts = ['דירה', 'דירה בבית קומות']


def load_nadlan_deals(path=work_david, csv=True,
                      years=[1998, 2020], filter_dealamount=True):
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
    print('Slicing to years {} to {}.'.format(*years))
    df = df[df['year'].isin(np.arange(years[0], years[1] + 1))]
    if filter_dealamount:
        print('Filtering DEALAMOUNT with IQR of  {}.'.format(1.5))
        df = df[~df.groupby('year')['DEALAMOUNT'].apply(
            is_outlier, method='iqr', k=1.5)]
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


def calculate_recurrent_times_and_pct_change(df):
    df = df.sort_values('DEALDATETIME')
    dff = df.groupby('GUSH')['DEALAMOUNT'].count()
    dff2 = df[df['GUSH'].isin(dff[dff > 1].index)]
    seconds_between_deals = dff2.groupby('GUSH')['DEALDATETIME'].diff().dt.total_seconds()
    deals_pct_change = dff2.groupby('GUSH')['DEALAMOUNT'].pct_change()
    df['years_between_deals'] = seconds_between_deals / 60 / 60 / 24 / 365.25
    df['mean_years_between_deals'] = df.groupby('GUSH')['years_between_deals'].transform('mean')
    df['deals_pct_change'] = deals_pct_change
    df['mean_deals_pct_change'] = df.groupby('GUSH')['deals_pct_change'].transform('mean')
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

def plot_deal_amount_room_number(df, room_min=2, room_max=5,
                                 path=nadlan_path, yrmin='2000', yrmax='2020',
                                 just_with_historic_change=False):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from cbs_procedures import read_bycode_city_data
    import numpy as np
    sns.set_theme(style='ticks', font_scale=1.5)
    df = df.loc[(df['ASSETROOMNUM'] >= room_min) &
                (df['ASSETROOMNUM'] <= room_max)]
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

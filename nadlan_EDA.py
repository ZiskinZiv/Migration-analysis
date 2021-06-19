#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 08:32:19 2021

@author: shlomi
"""
from MA_paths import work_david
nadlan_path = work_david / 'Nadlan_deals'
apts = ['דירה', 'דירה בבית קומות']


def load_nadlan_deals(path=work_david, csv=True):
    import pandas as pd
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
    # df['DEALAMOUNT'] = df['DEALAMOUNT'].astype(float)
    return df


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


def plot_recurrent_deals(df, max_number_of_sells=6):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    sns.set_theme(style='ticks', font_scale=1.5)
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

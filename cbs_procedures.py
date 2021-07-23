#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 11:44:45 2021

@author: ziskin
"""
# TODO: each call to body with street name yields district, city and street
# name and mid-street coords along with neiborhood based upon mid-street coords.
# however, if we request street and building number we get more precise location
# and neighborhood. We should re-run the body request for each line in the DataFrame
# and get the precise coords and nieghborhood
from MA_paths import work_david


def geo_location_settelments_israel(path=work_david):
    import pandas as pd
    import geopandas as gpd
    df = pd.read_csv(path/'Israel_setlmidpoint.csv',
                     encoding="cp1255", na_values='<Null>')
    df.columns = ['city_code', 'NameHe', 'NameEn', 'city_type', 'X', 'Y', 'data_year', 'data_version']
    df['city_type'] = df['city_type'].str.replace('יישוב עירוני', 'urban')
    df['city_type'] = df['city_type'].str.replace('יישוב כפרי', 'rural')
    df['city_type'] = df['city_type'].str.replace('מקום', 'place')
    df['city_type'] = df['city_type'].str.replace('מוקד תעסוקה', 'employment center')
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['X'], df['Y']))
    return df


def read_mean_salary(path=work_david, resample='AS'):
    import pandas as pd
    df = pd.read_excel(path/'mean_salary_1990-2019_cbs_table.xls', skiprows=8)
    df = df.drop(df.tail(5).index)
    df.columns = ['year', 'to_drop', '12', '11', '10', '9', '8', '7',
                  '6', '5', '4', '3', '2', '1', 'year_mean', 'to_drop1', 'to_drop2']
    df = df[[x for x in df.columns if 'to_drop' not in x]]
    df = df.loc[:, 'year':'1'].melt(
        var_name='month', value_name='mean_salary1', id_vars='year')
    df['dt'] = df['year'].astype(str) + '-' + df['month'].astype(str)
    df['dt'] = pd.to_datetime(df['dt'], format='%Y-%m')
    df = df.set_index('dt')
    df = df.drop(['year', 'month'], axis=1)
    df = df.sort_index()
    df1 = pd.read_excel(path/'mean_salary_2012-2021_cbs.xlsx', skiprows=21)
    df1 = df1.drop(df1.tail(3).index)
    df1.columns = ['time', 'mean_salary2']
    df1['dt'] = pd.to_datetime(df1['time'], format='%Y-%m')
    df1 = df1.set_index('dt')
    df1 = df1.drop('time', axis=1)
    df = pd.concat([df,df1],axis=1)
    df['mean_salary'] = df.mean(axis=1)
    if resample is not None:
        df = df.resample(resample).mean()
    # df = df.drop('year', axis=1)
    df['year'] = df.index.year
    return df[['mean_salary', 'year']]


# def read_social_economic_index(path=work_david):
#     import pandas as pd
#     df = pd.read_excel(
#         path/'social_economic_index_statistical_areas_2015.xls',
#         skiprows=7)
#     df.drop(df.tail(3).index, inplace=True)
#     df.columns = ['city_code', 'NameHe', 'stat_code', 'population',
#                   'index_value', 'rank', 'cluster', 'NameEn']
#     return df
def plot_SEI_and_P2015(path=work_david):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_theme(style='ticks', font_scale=1.5)
    # fig, ax = plt.subplots(figsize=(10, 8))
    df = read_social_economic_index(path, return_stat=False)
    dfp = read_periphery_index(path)
    df['P2015'] = dfp['P2015_value']
    g = sns.jointplot(data=df, x="index2017", y="P2015", hue='Type',
                      s=15, height=10)
    g.plot_joint(sns.kdeplot, zorder=-1, levels=6)
    g.ax_joint.grid(True)
    g.fig.tight_layout()
    g.ax_joint.set_xlabel('SEI 2017')
    g.ax_joint.set_ylabel('Periphery Index 2015')
    corr_lc = df[df['Type']=='City/LC'].loc[:,['index2017','P2015']].corr()['P2015']['index2017']
    n_lc = len(df[df['Type']=='City/LC'].loc[:,['index2017','P2015']].dropna())
    n_rc = len(df[df['Type']=='RC'].loc[:,['index2017','P2015']].dropna())
    corr_rc = df[df['Type']=='RC'].loc[:,['index2017','P2015']].corr()['P2015']['index2017']
    textstr = '\n'.join([r'City/LC corr: {:.2f}, n={}'.format(corr_lc, n_lc),
    r'RC        corr: {:.2f}, n={}'.format(corr_rc, n_rc)])
    # text_str = 'n_LC={}, r_LC={:.1f}'.format(n_lc, corr_lc)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5, edgecolor='k')
    print(textstr)
    g.ax_joint.text(0.24, 0.9, textstr,
                    verticalalignment='top', horizontalalignment='center',
                    transform=g.ax_joint.transAxes, color='k', fontsize=18, bbox=props)

    # ax.legend(['dd','gg'])
    # ax.grid(True)
    # g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)
    # g.plot_marginals(sns.rugplot, color="r", height=-.15, clip_on=False)
    return g


def read_social_economic_index(path=work_david, return_stat=True):
    import pandas as pd
    SEI_cluster_dict = {1: 1,
                        2: 1,
                        3: 2,
                        4: 2,
                        5: 3,
                        6: 3,
                        7: 4,
                        8: 4,
                        9: 5,
                        10: 5}
    # first read statistical areas:
    df = pd.read_excel(
        path/'social_economic_index_stat_areas_2017.xlsx',
        skiprows=8)
    df.drop(df.tail(6).index, inplace=True)
    df.columns = ['muni_status', 'city_code', 'NameHe', 'NameEn', 'city_cluster2017', 'city_cluster2015',
                  'stat_code', 'pop2017', 'index_value2017', 'rank2017', 'cluster2017', 'cluster2015']
    df = df.set_index(['city_code', 'stat_code'])
    # add 2015 data:
    df2015 = pd.read_excel(
        path/'social_economic_index_statistical_areas_2015.xls',
        skiprows=7)
    df2015.drop(df2015.tail(3).index, inplace=True)
    df2015.columns = ['city_code', 'NameHe', 'stat_code', 'pop2015',
                      'index_value2015', 'rank2015', 'cluster2015', 'NameEn']
    df2015 = df2015.set_index(['city_code', 'stat_code'])
    df = pd.concat(
        [df, df2015[['index_value2015', 'rank2015', 'pop2015']]], axis=1)
    df = df.reset_index()
    df['pop2015'] = pd.to_numeric(df['pop2015'])
    df['cluster2015'] = pd.to_numeric(df['cluster2015'], errors='coerce')
    df['muni_status'] = pd.to_numeric(df['muni_status'])
    # now, read cities and settelments within regional councils:
    dfm = pd.read_excel(
        path/'social_economic_index_local_muni_2017.xlsx',
        skiprows=5)
    dfm.drop(dfm.tail(4).index, inplace=True)
    dfm.columns = ['muni_state', 'city_code', 'NameHe', 'pop2017', 'index2017',
                   'rank2017', 'cluster2017', 'rank2015', 'cluster2015', 'cluster_diff', 'NameEn']
    dfm['Type'] = 'City/LC'
    dfr = pd.read_excel(
        path/'social_economic_index_regional_councils_2017.xlsx',
        skiprows=10)
    dfr.drop(dfr.tail(9).index, inplace=True)
    dfr.columns = ['muni_state', 'RC_NameHe', 'RC_NameEn', 'RC_cluster2017', 'RC_cluster2015', 'city_code',
                   'NameHe', 'NameEn', 'locality_type', 'pop2017', 'index2017', 'rank2017', 'cluster2017', 'cluster2015']
    dfr['Type'] = 'RC'
    dff = pd.concat([dfm, dfr], axis=0)
    # stype = dff['Type']
    # dff = dff.loc[:, 'muni_state':'NameEn']
    # dff['Type'] = stype
    dff = dff[~dff['city_code'].isnull()]
    dff.set_index('city_code', inplace=True)
    dff['SEI2_cluster'] = dff['cluster2017'].map(SEI_cluster_dict)
    dff['RC_SEI2_cluster'] = dff['RC_cluster2017'].map(SEI_cluster_dict)
    if return_stat:
        return df
    else:
        return dff


def read_statistical_areas_gis_file(path=work_david):
    import geopandas as gpd
    gdf = gpd.read_file(path/'statisticalareas_demography2019.gdb')
    gdf.columns = ['city_code', 'stat11', 'city_stat11', 'NameHe', 'NameEn',
                   'stat11_unite', 'stat11_ref', 'main_function_code',
                   'main_function_text', 'religion_code', 'religion_text',
                   'pop_total', 'male_total', 'female_total', 'age_0-4',
                   'age_5-9', 'age_10-14', 'age_15-19', 'age_20-24',
                   'age_25-29', 'age_30-34', 'age_35-39', 'age_40-44',
                   'age_45-49', 'age_50-54', 'age_55-59', 'age_60-64',
                   'age_65-69', 'age_70-74', 'age_75-79', 'age_80-84',
                   'age_85-up', 'SHAPE_Length', 'SHAPE_Area', 'geometry']
    return gdf


def read_bycode_city_data(path=work_david):
    import pandas as pd
    df = pd.read_excel(path/'bycode2019.xlsx')
    # read index for region, district and area codes:
    ind_file = path/'index2019.xlsx'
    idf = pd.read_excel(ind_file, sheet_name='מחוז ונפה', skiprows=2)
    idf.columns = ['sub-region', 'region', 'district', 'region_code']
    region_dict = dict(
        zip(idf['region_code'].dropna().astype(int), idf['region'].dropna()))
    district_dict = {}
    district_dict[1] = 'ירושלים'
    district_dict[2] = 'הצפון'
    district_dict[3] = 'חיפה'
    district_dict[4] = 'המרכז'
    district_dict[5] = 'תל אביב'
    district_dict[6] = 'הדרום'
    district_dict[7] = 'יו"ש'
    district_dict_en = {1: 'Jerusalem', 2: 'North', 3: 'Haifa', 4: 'Center',
                     5: 'Tel-Aviv', 6: 'South', 7: 'J&S'}
    idf = pd.read_excel(ind_file, sheet_name='אזור טבעי ', skiprows=2)
    idf.columns = ['to_drop1', 'comments', 'natural_area',
                   'sub-district', 'district', 'natural_area_code']
    idf = idf[[x for x in idf.columns if 'to_drop' not in x]]
    natural_area_dict = dict(
        zip(idf['natural_area_code'].dropna().astype(int), idf['natural_area'].dropna()))

    df.columns = ['NameHe', 'city_code', 'transliteration', 'district_code',
                  'region_code', 'natural_area_code', 'municipal_state',
                  'metropolitan_association', 'city_religion', 'total_pop_2019',
                  'jews+other', 'just_jews', 'arabs', 'founding_year',
                  'city_flow_form', 'organization', 'coords', 'mean_height',
                  'planning_committee', 'police', 'year', 'NameEn', 'cluster']
    df['district'] = df['district_code'].map(district_dict)
    df['region'] = df['region_code'].map(region_dict)
    df['natural_area'] = df['natural_area_code'].map(natural_area_dict)
    df['district_EN'] = df['district_code'].map(district_dict_en)
    df = df.set_index('city_code')
    return df


def read_street_city_names(path=work_david, filter_neighborhoods=True,
                           fix_alley=True):
    """get street names, codes and city names and codes"""
    import pandas as pd
    df = pd.read_csv(path/'street_names_israel.csv', encoding="cp1255")
    df.columns = ['city_code', 'city_name', 'street_code', 'street_name']
    df['city_name'] = df['city_name'].str.replace(
        'תל אביב - יפו', 'תל אביב יפו')
    if filter_neighborhoods:
        df = df[df['street_code'] <= 5999]
    if fix_alley:
        df['street_name'] = df['street_name'].str.replace('סמ ', 'סמטת ')
    return df


def read_periphery_index(path=work_david):
    import pandas as pd
    P_cluster_dict = {1: 1,
                      2: 1,
                      3: 2,
                      4: 2,
                      5: 3,
                      6: 4,
                      7: 4,
                      8: 5,
                      9: 5,
                      10: 5}
    P_cluster_name = {1: 'Very Peripheral',
                      2: 'Peripheral',
                      3: 'In Between',
                      4: 'Centralized',
                      5: 'Very Centralized'}
    df = pd.read_excel(
        path/'peripheri_index_cbs_2015_local_cities.xls', skiprows=8)
    cols = ['municipal_status', 'city_code', 'NameHe', 'NameEn', 'Pop2015', 'TLV_proximity_value',
            'TLV_proximity_rank', 'PAI_value', 'PAI_rank', 'P2015_value', 'P2015_rank', 'P2015_cluster', 'P2004_cluster']
    df.columns = cols
    df = df.dropna()
    df['city_code'] = df['city_code'].astype(int)
    df.set_index('city_code', inplace=True)
    df_r = pd.read_excel(
        path/'peripheri_index_cbs_2015_regional_councils.xls', skiprows=6)
    cols = ['municipal_status', 'RC_nameHe', 'RC_nameEn', 'city_code', 'NameHe', 'NameEn', 'Sub_district', 'Pop2015', 'TLV_distance',
            'TLV_proximity_value', 'TLV_proximity_rank', 'PAI_value', 'PAI_rank', 'P2015_value', 'P2015_rank', 'P2015_cluster']
    df_r.columns = cols
    df_r = df_r.dropna()
    df_r.set_index('city_code', inplace=True)
    dff = pd.concat([df, df_r], axis=0)
    dff.index = dff.index.astype(int)
    City = dff.loc[(dff['municipal_status'] == 0) |
                   (dff['municipal_status'] == 99)].index
    dff.loc[City, 'Type'] = 'City/LC'
    RC = dff.loc[(dff['municipal_status'] != 0) & (
        dff['municipal_status'] != 99)].index
    dff.loc[RC, 'Type'] = 'RC'
    # now read whole regional counsils P2015 cluster:
    df_rc = pd.read_excel(
        path/'peripheri_index_cbs_2015_just_regional_councils.xls', skiprows=9)
    df_rc.columns = ['municipal_status', 'RC_nameHe', 'RC_nameEn', 'RC_P2015_cluster', 'city_code', 'NameHe',
                     'NameEn', 'Pop2015', 'Sub_district', 'TLV_distance', 'P2015_value', 'P2015_rank', 'P2015_cluster']
    df_rc.set_index('city_code', inplace=True)
    dff['RC_P2015_cluster'] = df_rc['RC_P2015_cluster']
    dff['P2_cluster'] = dff['P2015_cluster'].map(P_cluster_dict)
    dff['P2_cluster_name'] = dff['P2_cluster'].map(P_cluster_name)
    dff['RC_P2_cluster'] = dff['RC_P2015_cluster'].map(P_cluster_dict)
    dff['RC_P2_cluster_name'] = dff['RC_P2_cluster'].map(P_cluster_name)
    return dff


def read_building_starts_ends(path=work_david, filename='BuildingIL_1995-2000 (Shlomi).xlsx'):
    import pandas as pd
    df = pd.read_excel(path/filename, sheet_name='Raw')
    df['ID'] = df['ID'].astype(str).str.replace('יו"ש', 'J&S')
    df['ID'] = df['ID'].astype(str).str.replace('--', 'NRC')
    df['ID'] = df['ID'].astype(str).str.replace('אחר', 'Other')
    df.loc[:,'1room':'Total'] = df.loc[:,'1room':'Total'].astype(float)
    return df


def calculate_building_rates(bdf, phase='Begin', rooms='Total', fillna=True):
    import pandas as pd
    df = bdf[bdf['Phase'] == phase]
    df = df.groupby(['ID', 'Year'])[rooms].sum().unstack().T
    da = df.diff().T.to_xarray().to_array('time')
    da.name = 'BR'
    da.attrs['long_name'] = 'Building rate'
    da.attrs['units'] = 'BPY'
    da.attrs['rooms'] = rooms
    da.attrs['phase'] = phase
    da['time'] = pd.to_datetime(da['time'], format='%Y')
    if fillna:
        da = da.resample(time='MS').asfreq().sel(time=slice('1996', '2020'))
        da = da.interpolate_na('time', method='linear')
        da = da.rolling(time=6, center=True).mean()
    df = da.to_dataset('ID').to_dataframe()
    return df


def calculate_building_growth_rate_constant(bdf, eyear=2019, syear=2006, phase='End', savepath=None):
    import pandas as pd
    div = eyear - syear + 1
    ccs = bdf['ID'].unique()
    bdf = bdf[bdf['Phase'] == phase]
    rates = []
    cc_ind = []
    for cc in ccs:
        end_data = bdf[(bdf['ID'] == cc) & (bdf['Year'] == eyear)
                       ].loc[:, '1room':'Total'].reset_index(drop=True)
        start_data = bdf[(bdf['ID'] == cc) & (bdf['Year'] == syear)
                         ].loc[:, '1room':'Total'].reset_index(drop=True)
        # if end_data.empty:
            # print(cc, 'end')
        rate = (end_data - start_data) / div
        if len(rate) > 1:
            continue
        if (rate.empty) or (rate.T[0].isnull().sum() == 7):
            # print(cc)
            continue
        else:
            rates.append(rate)
            cc_ind.append(cc)
    df = pd.concat(rates, axis=0)
    df.index = cc_ind
    df.index.name = 'ID'
    if savepath is not None:
        filename = 'Building_{}_growth_rate_{}-{}.csv'.format(phase, syear, eyear)
        df.to_csv(savepath/filename, na_rep='None')
        print('{} was saved to {}.'.format(filename, savepath))
    return df


def read_boi_mortgage(path=work_david, filename='pribmash.xls',
                      rolling=6):
    import pandas as pd
    df = pd.read_excel(path/filename, skiprows=5, sheet_name='RESULT_OLD')
    df.columns = ['average', '>20', '17-20', '15-17', '12-15', '5-12', '<=5', 'from', 'due', 'drop1', 'drop2']
    data_cols = ['average', '>20', '17-20', '15-17', '12-15', '5-12', '<=5']
    df = df[[x for x in df.columns if 'drop' not in x]]
    df_old = df.dropna()
    df = pd.read_excel(path/filename, skiprows=5, sheet_name='RESULT')
    df.columns = ['average', '>20', '17-20', '15-17', '12-15', '5-12', '<=5', 'from', 'due', 'drop1', 'drop2']
    df = df[[x for x in df.columns if 'drop' not in x]]
    df = pd.concat([df, df_old], axis=0)
    df = df.set_index('from')
    df = df.sort_index()
    df['average'] = df['average'].astype(float)
    if rolling is not None:
        df = df.resample('MS').mean()
        df[data_cols] = df[data_cols].rolling(rolling, center=True).mean()
    return df


def read_apts_sold(path=work_david, filename='Apts_sold_2021-07.xls',
                   filename2='Total_apts_yearly_sales.xlsx'):
    import pandas as pd
    df = pd.read_excel(path/filename, skiprows=2)
    df.columns = ['trend', 'deseasonlized', 'original','month','year']
    for ind, row in df.copy().iterrows():
        if not pd.isnull(row['year']):
            good_year = row['year']
        df.at[ind, 'year'] = good_year
    df['dt'] = df['year'].astype(int).astype(str) + '-' + df['month'].astype(str)
    df['dt'] = pd.to_datetime(df['dt'], format='%Y-%m')
    df = df.set_index('dt')
    df = df.sort_index()
    df1 = pd.read_excel(path/filename2,skiprows=18,na_values='--')    # years = df.groupby('month')['year'].unique()[1]
    df1.columns = ['year', 'original2']
    df1.set_index(pd.to_datetime(df1['year'], format='%Y'), inplace=True)
    # df = pd.concat([df, df1], axis=1)
    # groups=df.groupby('month').groups
    # for mnth, inds in groups.items():
        # print(len(years), len(inds))
        # df.loc[inds, 'year'] = years
    return df1, df


def read_boi_interest(path=work_david, filename='bointcrh.xls'):
    import pandas as pd
    df = pd.read_excel(path/filename, header=2, sheet_name='גיליון1')
    df.columns = ['month', 'nominal_simple', 'effective', 'eff_monitar_loans',
                  'eff_monitar_deposits', 'eff_inter_bank_transfer', 'to_drop1', 'to_drop2']
    data_cols = ['nominal_simple', 'effective', 'eff_monitar_loans',
                 'eff_monitar_deposits', 'eff_inter_bank_transfer']
    # drop some cols:
    df = df[[x for x in df.columns if 'to_drop' not in x]]
    # take care of datetimes:
    df['mnth_type'] = [pd.to_datetime(x, errors='coerce') for x in df['month']]
    df1 = df[~df['mnth_type'].isnull()]
    ind = df1.index[0]
    df1.set_index(pd.to_datetime(df1['month']))
    df1 = df1.set_index(pd.to_datetime(df1['month']))
    df2 = df.iloc[0:ind]
    df2['date1'] = [x[0] for x in df2['month'].str.split('-')]
    df2['date2'] = [x[-1] for x in df2['month'].str.split('-')]

    df2['date1'] = pd.to_datetime(df2['date1'])
    df2['date2'] = pd.to_datetime(df2['date2'])
    df2_data = []
    for i, row in df2.iterrows():
        dt1 = pd.to_datetime(row['date1'], format='%Y-%m')
        dt2 = pd.to_datetime(row['date2'], format='%Y-%m')
        dtr = pd.date_range(dt1, dt2, freq='M')
        dfr = pd.DataFrame([row[data_cols]]*len(dtr), index=dtr)
        df2_data.append(dfr)
    dff = pd.concat(df2_data, axis=0)
    df = pd.concat([df1[data_cols], dff], axis=0)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    return df


def run_mortgage_interest_building_lag_analysis(bdf, mdf, irate='average', months=24,
                                                cities=None):
    import pandas as pd
    df = mdf[irate].to_frame()
    # shift mortgage interest rates earlier in time:
    for i in range(months):
        df['{}_{}'.format(irate, i+1)] = df[irate].shift(-i-1)
    # add city and find best corrlation:
    city_corr = {}
    city_lag = {}
    if cities is None:
        cities = bdf.columns
    for city in cities:
        df[city] = bdf[city]
        corr=df.corr()[city].abs()
        if corr.isnull().sum() > 1:
            print('No correlation for {}.'.format(city))
            df = df.drop(city, axis=1)
            continue
        lag_str=corr.drop(city,axis=0).idxmax()
        corr_max = corr.drop(city,axis=0).max()
        try:
            print(city, lag_str)
            lag = lag_str.split('_')[-1]
            lag = int(lag)
        except ValueError:
            lag = 0
        city_corr[city] = corr_max
        city_lag[city] = lag
        df = df.drop(city, axis=1)
    df=pd.concat([pd.Series(city_lag), pd.Series(city_corr)], axis=1)
    df.columns = ['month_lag', 'correlation']
    return df


def read_cbs_main_indicies(start_date='1997-01', end_date=None,
                           savepath=None):
    import pandas as pd
    # import xml.etree.ElementTree as ET
    import xmltodict
    import requests
    st_date = ''.join(start_date.split('-'))
    if end_date is not None:
        ed_date = ''.join(end_date.split('-'))
    else:
        ed_date = pd.Timestamp.now().strftime('%Y%m')
    url = 'https://api.cbs.gov.il/index/data/price_selected_b?StartDate={}&EndDate={}'.format(
        st_date, ed_date)
    r = requests.get(url)
    xmlDict = xmltodict.parse(r.content)
    cols = []
    data = []
    for item in xmlDict['indices']['ind']:
        cols = [x for x in item.keys()]
        data.append([x for x in item.values()])
    df = pd.DataFrame(data, columns=cols)
    df.index = pd.to_datetime(df['date'], format='%Y-%m')
    dates = pd.to_datetime(df['date'].unique(), format='%Y-%m')
    dff = pd.DataFrame(index=dates)
    names = {}
    for ind_code in df['code'].unique():
        dff[str(ind_code)] = df[df['code'] == ind_code]['index'].astype(float)
        dff[str(ind_code)+'_base'] = df[df['code'] == ind_code]['base']
        names[ind_code] = df[df['code'] == ind_code]['name'].unique().item()
    dff.index.name = 'time'
    ds = dff.to_xarray()
    for da in ds:
        try:
            ds[da].attrs['index_name'] = names[da]
        except KeyError:
            pass
    # root = ET.XML(r.content)
    # data = []
    # cols = []
    # for i, child in enumerate(root):
    #     data.append([subchild.text for subchild in child])
    #     cols.append(child.tag)
    # df = pd.DataFrame(data).T  # Write in DF and transpose it
    # df.columns = cols
    return ds

def read_yearly_inner_migration(path=work_david, filename='inner_migration_2015-2019_IL.xlsx'):
    import pandas as pd
    years = [2015, 2016, 2017, 2018, 2019]
    dfs = []
    for year in years:
        df = pd.read_excel(path/filename, sheet_name='{}'.format(year), skiprows=2)
        df.columns = ['city_type_code', 'city_code', 'NameHe', 'inflow', 'outflow', 'net']
        df['year'] = year
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    return df

def read_various_parameters(path=work_david, file='various_parameters.xlsx',
                            add_flow_rate_index='Inflow'):
    import pandas as pd
    import numpy as np
    import geopandas as gpd
    df = pd.read_excel(path / file)
    df = df.set_index('ID')
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['ITMX'], df['ITMY']))
    if add_flow_rate_index is not None:
        df['{}_rate_index'.format(add_flow_rate_index)] = np.log(df['{}2019'.format(add_flow_rate_index)]/df['{}2014'.format(add_flow_rate_index)])
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


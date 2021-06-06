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


def read_social_economic_index(path=work_david):
    import pandas as pd
    df = pd.read_excel(
        path/'social_economic_index_statistical_areas_2015.xls',
        skiprows=7)
    df.drop(df.tail(3).index, inplace=True)
    df.columns = ['city_code', 'NameHe', 'stat_code', 'population',
                  'index_value', 'rank', 'cluster', 'NameEn']
    return df


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
    df.columns = ['NameHe', 'city_code', 'transliteration', 'district',
                  'region', 'natural_area', 'municipal_state',
                  'metropolitan_association', 'city_religion', 'total_pop_2019',
                  'jews+other', 'just_jews', 'arabs', 'founding_year',
                  'city_flow_form', 'organization', 'coords', 'mean_height',
                  'planning_committee', 'police', 'year', 'NameEn', 'cluster']
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
    return dff


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
